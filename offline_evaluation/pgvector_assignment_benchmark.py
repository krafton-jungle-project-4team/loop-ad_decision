from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb

from app.decision.assignment_service import (
    AssignmentPageMatchResult,
    AssignmentPageMatcher,
    ExactAssignmentPageMatcher,
)
from app.decision.matcher import (
    ANN_CANDIDATE_LIMIT,
    ANN_QUERY_USER_BATCH_SIZE,
    VECTOR_DIM,
    MatchResult,
    SegmentCandidateReranker,
    SegmentVector,
    UserVector,
)
from app.decision.repositories import (
    PostgresExecutor,
    PsycopgPostgresExecutor,
    SegmentVectorRepository,
)
from offline_evaluation.assignment_corpus import (
    SUPPORTED_DISTRIBUTIONS,
    FrozenAssignmentCorpus,
)


PGVECTOR_BENCHMARK_FORMAT_VERSION = "loopad.pgvector-assignment-benchmark.v1"
PGVECTOR_BENCHMARK_SUMMARY_VERSION = (
    "loopad.pgvector-assignment-benchmark-summary.v1"
)
PRODUCTION_POLICY = "exact_only"
SMOKE_USER_COUNTS = (256,)
SMOKE_SEGMENT_COUNTS = (256,)
REPRESENTATIVE_USER_COUNTS = (10_000,)
OPTIONAL_FULL_USER_COUNTS = (100_000,)
REPRESENTATIVE_SEGMENT_COUNTS = (50, 256, 1_000, 5_000)
FULL_100K_COMMAND_TEMPLATE = (
    ".venv/bin/python scripts/benchmark_pgvector_segment_assignments.py "
    "--dsn 'postgresql://USER:PASSWORD@127.0.0.1:5432/DATABASE' "
    "--representative --include-100k"
)
LOCAL_POSTGRES_HOSTS = {"localhost", "127.0.0.1", "::1"}
BENCHMARK_PROJECT_ID = "pgvector_benchmark_project"
BENCHMARK_PROMOTION_ID = "pgvector_benchmark_promotion"
BENCHMARK_ANALYSIS_ID = "pgvector_benchmark_analysis"
BENCHMARK_VECTOR_VERSION = "pgvector-benchmark-v1"
BENCHMARK_SOURCE_PATHS = (
    "app/decision/assignment_service.py",
    "app/decision/matcher.py",
    "app/decision/repositories.py",
    "offline_evaluation/assignment_corpus.py",
    "offline_evaluation/pgvector_assignment_benchmark.py",
    "scripts/benchmark_pgvector_segment_assignments.py",
)


def explain_capturing_postgres_executor(delegate: PostgresExecutor) -> Any:
    """Return a closure-backed executor that captures one production ANN plan."""

    state = SimpleNamespace(
        capture_pending=False,
        explain_plan=None,
        explain_seconds=None,
        production_query_sha256=None,
    )

    def capture_next_ann_query() -> None:
        state.capture_pending = True
        state.explain_plan = None
        state.explain_seconds = None
        state.production_query_sha256 = None

    def fetchone(
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        return delegate.fetchone(query, params)

    def fetchall(
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        if state.capture_pending and _is_production_ann_query(query):
            started_at = time.perf_counter()
            explain_rows = delegate.fetchall(
                "EXPLAIN (ANALYZE false, COSTS true, VERBOSE false, "
                "BUFFERS false, FORMAT JSON) "
                + query,
                params,
            )
            state.explain_seconds = time.perf_counter() - started_at
            state.explain_plan = _extract_explain_plan(explain_rows)
            state.production_query_sha256 = hashlib.sha256(
                _canonical_sql(query).encode("utf-8")
            ).hexdigest()
            state.capture_pending = False
        return delegate.fetchall(query, params)

    def execute(
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        delegate.execute(query, params)

    state.capture_next_ann_query = capture_next_ann_query
    state.fetchone = fetchone
    state.fetchall = fetchall
    state.execute = execute
    return state


@contextmanager
def peak_rss_monitor() -> Any:
    """Sample absolute client-process RSS without requiring psutil."""

    state = SimpleNamespace(peak_rss_bytes=resource_peak_rss_bytes())
    stop = threading.Event()
    thread: threading.Thread | None = None
    try:
        psutil = __import__("psutil")
        process = psutil.Process(os.getpid())
    except ImportError:
        process = None

    def sample() -> None:
        while not stop.wait(0.005):
            try:
                state.peak_rss_bytes = max(
                    state.peak_rss_bytes,
                    int(process.memory_info().rss),
                )
            except Exception:
                return

    if process is not None:
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
    try:
        yield state
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1.0)
        state.peak_rss_bytes = max(
            state.peak_rss_bytes,
            resource_peak_rss_bytes(),
        )


def validate_local_postgres_dsn(dsn: str) -> str:
    normalized = dsn.strip()
    if not normalized:
        raise ValueError("a localhost PostgreSQL DSN is required")
    try:
        parameters = conninfo_to_dict(normalized)
    except Exception as exc:
        raise ValueError("invalid PostgreSQL DSN") from exc
    if parameters.get("service"):
        raise ValueError("benchmark DSN must not use a libpq service")
    for key in ("host", "hostaddr"):
        addresses = parameters.get(key)
        if addresses and not _all_addresses_are_local(addresses, key=key):
            raise ValueError("benchmark DSN must target localhost or a Unix socket")
    return normalized


def benchmark_matrix(
    *,
    representative: bool = False,
    include_100k: bool = False,
    distributions: Sequence[str] = SUPPORTED_DISTRIBUTIONS,
) -> tuple[tuple[int, int, str], ...]:
    if include_100k and not representative:
        raise ValueError("include_100k requires representative mode")
    normalized_distributions = tuple(dict.fromkeys(distributions))
    if not normalized_distributions:
        raise ValueError("at least one distribution is required")
    unsupported = set(normalized_distributions) - set(SUPPORTED_DISTRIBUTIONS)
    if unsupported:
        raise ValueError(
            "unsupported distribution: " + ", ".join(sorted(unsupported))
        )
    if representative:
        user_counts = REPRESENTATIVE_USER_COUNTS + (
            OPTIONAL_FULL_USER_COUNTS if include_100k else ()
        )
        segment_counts = REPRESENTATIVE_SEGMENT_COUNTS
    else:
        user_counts = SMOKE_USER_COUNTS
        segment_counts = SMOKE_SEGMENT_COUNTS
    return tuple(
        (user_count, segment_count, distribution)
        for user_count in user_counts
        for segment_count in segment_counts
        for distribution in normalized_distributions
    )


def run_pgvector_benchmark_case(
    dsn: str,
    corpus: FrozenAssignmentCorpus,
    *,
    timing_trial_count: int = 3,
    warmup_user_count: int = 16,
    repository_root: Path | None = None,
    connect: Callable[..., Any] = psycopg.connect,
) -> dict[str, Any]:
    local_dsn = validate_local_postgres_dsn(dsn)
    _validate_benchmark_inputs(
        corpus=corpus,
        timing_trial_count=timing_trial_count,
        warmup_user_count=warmup_user_count,
    )
    root = repository_root or Path(__file__).resolve().parents[1]
    code_identity_before = benchmark_code_identity(root)
    users, segment_vectors = production_vectors(corpus)
    connection = connect(local_dsn, autocommit=False)
    try:
        delegate = PsycopgPostgresExecutor(connection)
        database = database_identity(delegate)
        if database["pgvector_version"] is None:
            raise RuntimeError(
                "localhost PostgreSQL must have the vector extension installed"
            )
        preparation = prepare_temporary_segment_vectors(
            connection,
            corpus=corpus,
            segment_vectors=segment_vectors,
        )
        executor = explain_capturing_postgres_executor(delegate)
        repository = SegmentVectorRepository(executor)
        reranker = SegmentCandidateReranker()
        ann_matcher = AssignmentPageMatcher(
            segment_vector_repository=repository,
            reranker=reranker,
        )
        exact_matcher = ExactAssignmentPageMatcher(reranker=reranker)

        query_preparation_started_at = time.perf_counter()
        repository.configure_ann_search()
        executor.capture_next_ann_query()
        repository.list_ann_candidates_for_users(
            project_id=BENCHMARK_PROJECT_ID,
            promotion_id=BENCHMARK_PROMOTION_ID,
            analysis_id=BENCHMARK_ANALYSIS_ID,
            segment_vector_ids=[
                segment.segment_vector_id for segment in segment_vectors
            ],
            vector_version=BENCHMARK_VECTOR_VERSION,
            user_ids=[users[0].user_id],
            query_vectors=[users[0].vector_values],
            limit=ANN_CANDIDATE_LIMIT,
        )
        plan_probe_seconds = time.perf_counter() - query_preparation_started_at
        if executor.explain_plan is None:
            raise RuntimeError("production ANN query EXPLAIN was not captured")

        measurements = benchmark_production_matchers(
            exact_matcher=exact_matcher,
            ann_matcher=ann_matcher,
            users=users,
            segment_vectors=segment_vectors,
            timing_trial_count=timing_trial_count,
            warmup_user_count=warmup_user_count,
        )
        query_preparation_seconds = (
            plan_probe_seconds + measurements.pop("warmup_seconds")
        )
        preparation.update(
            {
                "query_preparation_seconds": query_preparation_seconds,
                "query_plan_probe_seconds": plan_probe_seconds,
                "query_explain_seconds": executor.explain_seconds,
                "query_preparation_definition": (
                    "one production repository query with EXPLAIN plus exact and "
                    "ANN matcher warmup; excluded from timing trials"
                ),
            }
        )
        code_identity_after = benchmark_code_identity(root)
        if (
            code_identity_before["benchmark_source_sha256"]
            != code_identity_after["benchmark_source_sha256"]
        ):
            raise RuntimeError("benchmark source changed during the benchmark")

        explain_json = json.dumps(
            executor.explain_plan,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        return {
            "format_version": PGVECTOR_BENCHMARK_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "production_policy": PRODUCTION_POLICY,
            "ann_activation_changed": False,
            "evidence_use": "offline_only_non_activating",
            "corpus_manifest": corpus.manifest.to_dict(),
            "benchmark_code_identity": code_identity_before,
            "database": {
                **database,
                "hnsw_explain_plan": executor.explain_plan,
                "hnsw_index_used": (
                    "benchmark_segment_vectors_hnsw_idx" in explain_json
                ),
                "production_query_sha256": executor.production_query_sha256,
            },
            "matcher_configuration": {
                "ann_candidate_limit": ANN_CANDIDATE_LIMIT,
                "ann_query_user_batch_size": ANN_QUERY_USER_BATCH_SIZE,
                "hnsw_ef_search": SegmentVectorRepository.HNSW_EF_SEARCH,
                "hnsw_max_scan_tuples": (
                    SegmentVectorRepository.HNSW_MAX_SCAN_TUPLES
                ),
                "vector_dimension": VECTOR_DIM,
                "timing_trial_count": timing_trial_count,
                "requested_warmup_user_count": warmup_user_count,
                "effective_warmup_user_count": min(
                    warmup_user_count,
                    len(users),
                ),
            },
            "preparation": preparation,
            **measurements,
            "execution_environment": execution_environment(),
        }
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()


def prepare_temporary_segment_vectors(
    connection: Any,
    *,
    corpus: FrozenAssignmentCorpus,
    segment_vectors: Sequence[SegmentVector],
) -> dict[str, Any]:
    table_started_at = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TEMP TABLE segment_vectors (
                segment_vector_id text PRIMARY KEY,
                project_id text NOT NULL,
                promotion_id text,
                promotion_run_id text,
                analysis_id text,
                segment_id text NOT NULL,
                vector_dim integer NOT NULL,
                vector_values jsonb NOT NULL,
                vector_version text NOT NULL,
                source text NOT NULL,
                embedding vector(64) NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            ) ON COMMIT DROP
            """
        )
    table_setup_seconds = time.perf_counter() - table_started_at

    load_started_at = time.perf_counter()
    rows = []
    for segment, values in zip(
        segment_vectors,
        corpus.segment_vectors,
        strict=True,
    ):
        vector_values = [float(value) for value in values]
        rows.append(
            (
                segment.segment_vector_id,
                BENCHMARK_PROJECT_ID,
                BENCHMARK_PROMOTION_ID,
                BENCHMARK_ANALYSIS_ID,
                segment.segment_id,
                VECTOR_DIM,
                Jsonb(vector_values),
                BENCHMARK_VECTOR_VERSION,
                "synthetic_pgvector_benchmark",
                vector_literal(vector_values),
            )
        )
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO segment_vectors (
                segment_vector_id,
                project_id,
                promotion_id,
                analysis_id,
                segment_id,
                vector_dim,
                vector_values,
                vector_version,
                source,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            rows,
        )
    corpus_load_seconds = time.perf_counter() - load_started_at

    index_started_at = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE INDEX benchmark_segment_vectors_hnsw_idx
            ON segment_vectors
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    hnsw_index_build_seconds = time.perf_counter() - index_started_at

    analyze_started_at = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute("ANALYZE segment_vectors")
    analyze_seconds = time.perf_counter() - analyze_started_at
    index_preparation_seconds = (
        table_setup_seconds
        + corpus_load_seconds
        + hnsw_index_build_seconds
        + analyze_seconds
    )
    return {
        "temporary_table_setup_seconds": table_setup_seconds,
        "corpus_load_seconds": corpus_load_seconds,
        "hnsw_index_build_seconds": hnsw_index_build_seconds,
        "analyze_seconds": analyze_seconds,
        "index_preparation_seconds": index_preparation_seconds,
        "index_preparation_definition": (
            "temporary table creation + corpus load + HNSW index build + ANALYZE; "
            "excluded from matcher timing trials"
        ),
        "temporary_object_policy": (
            "pg_temp segment_vectors with ON COMMIT DROP; transaction rolled back "
            "and connection closed after each case"
        ),
    }


def benchmark_production_matchers(
    *,
    exact_matcher: Any,
    ann_matcher: Any,
    users: Sequence[UserVector],
    segment_vectors: Sequence[SegmentVector],
    timing_trial_count: int,
    warmup_user_count: int,
    clock: Callable[[], float] = time.perf_counter,
    monitor_factory: Callable[[], Any] = peak_rss_monitor,
) -> dict[str, Any]:
    if timing_trial_count < 2:
        raise ValueError("timing_trial_count must be at least 2")
    if warmup_user_count < 0:
        raise ValueError("warmup_user_count must be non-negative")
    if not users or not segment_vectors:
        raise ValueError("benchmark users and segment vectors must not be empty")

    warmup_started_at = clock()
    effective_warmup = min(warmup_user_count, len(users))
    if effective_warmup:
        warmup_users = users[:effective_warmup]
        _match_page(exact_matcher, warmup_users, segment_vectors)
        _match_page(ann_matcher, warmup_users, segment_vectors)
    warmup_seconds = clock() - warmup_started_at

    matchers = {"exact": exact_matcher, "pgvector_hnsw": ann_matcher}
    samples: dict[str, list[dict[str, Any]]] = {
        matcher_name: [] for matcher_name in matchers
    }
    stable_results: dict[str, AssignmentPageMatchResult] = {}
    trial_schedule: list[tuple[str, ...]] = []
    matcher_names = tuple(matchers)
    for trial_index in range(timing_trial_count):
        rotation = trial_index % len(matcher_names)
        order = matcher_names[rotation:] + matcher_names[:rotation]
        trial_schedule.append(order)
        for matcher_name in order:
            started_at = clock()
            with monitor_factory() as monitor:
                result = _match_page(
                    matchers[matcher_name],
                    users,
                    segment_vectors,
                )
            elapsed_seconds = clock() - started_at
            if elapsed_seconds < 0:
                raise RuntimeError("benchmark clock moved backwards")
            signature = assignment_result_signature(result)
            if matcher_name in stable_results:
                if signature != assignment_result_signature(
                    stable_results[matcher_name]
                ):
                    raise RuntimeError(
                        f"{matcher_name} output changed across timing trials"
                    )
            else:
                stable_results[matcher_name] = result
            samples[matcher_name].append(
                {
                    "trial_index": trial_index + 1,
                    "elapsed_seconds": elapsed_seconds,
                    "users_per_second": (
                        len(users) / elapsed_seconds
                        if elapsed_seconds > 0
                        else None
                    ),
                    "peak_rss_bytes": int(monitor.peak_rss_bytes),
                }
            )

    exact_result = stable_results["exact"]
    ann_result = stable_results["pgvector_hnsw"]
    return {
        "warmup_seconds": warmup_seconds,
        "timing_trial_schedule": trial_schedule,
        "timing": {
            matcher_name: summarize_timing_samples(matcher_samples)
            for matcher_name, matcher_samples in samples.items()
        },
        "agreement": compare_assignment_results(exact_result, ann_result),
        "rescue": rescue_metrics(ann_result),
    }


def compare_assignment_results(
    exact: AssignmentPageMatchResult,
    candidate: AssignmentPageMatchResult,
) -> dict[str, Any]:
    exact_user_ids = set(exact.matches)
    candidate_user_ids = set(candidate.matches)
    if exact_user_ids != candidate_user_ids:
        raise ValueError("exact and candidate outputs contain different users")
    user_count = len(exact_user_ids)
    if user_count == 0:
        raise ValueError("agreement requires at least one user")

    segment_agreement_count = 0
    fallback_agreement_count = 0
    fallback_reason_agreement_count = 0
    score_agreement_count = 0
    assignment_agreement_count = 0
    false_fallback_count = 0
    false_nonfallback_count = 0
    for user_id in sorted(exact_user_ids):
        exact_match = exact.matches[user_id]
        candidate_match = candidate.matches[user_id]
        segment_equal = exact_match.segment_id == candidate_match.segment_id
        fallback_equal = exact_match.fallback == candidate_match.fallback
        reason_equal = exact_match.fallback_reason == candidate_match.fallback_reason
        score_equal = similarity_scores_equal(
            exact_match.similarity_score,
            candidate_match.similarity_score,
        )
        segment_agreement_count += int(segment_equal)
        fallback_agreement_count += int(fallback_equal)
        fallback_reason_agreement_count += int(reason_equal)
        score_agreement_count += int(score_equal)
        assignment_agreement_count += int(
            segment_equal and fallback_equal and reason_equal
        )
        false_fallback_count += int(
            candidate_match.fallback and not exact_match.fallback
        )
        false_nonfallback_count += int(
            exact_match.fallback and not candidate_match.fallback
        )

    return {
        "oracle": "ExactAssignmentPageMatcher",
        "candidate": "AssignmentPageMatcher+SegmentVectorRepository",
        "user_count": user_count,
        "assignment_agreement_count": assignment_agreement_count,
        "assignment_agreement_rate": assignment_agreement_count / user_count,
        "segment_agreement_rate": segment_agreement_count / user_count,
        "fallback_agreement_rate": fallback_agreement_count / user_count,
        "fallback_reason_agreement_rate": (
            fallback_reason_agreement_count / user_count
        ),
        "similarity_score_agreement_rate": score_agreement_count / user_count,
        "observed_disagreement_count": user_count - assignment_agreement_count,
        "observed_disagreement_rate": (
            (user_count - assignment_agreement_count) / user_count
        ),
        "false_fallback_count": false_fallback_count,
        "false_nonfallback_count": false_nonfallback_count,
        "score_tolerance": {"relative": 1e-9, "absolute": 1e-9},
    }


def rescue_metrics(result: AssignmentPageMatchResult) -> dict[str, Any]:
    user_count = len(result.matches)
    return {
        "ann_candidate_count": result.ann_candidate_count,
        "ann_query_user_count": result.ann_query_user_count,
        "ann_underfilled_user_count": result.ann_underfilled_user_count,
        "exact_rescue_user_count": result.exact_rescue_user_count,
        "exact_rescue_rate": (
            result.exact_rescue_user_count / user_count if user_count else None
        ),
        "exact_reranked_pair_count": result.exact_reranked_pair_count,
    }


def summarize_timing_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one timing sample is required")
    elapsed_ms = [float(sample["elapsed_seconds"]) * 1000.0 for sample in samples]
    throughput = [
        float(sample["users_per_second"])
        for sample in samples
        if sample["users_per_second"] is not None
    ]
    return {
        "trial_count": len(samples),
        "latency_p50_ms": linear_percentile(elapsed_ms, 50),
        "latency_p95_ms": linear_percentile(elapsed_ms, 95),
        "latency_p50_bootstrap_95_ci_ms": bootstrap_median_95_ci(elapsed_ms),
        "users_per_second_p50": (
            linear_percentile(throughput, 50) if throughput else None
        ),
        "users_per_second_p50_bootstrap_95_ci": (
            bootstrap_median_95_ci(throughput) if throughput else None
        ),
        "peak_rss_bytes": max(int(sample["peak_rss_bytes"]) for sample in samples),
        "peak_rss_definition": (
            "maximum absolute client-process RSS observed during matcher trials; "
            "PostgreSQL server RSS is excluded"
        ),
        "percentile_method": "linear_interpolation",
        "confidence_interval_method": (
            "deterministic_nonparametric_bootstrap_median_2000_resamples_seed_20260714"
        ),
        "trial_samples": list(samples),
    }


def summarize_benchmark_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    include_100k: bool,
) -> dict[str, Any]:
    if mode not in {"smoke", "representative"}:
        raise ValueError("unsupported pgvector benchmark mode")
    crossover_observations: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                str(case["corpus_manifest"]["distribution"]),
                int(case["corpus_manifest"]["user_count"]),
            )
            for case in cases
        }
    )
    for distribution, user_count in group_keys:
        grouped = sorted(
            (
                case
                for case in cases
                if case["corpus_manifest"]["distribution"] == distribution
                and case["corpus_manifest"]["user_count"] == user_count
            ),
            key=lambda case: int(case["corpus_manifest"]["segment_count"]),
        )
        faster_counts = [
            int(case["corpus_manifest"]["segment_count"])
            for case in grouped
            if case["timing"]["pgvector_hnsw"]["latency_p95_ms"]
            < case["timing"]["exact"]["latency_p95_ms"]
        ]
        evaluated_counts = [
            int(case["corpus_manifest"]["segment_count"]) for case in grouped
        ]
        crossover_observations.append(
            {
                "distribution": distribution,
                "user_count": user_count,
                "evaluated_segment_counts": evaluated_counts,
                "minimum_observed_faster_segment_count": (
                    min(faster_counts) if faster_counts else None
                ),
                "status": (
                    "observed_evidence_only"
                    if faster_counts
                    else "not_observed_non_blocking"
                ),
                "timing_comparison_basis": "matcher_latency_p95_ms",
                "production_policy_effect": "none_exact_only",
            }
        )
    full_100k_status = (
        "completed"
        if include_100k
        and any(
            int(case["corpus_manifest"]["user_count"]) == 100_000
            for case in cases
        )
        else "not_requested_non_blocking"
        if not include_100k
        else "not_completed_non_blocking"
    )
    full_100k_not_run_reason = (
        None
        if full_100k_status == "completed"
        else "resource_intensive_opt_in_not_requested"
        if not include_100k
        else "requested_matrix_did_not_complete"
    )
    return {
        "format_version": PGVECTOR_BENCHMARK_SUMMARY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "mode": mode,
        "production_policy": PRODUCTION_POLICY,
        "ann_activation_changed": False,
        "case_count": len(cases),
        "cases": list(cases),
        "crossover_observations": crossover_observations,
        "full_100k_status": full_100k_status,
        "full_100k_reproduction": {
            "command_template": FULL_100K_COMMAND_TEMPLATE,
            "not_run_reason": full_100k_not_run_reason,
        },
        "goal_blockers": [],
        "non_blocking_conditions": [
            "crossover_not_observed",
            "representative_or_100k_matrix_not_run",
        ],
    }


def benchmark_code_identity(repository_root: Path) -> dict[str, Any]:
    aggregate = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for relative_path in BENCHMARK_SOURCE_PATHS:
        payload = (repository_root / relative_path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        file_hashes[relative_path] = digest
        encoded_path = relative_path.encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(4, "big"))
        aggregate.update(encoded_path)
        aggregate.update(bytes.fromhex(digest))
    dirty = git_output(
        repository_root,
        "status",
        "--porcelain",
        "--",
        *BENCHMARK_SOURCE_PATHS,
    )
    return {
        "git_commit": git_output(repository_root, "rev-parse", "HEAD"),
        "git_tree": git_output(repository_root, "rev-parse", "HEAD^{tree}"),
        "git_branch": git_output(
            repository_root,
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ),
        "benchmark_sources_dirty": None if dirty is None else bool(dirty),
        "benchmark_source_sha256": aggregate.hexdigest(),
        "benchmark_source_files_sha256": file_hashes,
    }


def database_identity(db: PostgresExecutor) -> dict[str, Any]:
    row = db.fetchone(
        """
        SELECT
            current_setting('server_version') AS postgres_version,
            current_setting('server_version_num') AS postgres_version_num,
            (
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector'
            ) AS pgvector_version
        """
    )
    if row is None:
        raise RuntimeError("failed to read PostgreSQL version information")
    return {
        "postgres_version": str(row["postgres_version"]),
        "postgres_version_num": str(row["postgres_version_num"]),
        "pgvector_version": (
            None
            if row["pgvector_version"] is None
            else str(row["pgvector_version"])
        ),
    }


def production_vectors(
    corpus: FrozenAssignmentCorpus,
) -> tuple[tuple[UserVector, ...], tuple[SegmentVector, ...]]:
    users = tuple(
        UserVector(
            user_id=user_id,
            vector_dim=VECTOR_DIM,
            vector_values=[float(value) for value in values],
        )
        for user_id, values in zip(
            corpus.user_ids,
            corpus.user_vectors,
            strict=True,
        )
    )
    segments = tuple(
        SegmentVector(
            segment_vector_id=f"pgvector-benchmark-segvec-{index:06d}",
            segment_id=segment_id,
            vector_dim=VECTOR_DIM,
            embedding_values=[float(value) for value in values],
        )
        for index, (segment_id, values) in enumerate(
            zip(corpus.segment_ids, corpus.segment_vectors, strict=True)
        )
    )
    return users, segments


def assignment_result_signature(
    result: AssignmentPageMatchResult,
) -> tuple[Any, ...]:
    return (
        tuple(
            (
                user_id,
                match.segment_id,
                match.similarity_score,
                match.fallback,
                match.fallback_reason,
            )
            for user_id, match in sorted(result.matches.items())
        ),
        result.ann_candidate_count,
        result.exact_reranked_pair_count,
        result.ann_underfilled_user_count,
        result.exact_rescue_user_count,
        result.ann_query_user_count,
    )


def similarity_scores_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile values must not be empty")
    if quantile < 0 or quantile > 100:
        raise ValueError("quantile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def bootstrap_median_95_ci(
    values: Sequence[float],
    *,
    resample_count: int = 2_000,
    seed: int = 20_260_714,
) -> list[float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    if resample_count <= 0:
        raise ValueError("bootstrap resample_count must be positive")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("bootstrap values must be finite")
    random_source = random.Random(seed)
    medians = [
        linear_percentile(
            [
                normalized[random_source.randrange(len(normalized))]
                for _ in normalized
            ],
            50,
        )
        for _ in range(resample_count)
    ]
    return [
        linear_percentile(medians, 2.5),
        linear_percentile(medians, 97.5),
    ]


def write_json_report(
    payload: Mapping[str, Any],
    destination: Path,
    *,
    overwrite: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8") as output:
        output.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def execution_environment() -> dict[str, Any]:
    try:
        psutil_version = str(__import__("psutil").__version__)
    except ImportError:
        psutil_version = None
    return {
        "python_version": platform.python_version(),
        "psycopg_version": psycopg.__version__,
        "psutil_version": psutil_version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "process_id": os.getpid(),
        "database_location_policy": "localhost_or_unix_socket_only",
        "client_and_database_process_isolation": "not_isolated",
    }


def resource_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def vector_literal(values: Sequence[float]) -> str:
    if len(values) != VECTOR_DIM:
        raise ValueError(f"benchmark vectors must contain {VECTOR_DIM} values")
    numeric_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("benchmark vectors must be finite")
    return "[" + ",".join(format(value, ".9g") for value in numeric_values) + "]"


def git_output(repository_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _match_page(
    matcher: Any,
    users: Sequence[UserVector],
    segment_vectors: Sequence[SegmentVector],
) -> AssignmentPageMatchResult:
    return matcher.match_page(
        project_id=BENCHMARK_PROJECT_ID,
        promotion_id=BENCHMARK_PROMOTION_ID,
        analysis_id=BENCHMARK_ANALYSIS_ID,
        vector_version=BENCHMARK_VECTOR_VERSION,
        users=users,
        segment_vectors=segment_vectors,
    )


def _validate_benchmark_inputs(
    *,
    corpus: FrozenAssignmentCorpus,
    timing_trial_count: int,
    warmup_user_count: int,
) -> None:
    if corpus.manifest.dimension != VECTOR_DIM:
        raise ValueError(f"benchmark corpus dimension must be {VECTOR_DIM}")
    if corpus.manifest.user_count <= 0 or corpus.manifest.segment_count <= 0:
        raise ValueError("benchmark corpus must not be empty")
    if timing_trial_count < 2:
        raise ValueError("timing_trial_count must be at least 2")
    if warmup_user_count < 0:
        raise ValueError("warmup_user_count must be non-negative")


def _all_addresses_are_local(addresses: str, *, key: str) -> bool:
    for address in addresses.split(","):
        candidate = address.strip()
        if not candidate:
            continue
        if key == "host" and candidate.startswith("/"):
            continue
        if candidate not in LOCAL_POSTGRES_HOSTS:
            return False
    return True


def _is_production_ann_query(query: str) -> bool:
    normalized = _canonical_sql(query).lower()
    return (
        "cross join lateral" in normalized
        and "order by embedding <=>" in normalized
    )


def _canonical_sql(query: str) -> str:
    return " ".join(query.split())


def _extract_explain_plan(rows: Sequence[Mapping[str, Any]]) -> Any:
    if len(rows) != 1 or not rows[0]:
        raise RuntimeError("EXPLAIN did not return exactly one plan row")
    row = rows[0]
    for key in ("QUERY PLAN", "query_plan"):
        if key in row:
            return row[key]
    if len(row) == 1:
        return next(iter(row.values()))
    raise RuntimeError("EXPLAIN plan column was not found")
