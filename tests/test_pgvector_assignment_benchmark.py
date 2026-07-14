from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.decision.assignment_service import AssignmentPageMatchResult
from app.decision.matcher import MatchResult, SegmentVector, UserVector
from offline_evaluation.pgvector_assignment_benchmark import (
    BENCHMARK_SOURCE_PATHS,
    benchmark_code_identity,
    benchmark_matrix,
    benchmark_production_matchers,
    bootstrap_median_95_ci,
    compare_assignment_results,
    database_identity,
    explain_capturing_postgres_executor,
    linear_percentile,
    rescue_metrics,
    summarize_benchmark_cases,
    validate_local_postgres_dsn,
    vector_literal,
    write_json_report,
)


@pytest.mark.parametrize(
    "dsn",
    (
        "dbname=loopad host=localhost",
        "postgresql://loopad@127.0.0.1/loopad",
        "dbname=loopad host=/tmp",
        "dbname=loopad",
    ),
)
def test_localhost_dsn_guard_accepts_only_local_forms(dsn: str) -> None:
    assert validate_local_postgres_dsn(dsn) == dsn


@pytest.mark.parametrize(
    "dsn",
    (
        "dbname=loopad host=db.example.com",
        "dbname=loopad host=localhost,db.example.com",
        "dbname=loopad hostaddr=10.0.0.12",
        "service=loopad-dev",
    ),
)
def test_localhost_dsn_guard_rejects_remote_or_indirect_targets(dsn: str) -> None:
    with pytest.raises(ValueError, match="localhost|service"):
        validate_local_postgres_dsn(dsn)


def test_benchmark_matrix_defaults_to_smoke_and_keeps_100k_opt_in() -> None:
    smoke = benchmark_matrix()

    assert len(smoke) == 4
    assert {(users, segments) for users, segments, _ in smoke} == {(256, 256)}

    representative = benchmark_matrix(representative=True)
    assert len(representative) == 16
    assert {users for users, _, _ in representative} == {10_000}
    assert {segments for _, segments, _ in representative} == {
        50,
        256,
        1_000,
        5_000,
    }

    full = benchmark_matrix(representative=True, include_100k=True)
    assert len(full) == 32
    assert {users for users, _, _ in full} == {10_000, 100_000}
    with pytest.raises(ValueError, match="requires representative"):
        benchmark_matrix(include_100k=True)


def test_explain_executor_captures_the_real_production_query_shape() -> None:
    delegate = stub_postgres_executor()
    executor = explain_capturing_postgres_executor(delegate)
    production_query = """
        WITH query_users AS (SELECT 1)
        SELECT *
        FROM query_users q
        CROSS JOIN LATERAL (
            SELECT * FROM segment_vectors
            ORDER BY embedding <=> q.query_vector::vector
            LIMIT %s
        ) sv
        ON true
    """

    executor.capture_next_ann_query()
    assert executor.fetchall(production_query, (50,)) == []

    assert len(delegate.fetchall_calls) == 2
    assert delegate.fetchall_calls[0][0].startswith("EXPLAIN")
    assert delegate.fetchall_calls[0][0].endswith(production_query)
    assert delegate.fetchall_calls[1][0] == production_query
    assert executor.explain_plan == [
        {
            "Plan": {
                "Node Type": "Index Scan",
                "Index Name": "benchmark_segment_vectors_hnsw_idx",
            }
        }
    ]
    assert executor.explain_seconds is not None
    assert len(str(executor.production_query_sha256)) == 64


def test_benchmark_matchers_records_interleaved_trials_and_rescue() -> None:
    users = (user("user-a"), user("user-b"))
    segments = (segment("segment-a"),)
    exact_result = page_result(
        {
            "user-a": match("segment-a", 1.0),
            "user-b": match("segment-a", 1.0),
        },
        exact_pairs=2,
    )
    ann_result = page_result(
        {
            "user-a": match("segment-a", 1.0),
            "user-b": match("segment-a", 1.0),
        },
        ann_candidates=2,
        exact_pairs=2,
        ann_queries=2,
        rescues=1,
        underfilled=1,
    )
    exact_matcher = stub_matcher(exact_result)
    ann_matcher = stub_matcher(ann_result)

    report = benchmark_production_matchers(
        exact_matcher=exact_matcher,
        ann_matcher=ann_matcher,
        users=users,
        segment_vectors=segments,
        timing_trial_count=3,
        warmup_user_count=1,
        monitor_factory=stub_rss_monitor,
    )

    assert report["timing_trial_schedule"] == [
        ("exact", "pgvector_hnsw"),
        ("pgvector_hnsw", "exact"),
        ("exact", "pgvector_hnsw"),
    ]
    assert len(exact_matcher.user_counts) == 4
    assert len(ann_matcher.user_counts) == 4
    assert exact_matcher.user_counts[0] == 1
    assert ann_matcher.user_counts[0] == 1
    for matcher_name in ("exact", "pgvector_hnsw"):
        timing = report["timing"][matcher_name]
        assert timing["trial_count"] == 3
        assert timing["latency_p95_ms"] >= timing["latency_p50_ms"]
        assert timing["latency_p50_bootstrap_95_ci_ms"][0] <= (
            timing["latency_p50_ms"]
        )
        assert timing["latency_p50_bootstrap_95_ci_ms"][1] >= (
            timing["latency_p50_ms"]
        )
        assert timing["peak_rss_bytes"] == 123_456
        assert len(timing["trial_samples"]) == 3
    assert report["agreement"]["assignment_agreement_rate"] == 1.0
    assert report["rescue"]["exact_rescue_user_count"] == 1
    assert report["rescue"]["exact_rescue_rate"] == 0.5


def test_agreement_and_rescue_metrics_report_semantic_differences() -> None:
    exact = page_result(
        {
            "user-a": match("segment-a", 0.9),
            "user-b": MatchResult(
                segment_id="seg_existing_all",
                similarity_score=0.2,
                fallback=True,
                fallback_reason="below_threshold",
            ),
        },
        exact_pairs=4,
    )
    candidate = page_result(
        {
            "user-a": match("segment-a", 0.9 + 1e-12),
            "user-b": match("segment-b", 0.8),
        },
        ann_candidates=4,
        exact_pairs=4,
        ann_queries=2,
        rescues=1,
    )

    agreement = compare_assignment_results(exact, candidate)

    assert agreement["assignment_agreement_count"] == 1
    assert agreement["assignment_agreement_rate"] == 0.5
    assert agreement["similarity_score_agreement_rate"] == 0.5
    assert agreement["false_nonfallback_count"] == 1
    assert agreement["false_fallback_count"] == 0
    assert rescue_metrics(candidate) == {
        "ann_candidate_count": 4,
        "ann_query_user_count": 2,
        "ann_underfilled_user_count": 0,
        "exact_rescue_user_count": 1,
        "exact_rescue_rate": 0.5,
        "exact_reranked_pair_count": 4,
    }


def test_percentiles_and_vector_literal_do_not_need_numpy() -> None:
    assert linear_percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert linear_percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    confidence_interval = bootstrap_median_95_ci([1.0, 2.0, 3.0, 4.0])
    assert confidence_interval[0] <= 2.5 <= confidence_interval[1]
    literal = vector_literal([1.0] + [0.0] * 63)
    assert literal.startswith("[1,")
    assert literal.endswith("]")
    with pytest.raises(ValueError, match="64"):
        vector_literal([1.0])


def test_summary_keeps_crossover_and_missing_100k_non_blocking() -> None:
    cases = [
        summary_case(segment_count=50, exact_p95=10.0, ann_p95=20.0),
        summary_case(segment_count=256, exact_p95=30.0, ann_p95=20.0),
    ]

    summary = summarize_benchmark_cases(
        cases,
        mode="representative",
        include_100k=False,
    )

    assert summary["production_policy"] == "exact_only"
    assert summary["ann_activation_changed"] is False
    assert summary["goal_blockers"] == []
    assert summary["full_100k_status"] == "not_requested_non_blocking"
    assert summary["full_100k_reproduction"]["not_run_reason"] == (
        "resource_intensive_opt_in_not_requested"
    )
    assert "--representative --include-100k" in summary[
        "full_100k_reproduction"
    ]["command_template"]
    assert summary["crossover_observations"] == [
        {
            "distribution": "clustered",
            "user_count": 10_000,
            "evaluated_segment_counts": [50, 256],
            "minimum_observed_faster_segment_count": 256,
            "status": "observed_evidence_only",
            "timing_comparison_basis": "matcher_latency_p95_ms",
            "production_policy_effect": "none_exact_only",
        }
    ]


def test_versions_code_identity_and_json_are_bounded_and_serializable(
    tmp_path: Path,
) -> None:
    db = stub_postgres_executor(
        version_row={
            "postgres_version": "17.5",
            "postgres_version_num": "170005",
            "pgvector_version": "0.8.0",
        }
    )
    assert database_identity(db) == {
        "postgres_version": "17.5",
        "postgres_version_num": "170005",
        "pgvector_version": "0.8.0",
    }

    root = Path(__file__).resolve().parents[1]
    identity = benchmark_code_identity(root)
    assert len(identity["benchmark_source_sha256"]) == 64
    assert set(identity["benchmark_source_files_sha256"]) == set(
        BENCHMARK_SOURCE_PATHS
    )

    destination = tmp_path / "report.json"
    write_json_report(
        {
            "production_policy": "exact_only",
            "benchmark_source_sha256": identity["benchmark_source_sha256"],
        },
        destination,
    )
    assert json.loads(destination.read_text(encoding="utf-8"))[
        "production_policy"
    ] == "exact_only"
    with pytest.raises(FileExistsError):
        write_json_report({}, destination)


def stub_postgres_executor(
    *,
    version_row: dict[str, Any] | None = None,
) -> Any:
    state = SimpleNamespace(version_row=version_row, fetchall_calls=[])

    def fetchone(query: str, params: Any = ()) -> dict[str, Any] | None:
        del query, params
        return state.version_row

    def fetchall(query: str, params: Any = ()) -> list[dict[str, Any]]:
        state.fetchall_calls.append((query, params))
        if query.startswith("EXPLAIN"):
            return [
                {
                    "QUERY PLAN": [
                        {
                            "Plan": {
                                "Node Type": "Index Scan",
                                "Index Name": (
                                    "benchmark_segment_vectors_hnsw_idx"
                                ),
                            }
                        }
                    ]
                }
            ]
        return []

    def execute(query: str, params: Any = ()) -> None:
        del query, params

    state.fetchone = fetchone
    state.fetchall = fetchall
    state.execute = execute
    return state


def stub_matcher(result: AssignmentPageMatchResult) -> Any:
    state = SimpleNamespace(result=result, user_counts=[])

    def match_page(**kwargs: Any) -> AssignmentPageMatchResult:
        state.user_counts.append(len(kwargs["users"]))
        return state.result

    state.match_page = match_page
    return state


@contextmanager
def stub_rss_monitor() -> Any:
    yield SimpleNamespace(peak_rss_bytes=123_456)


def user(user_id: str) -> UserVector:
    return UserVector(
        user_id=user_id,
        vector_dim=64,
        vector_values=[1.0] + [0.0] * 63,
    )


def segment(segment_id: str) -> SegmentVector:
    return SegmentVector(
        segment_vector_id=f"segment-vector-{segment_id}",
        segment_id=segment_id,
        vector_dim=64,
        embedding_values=[1.0] + [0.0] * 63,
    )


def match(segment_id: str, score: float) -> MatchResult:
    return MatchResult(
        segment_id=segment_id,
        similarity_score=score,
        fallback=False,
        fallback_reason=None,
    )


def page_result(
    matches: dict[str, MatchResult],
    *,
    ann_candidates: int = 0,
    exact_pairs: int = 0,
    underfilled: int = 0,
    rescues: int = 0,
    ann_queries: int = 0,
) -> AssignmentPageMatchResult:
    return AssignmentPageMatchResult(
        matches=matches,
        ann_candidate_count=ann_candidates,
        exact_reranked_pair_count=exact_pairs,
        ann_underfilled_user_count=underfilled,
        exact_rescue_user_count=rescues,
        ann_query_user_count=ann_queries,
    )


def summary_case(
    *,
    segment_count: int,
    exact_p95: float,
    ann_p95: float,
) -> dict[str, Any]:
    return {
        "corpus_manifest": {
            "distribution": "clustered",
            "user_count": 10_000,
            "segment_count": segment_count,
        },
        "timing": {
            "exact": {"latency_p95_ms": exact_p95},
            "pgvector_hnsw": {"latency_p95_ms": ann_p95},
        },
    }
