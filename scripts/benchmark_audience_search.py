#!/usr/bin/env python3
"""Run and synthesize the offline Audience V2 ANN search experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import load_settings  # noqa: E402
from app.db import create_clickhouse_client, create_postgres_connection  # noqa: E402
from app.internal.schemas import UserBehaviorVectorBuildRequest  # noqa: E402
from app.internal.user_behavior_vectors import (  # noqa: E402
    HOTEL_BEHAVIOR_V2,
    UserBehaviorVectorBatchService,
    UserBehaviorVectorBuildRepository,
    _build_hotel_behavior_v2_insert_sql,
    _clickhouse_datetime,
)
from offline_evaluation.ann_search_benchmark import (  # noqa: E402
    BenchmarkManifest,
    CellRunConfig,
    LiveAnnSearchBenchmark,
    full_hnsw_grid,
)
from offline_evaluation.ann_search_cohort import (  # noqa: E402
    AnnBenchmarkCohortPreparer,
    CohortPreparation,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_EXPERIMENT_VERSION,
    ScaleCohortScope,
)
from offline_evaluation.ann_search_experiment import (  # noqa: E402
    BenchmarkPhase,
    build_phase_report,
    CacheMode,
    CORPUS_SIZES,
    EXPERIMENT_VERSION,
    HnswSettings,
    SearchPlan,
    experiment_manifest_template,
    load_observations,
    synthesize_policy,
    validate_implementation_fixtures,
    write_synthesis_artifacts,
)
from offline_evaluation.ann_search_macro import (  # noqa: E402
    MacroMode,
    MacroObservation,
    build_macro_report,
    load_macro_observations,
    write_macro_observations,
)
from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    AppendOnlyJsonl,
)


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def main() -> int:
    args = parse_args()
    if args.command == "template":
        return write_template(args)
    if args.command == "synthesize":
        return synthesize(args)
    if args.command == "phase-report":
        return phase_report(args)
    if args.command == "validate-policy":
        return validate_policy(args)
    if args.command == "macro-report":
        return macro_report(args)
    load_dotenv(args.env_file, override=False)
    settings = load_settings()
    if args.command == "run-macro":
        _require_local_benchmark_databases(settings)
        if not args.confirm_disposable_postgres:
            raise ValueError(
                "run-macro requires --confirm-disposable-postgres"
            )
        return run_macro(args, settings=settings)
    if args.command in {
        "prepare-vectors",
        "prepare-cohort",
        "ground-truth",
        "run-cell",
    }:
        _require_local_benchmark_databases(settings)
    if args.command == "prepare-vectors" and not args.confirm_local_clickhouse_write:
        raise ValueError(
            "prepare-vectors requires --confirm-local-clickhouse-write"
        )
    if args.command == "prepare-cohort":
        if not args.confirm_empty_disposable_postgres:
            raise ValueError(
                "prepare-cohort requires --confirm-empty-disposable-postgres"
            )
    if args.command == "run-cell":
        if not args.confirm_disposable_postgres:
            raise ValueError(
                "run-cell requires --confirm-disposable-postgres because it "
                "executes expensive EXPLAIN ANALYZE queries and temp-table work"
            )
    connection = create_postgres_connection(settings)
    clickhouse = create_clickhouse_client(settings)
    try:
        if args.command == "prepare-vectors":
            return prepare_vectors(args, clickhouse=clickhouse)
        if args.command == "prepare-cohort":
            return prepare_cohort(args, connection=connection, clickhouse=clickhouse)
        manifest = BenchmarkManifest.load(args.manifest)
        benchmark = LiveAnnSearchBenchmark(
            postgres_connection=connection,
            clickhouse=clickhouse,
            scale_cohort_scope=_scale_scope_from_args(args, manifest),
        )
        if args.command == "preflight":
            print(json.dumps(benchmark.preflight(manifest), indent=2, default=str))
            return 0
        if args.command == "ground-truth":
            summary = benchmark.export_ground_truth(
                manifest=manifest,
                scenario=manifest.require_scenario(args.scenario_id),
                output_path=args.output,
                reference_sample_size=args.reference_sample_size,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "run-cell":
            return run_cell(args, benchmark=benchmark, manifest=manifest)
        raise ValueError(f"unsupported command: {args.command}")
    finally:
        connection.close()
        close = getattr(clickhouse, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser(
        "template", help="write the versioned experiment grid template"
    )
    template.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser(
        "preflight", help="validate one prepared cohort without changing it"
    )
    _add_connection_args(preflight)
    preflight.add_argument("--manifest", type=Path, required=True)

    ground_truth = subparsers.add_parser(
        "ground-truth",
        help="export exact full-corpus ranks and final membership as CSV.gz",
    )
    _add_connection_args(ground_truth)
    ground_truth.add_argument("--manifest", type=Path, required=True)
    ground_truth.add_argument("--scenario-id", required=True)
    ground_truth.add_argument(
        "--reference-sample-size", type=positive_int, default=50_000
    )
    ground_truth.add_argument("--output", type=Path, required=True)

    vectors = subparsers.add_parser(
        "prepare-vectors",
        help="build the fixed-window production hotel_behavior.v2 source corpus",
    )
    _add_connection_args(vectors)
    vectors.add_argument("--project-id", required=True)
    vectors.add_argument(
        "--vector-version", default=HOTEL_BEHAVIOR_V2, choices=(HOTEL_BEHAVIOR_V2,)
    )
    vectors.add_argument(
        "--window-end",
        type=parse_utc_datetime,
        default=parse_utc_datetime("2015-01-01T00:00:00Z"),
    )
    vectors.add_argument("--window-days", type=positive_int, default=730)
    vectors.add_argument("--output", type=Path, required=True)
    vectors.add_argument(
        "--clickhouse-max-threads",
        type=positive_int,
        help="offline source-build session override; production SQL is unchanged",
    )
    vectors.add_argument(
        "--clickhouse-external-group-by-bytes",
        type=positive_int,
        help="spill threshold for the offline full-source vector GROUP BY",
    )
    vectors.add_argument(
        "--build-shard-count",
        type=positive_int,
        default=1,
        help=(
            "offline-only deterministic user shards used to bound production "
            "vector-build peak memory"
        ),
    )
    vectors.add_argument(
        "--build-checkpoint",
        type=Path,
        help="append-only per-shard checkpoint; required when shard count is > 1",
    )
    vectors.add_argument("--confirm-local-clickhouse-write", action="store_true")

    prepare = subparsers.add_parser(
        "prepare-cohort",
        help="load one deterministic nested cohort into an empty Data Contract DB",
    )
    _add_connection_args(prepare)
    prepare.add_argument("--project-id", required=True)
    prepare.add_argument("--vector-version", default="hotel_behavior.v2")
    prepare.add_argument("--manifest-hash", required=True)
    prepare.add_argument(
        "--window-start",
        type=parse_utc_datetime,
        default=parse_utc_datetime("2013-01-01T00:00:00Z"),
    )
    prepare.add_argument(
        "--window-end",
        type=parse_utc_datetime,
        default=parse_utc_datetime("2015-01-01T00:00:00Z"),
    )
    prepare.add_argument(
        "--source-revision-cutoff",
        type=parse_utc_datetime,
        required=True,
        help="stable ingestion snapshot for user_behavior_vector_revisions",
    )
    prepare.add_argument("--cohort-size", type=int, choices=CORPUS_SIZES, required=True)
    prepare.add_argument("--cohort-seed", default="expedia-ann-v1")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--confirm-empty-disposable-postgres", action="store_true")

    run = subparsers.add_parser(
        "run-cell", help="run one scenario against one prepared cohort database"
    )
    _add_connection_args(run)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--scenario-id", required=True)
    run.add_argument(
        "--phase",
        choices=[item.value for item in BenchmarkPhase],
        default=BenchmarkPhase.SCREENING.value,
    )
    run.add_argument(
        "--cache-mode",
        choices=[item.value for item in CacheMode],
        default=CacheMode.WARM.value,
    )
    run.add_argument("--warmups", type=nonnegative_int, default=5)
    run.add_argument("--repetitions", type=positive_int, default=10)
    run.add_argument(
        "--sample-sizes",
        type=parse_positive_ints,
        default=(2_000, 5_000, 10_000, 20_000, 50_000),
    )
    run.add_argument(
        "--requested-k",
        type=parse_positive_ints,
        default=(),
        help="comma-separated explicit K values; default derives the full K grid",
    )
    run.add_argument(
        "--hnsw-grid",
        choices=("baseline", "full"),
        default="baseline",
    )
    run.add_argument(
        "--hnsw",
        action="append",
        type=parse_hnsw,
        default=[],
        metavar="EF:MODE:MAX_SCAN",
        help="explicit HNSW tuple; repeat to add multiple settings",
    )
    run.add_argument("--random-seed", type=int, default=20260718)
    run.add_argument(
        "--plan",
        action="append",
        choices=[item.value for item in SearchPlan],
        default=[],
        help="plan to execute; repeat to select multiple plans",
    )
    run.add_argument("--exclusion-ratio", type=unit_interval, default=0.0)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--append", action="store_true")
    run.add_argument("--diagnostics-dir", type=Path)
    run.add_argument("--confirm-disposable-postgres", action="store_true")

    macro = subparsers.add_parser(
        "run-macro",
        help="run the real three-segment prepare-many workload",
    )
    _add_connection_args(macro)
    macro.add_argument("--manifest", type=Path, required=True)
    macro.add_argument("--scenario-id", action="append", required=True)
    macro.add_argument(
        "--mode", choices=[item.value for item in MacroMode], required=True
    )
    macro.add_argument("--policy", type=Path)
    macro.add_argument("--sample-size", type=positive_int, default=10_000)
    macro.add_argument(
        "--concurrency", type=parse_positive_ints, default=(1, 4)
    )
    macro.add_argument("--warmup-seconds", type=positive_int, default=120)
    macro.add_argument("--duration-seconds", type=positive_int, default=900)
    macro.add_argument("--output", type=Path, required=True)
    macro.add_argument("--append", action="store_true")
    macro.add_argument("--confirm-disposable-postgres", action="store_true")

    synthesis = subparsers.add_parser(
        "synthesize", help="generate policy and implementation fixtures from JSONL"
    )
    synthesis.add_argument("--input", type=Path, required=True)
    synthesis.add_argument("--scenario-manifest", type=Path, required=True)
    synthesis.add_argument("--dataset-hash", required=True)
    synthesis.add_argument("--environment-json", type=Path, required=True)
    synthesis.add_argument(
        "--macro-input",
        type=Path,
        help="measured current_runtime and candidate_policy macro JSONL",
    )
    synthesis.add_argument("--output-dir", type=Path, required=True)

    report = subparsers.add_parser(
        "phase-report",
        help="select screening survivors or per-cell HNSW tuning winners",
    )
    report.add_argument("--input", type=Path, required=True)
    report.add_argument(
        "--phase",
        choices=(BenchmarkPhase.SCREENING.value, BenchmarkPhase.TUNING.value),
        required=True,
    )
    report.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate-policy", help="replay generated implementation fixtures"
    )
    validate.add_argument("--policy", type=Path, required=True)
    validate.add_argument("--fixtures", type=Path, required=True)

    macro_summary = subparsers.add_parser(
        "macro-report", help="summarize and gate three-segment workload JSONL"
    )
    macro_summary.add_argument("--input", type=Path, required=True)
    macro_summary.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--scale-series-id")
    parser.add_argument("--scale-cohort-size", type=positive_int)
    parser.add_argument("--reference-sample-seed")


def _scale_scope_from_args(
    args: argparse.Namespace,
    manifest: BenchmarkManifest,
) -> ScaleCohortScope | None:
    values = (
        getattr(args, "scale_series_id", None),
        getattr(args, "scale_cohort_size", None),
        getattr(args, "reference_sample_seed", None),
    )
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError(
            "scale v2 requires --scale-series-id, --scale-cohort-size, and "
            "--reference-sample-seed together"
        )
    if manifest.experiment_version != SCALE_EXPERIMENT_VERSION:
        raise ValueError("scale scope arguments require a benchmark v2 manifest")
    return ScaleCohortScope(
        scale_series_id=str(values[0]),
        cohort_size=int(values[1]),
        reference_sample_seed=str(values[2]),
    )


def write_template(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(experiment_manifest_template(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def prepare_vectors(args: argparse.Namespace, *, clickhouse: Any) -> int:
    query_settings = {
        key: value
        for key, value in (
            ("max_threads", args.clickhouse_max_threads),
            (
                "max_bytes_before_external_group_by",
                args.clickhouse_external_group_by_bytes,
            ),
        )
        if value is not None
    }
    if query_settings:
        clickhouse = _ClickHouseSettingsClient(clickhouse, query_settings)
    if args.build_shard_count > 1 and args.build_checkpoint is None:
        raise ValueError("sharded vector build requires --build-checkpoint")
    repository: UserBehaviorVectorBuildRepository
    if args.build_shard_count > 1:
        repository = _ShardedVectorBuildRepository(
            clickhouse,
            shard_count=args.build_shard_count,
            checkpoint_path=args.build_checkpoint,
        )
    else:
        repository = UserBehaviorVectorBuildRepository(clickhouse)
    result = UserBehaviorVectorBatchService(
        repository=repository,
        now=args.window_end,
    ).build(
        UserBehaviorVectorBuildRequest(
            project_id=args.project_id,
            vector_version=args.vector_version,
            window_days=args.window_days,
        )
    )
    expected_window_start = args.window_end - timedelta(days=args.window_days)
    if result.window_start != expected_window_start or result.window_end != args.window_end:
        raise RuntimeError("production vector builder did not honor the fixed window")
    payload = {
        "project_id": result.project_id,
        "vector_version": result.vector_version,
        "vector_dim": result.vector_dim,
        "processed_user_count": result.processed_user_count,
        "vector_generation_id": result.vector_generation_id,
        "manifest_hash": result.manifest_hash,
        "source_revision_cutoff": result.source_revision_cutoff.isoformat(),
        "window_start": result.window_start.isoformat(),
        "window_end": result.window_end.isoformat(),
        "status": result.status,
        "clickhouse_query_settings": query_settings,
        "build_shard_count": args.build_shard_count,
        "build_checkpoint": (
            str(args.build_checkpoint) if args.build_checkpoint is not None else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


_RAW_EVENT_SHARD_ANCHOR = """AND validation_status = 'valid'"""


def _inject_user_shard_predicate(query: str) -> str:
    """Add an offline user partition without changing production vector formulas."""
    if query.count(_RAW_EVENT_SHARD_ANCHOR) != 1:
        raise RuntimeError("production vector SQL shard anchor changed")
    return query.replace(
        _RAW_EVENT_SHARD_ANCHOR,
        _RAW_EVENT_SHARD_ANCHOR
        + "\n                      AND modulo(cityHash64(user_id), "
        + "{build_shard_count:UInt64}) = {build_shard_index:UInt64}",
        1,
    )


class _ShardedVectorBuildRepository(UserBehaviorVectorBuildRepository):
    """Checkpointed, memory-bounded executor for the production per-user SQL."""

    def __init__(
        self,
        client: Any,
        *,
        shard_count: int,
        checkpoint_path: Path,
    ) -> None:
        super().__init__(client)
        if shard_count <= 1:
            raise ValueError("sharded repository requires at least two shards")
        self._shard_count = shard_count
        self._checkpoint_path = checkpoint_path

    def insert_raw_event_user_vectors(
        self,
        *,
        project_id: str,
        vector_version: str,
        source: str,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        if vector_version != HOTEL_BEHAVIOR_V2:
            raise ValueError("sharded build is only supported for hotel_behavior.v2")
        expected = self._raw_user_counts(
            project_id=project_id,
            window_start=window_start,
            window_end=window_end,
        )
        completed = self._completed_shards(
            project_id=project_id,
            vector_version=vector_version,
            window_start=window_start,
            window_end=window_end,
        )
        checkpoint = AppendOnlyJsonl(
            self._checkpoint_path,
            identity_fields=("build_shard_count", "build_shard_index"),
        )
        query = _inject_user_shard_predicate(_build_hotel_behavior_v2_insert_sql())
        base_parameters: dict[str, Any] = {
            "project_id": project_id,
            "vector_dim": self.VECTOR_DIM,
            "vector_version": vector_version,
            "source": source,
            "window_start": _clickhouse_datetime(window_start),
            "window_end": _clickhouse_datetime(window_end),
            "build_shard_count": self._shard_count,
        }
        for shard_index in range(self._shard_count):
            expected_count = expected.get(shard_index, 0)
            actual = self._materialized_counts(
                project_id=project_id,
                vector_version=vector_version,
                window_start=window_start,
                window_end=window_end,
                shard_index=shard_index,
            )
            if shard_index in completed:
                self._require_materialized_count(
                    shard_index=shard_index,
                    expected_count=expected_count,
                    actual=actual,
                )
                print(
                    json.dumps({"vector_build_shard": shard_index, "status": "resumed"}),
                    flush=True,
                )
                continue
            status = "validated_existing_after_interruption"
            if actual == (0, 0):
                parameters = {**base_parameters, "build_shard_index": shard_index}
                self._execute_insert(query, parameters)
                actual = self._materialized_counts(
                    project_id=project_id,
                    vector_version=vector_version,
                    window_start=window_start,
                    window_end=window_end,
                    shard_index=shard_index,
                )
                status = "insert_command_succeeded"
            self._require_materialized_count(
                shard_index=shard_index,
                expected_count=expected_count,
                actual=actual,
            )
            checkpoint.append(
                (
                    {
                        "experiment_version": SCALE_EXPERIMENT_VERSION,
                        "project_id": project_id,
                        "vector_version": vector_version,
                        "window_start": window_start.astimezone(UTC).isoformat(),
                        "window_end": window_end.astimezone(UTC).isoformat(),
                        "build_shard_count": self._shard_count,
                        "build_shard_index": shard_index,
                        "expected_user_count": expected_count,
                        "vector_row_count": actual[0],
                        "revision_row_count": actual[1],
                        "status": status,
                    },
                )
            )
            print(
                json.dumps(
                    {
                        "vector_build_shard": shard_index,
                        "build_shard_count": self._shard_count,
                        "user_count": expected_count,
                        "status": status,
                    }
                ),
                flush=True,
            )

    def _raw_user_counts(
        self,
        *,
        project_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[int, int]:
        result = self._client.query(
            """
            SELECT
                modulo(cityHash64(user_id), {build_shard_count:UInt64}) AS shard,
                uniqExact(user_id) AS user_count
            FROM raw_events
            WHERE project_id = {project_id:String}
              AND validation_status = 'valid'
              AND event_time >= toDateTime64(
                  parseDateTimeBestEffort({window_start:String}), 3, 'UTC'
              )
              AND event_time < toDateTime64(
                  parseDateTimeBestEffort({window_end:String}), 3, 'UTC'
              )
            GROUP BY shard
            ORDER BY shard
            """,
            parameters={
                "build_shard_count": self._shard_count,
                "project_id": project_id,
                "window_start": _clickhouse_datetime(window_start),
                "window_end": _clickhouse_datetime(window_end),
            },
        )
        rows = list(result.named_results())
        counts = {int(row["shard"]): int(row["user_count"]) for row in rows}
        if sum(counts.values()) <= 0:
            raise RuntimeError("sharded vector build source is empty")
        return counts

    def _materialized_counts(
        self,
        *,
        project_id: str,
        vector_version: str,
        window_start: datetime,
        window_end: datetime,
        shard_index: int,
    ) -> tuple[int, int]:
        parameters = {
            "project_id": project_id,
            "vector_version": vector_version,
            "window_start": _clickhouse_datetime(window_start),
            "window_end": _clickhouse_datetime(window_end),
            "build_shard_count": self._shard_count,
            "build_shard_index": shard_index,
        }
        result = self._client.query(
            """
            SELECT source_table, row_count
            FROM
            (
                SELECT 'vectors' AS source_table, count() AS row_count
                FROM user_behavior_vectors
                WHERE project_id = {project_id:String}
                  AND vector_version = {vector_version:String}
                  AND window_start = toDateTime64(
                      parseDateTimeBestEffort({window_start:String}), 3, 'UTC'
                  )
                  AND window_end = toDateTime64(
                      parseDateTimeBestEffort({window_end:String}), 3, 'UTC'
                  )
                  AND modulo(cityHash64(user_id), {build_shard_count:UInt64})
                      = {build_shard_index:UInt64}
                UNION ALL
                SELECT 'revisions' AS source_table, count() AS row_count
                FROM user_behavior_vector_revisions
                WHERE project_id = {project_id:String}
                  AND vector_version = {vector_version:String}
                  AND window_start = toDateTime64(
                      parseDateTimeBestEffort({window_start:String}), 3, 'UTC'
                  )
                  AND window_end = toDateTime64(
                      parseDateTimeBestEffort({window_end:String}), 3, 'UTC'
                  )
                  AND modulo(cityHash64(user_id), {build_shard_count:UInt64})
                      = {build_shard_index:UInt64}
            )
            """,
            parameters=parameters,
        )
        rows = {row["source_table"]: int(row["row_count"]) for row in result.named_results()}
        return rows.get("vectors", 0), rows.get("revisions", 0)

    def _completed_shards(
        self,
        *,
        project_id: str,
        vector_version: str,
        window_start: datetime,
        window_end: datetime,
    ) -> set[int]:
        if not self._checkpoint_path.exists():
            return set()
        expected_identity = {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "project_id": project_id,
            "vector_version": vector_version,
            "window_start": window_start.astimezone(UTC).isoformat(),
            "window_end": window_end.astimezone(UTC).isoformat(),
            "build_shard_count": self._shard_count,
        }
        completed: set[int] = set()
        for line_number, line in enumerate(
            self._checkpoint_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise RuntimeError(f"blank vector checkpoint row {line_number}")
            row = json.loads(line)
            for key, value in expected_identity.items():
                if row.get(key) != value:
                    raise RuntimeError(
                        f"vector checkpoint row {line_number} has mismatched {key}"
                    )
            shard_index = int(row["build_shard_index"])
            if shard_index in completed:
                raise RuntimeError("vector checkpoint contains duplicate shard")
            completed.add(shard_index)
        return completed

    @staticmethod
    def _require_materialized_count(
        *,
        shard_index: int,
        expected_count: int,
        actual: tuple[int, int],
    ) -> None:
        required = (expected_count, expected_count)
        if actual != required:
            raise RuntimeError(
                f"vector shard {shard_index} expected {required}, observed {actual}"
            )


class _ClickHouseSettingsClient:
    """Inject offline resource settings without changing production SQL/service."""

    def __init__(self, client: Any, settings: Mapping[str, int]) -> None:
        self._client = client
        self._settings = dict(settings)

    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
        settings: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._client.query(
            query,
            parameters=parameters,
            settings={**self._settings, **dict(settings or {})},
            **kwargs,
        )

    def command(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
        settings: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._client.command(
            query,
            parameters=parameters,
            settings={**self._settings, **dict(settings or {})},
            **kwargs,
        )


def prepare_cohort(
    args: argparse.Namespace,
    *,
    connection: Any,
    clickhouse: Any,
) -> int:
    config = CohortPreparation(
        project_id=args.project_id,
        vector_version=args.vector_version,
        manifest_hash=args.manifest_hash,
        window_start=args.window_start,
        window_end=args.window_end,
        source_revision_cutoff=args.source_revision_cutoff,
        cohort_size=args.cohort_size,
        cohort_seed=args.cohort_seed,
    )
    result = AnnBenchmarkCohortPreparer(
        postgres_connection=connection,
        clickhouse=clickhouse,
    ).prepare(config)
    payload = {
        "preparation": {
            "project_id": config.project_id,
            "vector_version": config.vector_version,
            "manifest_hash": config.manifest_hash,
            "window_start": config.window_start.isoformat(),
            "window_end": config.window_end.isoformat(),
            "source_revision_cutoff": config.source_revision_cutoff.isoformat(),
            "cohort_size": config.cohort_size,
            "cohort_seed": config.cohort_seed,
        },
        "result": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def run_cell(
    args: argparse.Namespace,
    *,
    benchmark: LiveAnnSearchBenchmark,
    manifest: BenchmarkManifest,
) -> int:
    hnsw = tuple(args.hnsw)
    if not hnsw:
        hnsw = (
            full_hnsw_grid()
            if args.hnsw_grid == "full"
            else (HnswSettings(100, "strict_order", 20_000),)
        )
    observations = benchmark.run_scenario(
        manifest=manifest,
        scenario=manifest.require_scenario(args.scenario_id),
        config=CellRunConfig(
            phase=BenchmarkPhase(args.phase),
            cache_mode=CacheMode(args.cache_mode),
            warmups=args.warmups,
            repetitions=args.repetitions,
            sample_sizes=tuple(args.sample_sizes),
            hnsw_settings=hnsw,
            requested_k_values=tuple(args.requested_k),
            plans=(
                tuple(SearchPlan(value) for value in args.plan)
                if args.plan
                else tuple(SearchPlan)
            ),
            exclusion_ratio=args.exclusion_ratio,
            random_seed=args.random_seed,
        ),
        diagnostics_dir=args.diagnostics_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.to_dict(), sort_keys=True))
            handle.write("\n")
    return 0


def run_macro(args: argparse.Namespace, *, settings: Any) -> int:
    manifest = BenchmarkManifest.load(args.manifest)
    if len(args.scenario_id) != 3 or len(set(args.scenario_id)) != 3:
        raise ValueError("run-macro requires exactly three distinct --scenario-id")
    scenarios = tuple(
        manifest.require_scenario(value) for value in args.scenario_id
    )
    if any(item.scenario_set.value != "confirmation" for item in scenarios):
        raise ValueError("run-macro requires confirmation scenarios")
    mode = MacroMode(args.mode)
    policy = _load_json_object(args.policy) if args.policy else None
    if mode == MacroMode.CANDIDATE_POLICY and policy is None:
        raise ValueError("candidate_policy mode requires --policy")
    if mode == MacroMode.CURRENT_RUNTIME and policy is not None:
        raise ValueError("current_runtime mode must not receive --policy")

    observations: list[MacroObservation] = []
    for concurrency in args.concurrency:
        observations.extend(
            _run_macro_phase(
                settings=settings,
                manifest=manifest,
                scenarios=scenarios,
                mode=mode,
                policy=policy,
                sample_size=args.sample_size,
                concurrency=concurrency,
                duration_seconds=args.warmup_seconds,
                measured=False,
            )
        )
        observations.extend(
            _run_macro_phase(
                settings=settings,
                manifest=manifest,
                scenarios=scenarios,
                mode=mode,
                policy=policy,
                sample_size=args.sample_size,
                concurrency=concurrency,
                duration_seconds=args.duration_seconds,
                measured=True,
            )
        )
    write_macro_observations(args.output, observations, append=args.append)
    return 0


def _run_macro_phase(
    *,
    settings: Any,
    manifest: BenchmarkManifest,
    scenarios: Any,
    mode: MacroMode,
    policy: Mapping[str, Any] | None,
    sample_size: int,
    concurrency: int,
    duration_seconds: int,
    measured: bool,
) -> list[MacroObservation]:
    barrier = threading.Barrier(concurrency)

    def worker() -> list[MacroObservation]:
        barrier.wait()
        connection = create_postgres_connection(settings)
        clickhouse = create_clickhouse_client(settings)
        benchmark = LiveAnnSearchBenchmark(
            postgres_connection=connection,
            clickhouse=clickhouse,
        )
        output: list[MacroObservation] = []
        try:
            deadline = time.monotonic() + duration_seconds
            while time.monotonic() < deadline or not output:
                started = time.perf_counter_ns()
                try:
                    output.append(
                        benchmark.run_macro_request(
                            manifest=manifest,
                            scenarios=scenarios,
                            mode=mode,
                            sample_size=sample_size,
                            concurrency=concurrency,
                            measured=measured,
                            policy=policy,
                        )
                    )
                except Exception as exc:  # preserve errors as benchmark evidence
                    connection.rollback()
                    output.append(
                        MacroObservation(
                            experiment_version=EXPERIMENT_VERSION,
                            mode=mode,
                            concurrency=concurrency,
                            measured=measured,
                            scenario_ids=tuple(
                                item.scenario_id for item in scenarios
                            ),
                            duration_ms=(
                                time.perf_counter_ns() - started
                            )
                            / 1_000_000,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        finally:
            connection.close()
            close = getattr(clickhouse, "close", None)
            if callable(close):
                close()
        return output

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        return [item for future in futures for item in future.result()]


def macro_report(args: argparse.Namespace) -> int:
    report = build_macro_report(load_macro_observations(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def synthesize(args: argparse.Namespace) -> int:
    manifest_payload = _load_json_object(args.scenario_manifest)
    if str(manifest_payload.get("experiment_version")) != EXPERIMENT_VERSION:
        raise ValueError("scenario manifest version is invalid")
    environment = _load_json_object(args.environment_json)
    macro = None
    if args.macro_input is not None:
        macro = build_macro_report(load_macro_observations(args.macro_input))
        if not macro["passed"]:
            raise ValueError("macro benchmark gate did not pass")
    result = synthesize_policy(
        load_observations(args.input),
        vector_version=str(manifest_payload["vector_version"]),
        manifest_hash=str(manifest_payload["manifest_hash"]),
        dataset_hash=args.dataset_hash,
        scenario_set_hash=_sha256_file(args.scenario_manifest),
        environment=environment,
        status="candidate" if macro is not None else "preliminary",
    )
    paths = write_synthesis_artifacts(result, output_dir=args.output_dir)
    if macro is not None:
        macro_path = args.output_dir / "macro-benchmark-report.json"
        macro_path.write_text(
            json.dumps(macro, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths = {**paths, "macro": macro_path}
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


def phase_report(args: argparse.Namespace) -> int:
    report = build_phase_report(
        load_observations(args.input),
        phase=BenchmarkPhase(args.phase),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def validate_policy(args: argparse.Namespace) -> int:
    policy = _load_json_object(args.policy)
    fixtures_payload = _load_json_object(args.fixtures)
    fixtures = fixtures_payload.get("fixtures")
    if not isinstance(fixtures, list):
        raise ValueError("fixtures file must contain a fixtures array")
    validate_implementation_fixtures(policy, fixtures)
    print(json.dumps({"status": "valid", "fixture_count": len(fixtures)}))
    return 0


def parse_hnsw(value: str) -> HnswSettings:
    try:
        ef_search, iterative_scan, max_scan_tuples = value.split(":", maxsplit=2)
        return HnswSettings(
            int(ef_search), iterative_scan, int(max_scan_tuples)
        )
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "HNSW settings must use EF:strict_order|relaxed_order:MAX_SCAN"
        ) from exc


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed.astimezone(UTC)


def _require_local_benchmark_databases(settings: Any) -> None:
    if settings.aurora_host not in LOCAL_HOSTS and not settings.aurora_host.startswith("/"):
        raise ValueError("ANN benchmark PostgreSQL must be local/disposable")
    clickhouse_host = urlparse(settings.clickhouse_url).hostname
    if clickhouse_host not in LOCAL_HOSTS:
        raise ValueError("ANN benchmark ClickHouse must be local/disposable")


def _load_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
