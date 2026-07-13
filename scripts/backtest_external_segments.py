#!/usr/bin/env python3
"""Run the production segment candidate logic against external datasets."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.analysis.segment_performance import (  # noqa: E402
    build_segment_performance_predictor,
)
from app.config import DECISION_SERVICE_ID, Settings  # noqa: E402
from app.logging import (  # noqa: E402
    configure_logging,
    duration_ms,
    log,
    log_context_scope,
    now_ms,
)
from offline_evaluation.external_backtest import (  # noqa: E402
    ExternalBacktestConfig,
    ExternalBacktestError,
    run_external_backtest,
    write_external_backtest_artifacts,
)
from offline_evaluation.external_datasets import (  # noqa: E402
    ExternalAdapterConfig,
    load_external_dataset,
)


DEFAULT_SOURCE_DIRS = {
    "airbnb": Path(
        "artifacts/external-datasets/airbnb-recruiting-new-user-bookings"
    ),
    "booking-com": Path("artifacts/external-datasets/booking-com"),
    "synerise": Path("artifacts/external-datasets/synerise_dataset"),
}


def main() -> int:
    args = parse_args()
    configure_logging(_logging_settings())
    try:
        return run_validation(args)
    except (ExternalBacktestError, ValueError) as exc:
        log.warn(
            "external_backtest_invalid",
            {"datasetId": args.dataset, "err": exc},
        )
        return 2


@log_context_scope
def run_validation(args: argparse.Namespace) -> int:
    started_at = now_ms()
    source_dir = args.source_dir or DEFAULT_SOURCE_DIRS[args.dataset]
    output_dir = args.output_dir or _default_output_dir(args.dataset)
    log.assign_context(
        {
            "datasetId": args.dataset,
            "evaluationMode": "external_validation",
        }
    )
    log.info(
        "started",
        {
            "sourceDir": source_dir,
            "profilePoolLimit": args.profile_pool_limit,
            "sampleModulo": args.sample_modulo,
            "sampleRemainder": args.sample_remainder,
        },
    )
    adapter_config = ExternalAdapterConfig(
        profile_pool_limit=args.profile_pool_limit,
        max_scenarios=args.max_scenarios,
        min_scenario_users=args.min_scenario_users,
        sample_modulo=args.sample_modulo,
        sample_remainder=args.sample_remainder,
        include_checksum=not args.skip_checksum,
        cutoff=args.cutoff,
        lookback_days=args.lookback_days,
        outcome_days=args.outcome_days,
    )
    bundle = load_external_dataset(
        args.dataset,
        source_dir,
        config=adapter_config,
    )
    predictor = build_segment_performance_predictor(args.model_path)
    run = run_external_backtest(
        bundle.cases,
        config=ExternalBacktestConfig(
            max_suggested_segments=args.max_segments,
            min_sample_size=args.min_sample_size,
        ),
        performance_predictor=predictor,
    )
    if not run.results:
        raise ExternalBacktestError(
            "external backtest produced no candidates; increase the profile "
            "pool or lower sample thresholds"
        )
    artifacts = write_external_backtest_artifacts(
        run,
        manifest=bundle.manifest,
        output_dir=output_dir,
        model_metadata=predictor.metadata(),
    )
    log.info(
        "external_backtest_artifacts_created",
        {
            "scenarioCount": run.summary["scenario_count"],
            "candidateResultCount": run.summary["candidate_result_count"],
            "rankOneBeatsBaselineRate": run.summary[
                "rank_one_beats_baseline_rate"
            ],
            "outputDir": output_dir,
            "summaryPath": artifacts["summary"],
            "reportPath": artifacts["report"],
        },
    )
    log.info(
        "completed",
        {
            "datasetId": args.dataset,
            "durationMs": duration_ms(started_at),
            "outcome": "success",
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate LoopAd segment recommendations on external datasets.",
    )
    parser.add_argument("dataset", choices=tuple(DEFAULT_SOURCE_DIRS))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-pool-limit", type=positive_int, default=1000)
    parser.add_argument("--max-scenarios", type=positive_int, default=3)
    parser.add_argument("--max-segments", type=positive_int, default=3)
    parser.add_argument("--min-sample-size", type=positive_int, default=20)
    parser.add_argument("--min-scenario-users", type=positive_int, default=20)
    parser.add_argument("--sample-modulo", type=positive_int, default=1)
    parser.add_argument("--sample-remainder", type=nonnegative_int, default=0)
    parser.add_argument(
        "--cutoff",
        type=parse_datetime,
        default=datetime(2022, 11, 10, tzinfo=UTC),
        help="Synerise outcome cutoff in ISO-8601 format.",
    )
    parser.add_argument("--lookback-days", type=positive_int, default=90)
    parser.add_argument("--outcome-days", type=positive_int, default=28)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()
    if args.sample_remainder >= args.sample_modulo:
        parser.error("--sample-remainder must be smaller than --sample-modulo")
    return args


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _default_output_dir(dataset_id: str) -> Path:
    run_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path("artifacts")
        / "external-segment-backtest"
        / dataset_id
        / f"validation-{run_at}"
    )


def _logging_settings() -> Settings:
    return Settings(
        env="local-backtest",
        service_id=DECISION_SERVICE_ID,
        port=1,
        internal_api_key="unused",
        aurora_host="unused",
        aurora_port=1,
        aurora_database="unused",
        aurora_username="unused",
        aurora_password="unused",
        clickhouse_url="unused",
        clickhouse_database="unused",
        clickhouse_username="unused",
        clickhouse_password="unused",
        data_storage_bucket="unused",
        genai_assets_base_prefix="unused",
        openai_api_key="unused",
        gemini_api_key="unused",
    )


if __name__ == "__main__":
    sys.exit(main())
