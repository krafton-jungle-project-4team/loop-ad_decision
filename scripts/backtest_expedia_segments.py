#!/usr/bin/env python3
"""Prepare and run a leakage-safe Expedia segment recommendation backtest.

Examples:
    python3 scripts/backtest_expedia_segments.py prepare \
        --train-csv /path/to/expedia-hotel-recommendations/train.csv
    python3 scripts/backtest_expedia_segments.py smoke
    python3 scripts/backtest_expedia_segments.py run \
        --start-cutoff 2014-01-01 \
        --end-cutoff 2014-12-01 \
        --user-sample-modulo 1
    python3 scripts/backtest_expedia_segments.py holdout \
        --user-sample-modulo 1
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import os
from pathlib import Path
import sys
from typing import Any

import clickhouse_connect
from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.analysis.expedia_backtest import (  # noqa: E402
    ClickHouseExpediaBacktestRepository,
    ExpediaBacktestConfig,
    ExpediaBacktestError,
    ExpediaSegmentBacktestService,
    monthly_cutoffs,
    run_temporal_holdout_backtest,
    validate_source_window,
    write_backtest_artifacts,
    write_temporal_holdout_artifacts,
)
from app.config import DECISION_SERVICE_ID, Settings  # noqa: E402
from app.logging import (  # noqa: E402
    configure_logging,
    duration_ms,
    log,
    log_context_scope,
    now_ms,
)


DEFAULT_SOURCE_TABLE = "expedia_hotel_events"
DEFAULT_PROJECT_ID = "expedia_backtest"


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file, override=False)
    connection = resolve_connection(args)
    configure_logging(_logging_settings(connection))
    try:
        return run_command(args, connection)
    except (ExpediaBacktestError, ValueError) as exc:
        log.warn("backtest_invalid", {"err": exc, "mode": args.command})
        return 2


@log_context_scope
def run_command(args: argparse.Namespace, connection: dict[str, Any]) -> int:
    started_at = now_ms()
    log.assign_context({"projectId": DEFAULT_PROJECT_ID})
    log.info(
        "started",
        {
            "mode": args.command,
            "sourceTable": args.source_table,
            "database": connection["database"],
        },
    )
    client = clickhouse_connect.get_client(
        interface="http",
        dsn=connection["url"],
        database=connection["database"],
        username=connection["username"],
        password=connection["password"],
        connect_timeout=10,
        send_receive_timeout=7200,
    )
    repository = ClickHouseExpediaBacktestRepository(
        client,
        source_table=args.source_table,
        project_id=DEFAULT_PROJECT_ID,
    )
    if args.command == "prepare":
        row_count = repository.load_train_csv(args.train_csv, replace=args.replace)
        stats = repository.source_stats()
        log.info(
            "expedia_source_prepared",
            {
                "rowCount": row_count,
                "userCount": stats.user_count,
                "bookingRowRate": stats.booking_row_rate,
                "firstEventAt": stats.first_event_at,
                "lastEventAt": stats.last_event_at,
            },
        )
        log.info(
            "completed",
            {
                "mode": args.command,
                "rowCount": row_count,
                "durationMs": duration_ms(started_at),
            },
        )
        return 0

    repository.ensure_source_table()
    config = backtest_config(args)
    if args.command == "holdout":
        training_cutoffs = monthly_cutoffs(
            args.train_start_cutoff,
            args.train_end_cutoff,
        )
        holdout_cutoffs = monthly_cutoffs(
            args.start_cutoff,
            args.end_cutoff,
        )
        stats = repository.source_stats()
        validate_source_window(
            stats,
            cutoffs=[*training_cutoffs, *holdout_cutoffs],
            lookback_days=config.lookback_days,
            outcome_days=config.outcome_days,
        )
        temporal_run = run_temporal_holdout_backtest(
            repository,
            config=config,
            training_cutoffs=training_cutoffs,
            holdout_cutoffs=holdout_cutoffs,
        )
        output_dir = args.output_dir or default_output_dir(args.command)
        artifacts = write_temporal_holdout_artifacts(
            temporal_run,
            output_dir=output_dir,
            source_stats=stats,
            config=config,
        )
        log.info(
            "temporal_holdout_artifacts_created",
            {
                "trainingResultCount": len(temporal_run.training_run.results),
                "holdoutResultCount": len(temporal_run.holdout_run.results),
                "outputDir": output_dir,
                "reportPath": artifacts["report"],
                "summaryPath": artifacts["summary"],
                "modelPath": artifacts["model"],
            },
        )
        log.info(
            "completed",
            {
                "mode": args.command,
                "trainingResultCount": len(temporal_run.training_run.results),
                "holdoutResultCount": len(temporal_run.holdout_run.results),
                "durationMs": duration_ms(started_at),
            },
        )
        return 0

    cutoffs = resolve_cutoffs(args)
    stats = repository.source_stats()
    validate_source_window(
        stats,
        cutoffs=cutoffs,
        lookback_days=config.lookback_days,
        outcome_days=config.outcome_days,
    )
    run = ExpediaSegmentBacktestService(repository, config=config).run(cutoffs)
    if not run.results:
        raise ExpediaBacktestError(
            "backtest produced no segment results; lower --min-scenario-users or "
            "--user-sample-modulo, or verify the source date range"
        )
    output_dir = args.output_dir or default_output_dir(args.command)
    artifacts = write_backtest_artifacts(
        run,
        output_dir=output_dir,
        source_stats=stats,
        config=config,
    )
    log.info(
        "backtest_artifacts_created",
        {
            "scenarioResultCount": len(run.results),
            "skippedScenarioCount": len(run.skipped_scenarios),
            "outputDir": output_dir,
            "reportPath": artifacts["report"],
            "summaryPath": artifacts["summary"],
            "resultsPath": artifacts["results"],
        },
    )
    log.info(
        "completed",
        {
            "mode": args.command,
            "scenarioResultCount": len(run.results),
            "durationMs": duration_ms(started_at),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest current LoopAd segment candidate generation with past Expedia "
            "behavior and future booking labels."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create/validate the ClickHouse source table and stream train.csv into it.",
    )
    add_connection_arguments(prepare)
    prepare.add_argument("--train-csv", type=Path, required=True)
    prepare.add_argument(
        "--replace",
        action="store_true",
        help="Truncate an existing non-empty source table before loading.",
    )

    smoke = subparsers.add_parser(
        "smoke",
        help="Run one fast cutoff with deterministic 5%% user sampling.",
    )
    add_connection_arguments(smoke)
    add_backtest_arguments(smoke, smoke=True)
    smoke.add_argument("--cutoff", type=parse_date, default=date(2014, 10, 1))

    run = subparsers.add_parser(
        "run",
        help="Run monthly rolling backtests and write CSV/JSON/Markdown artifacts.",
    )
    add_connection_arguments(run)
    add_backtest_arguments(run, smoke=False)
    run.add_argument(
        "--start-cutoff", type=parse_date, default=date(2014, 1, 1)
    )
    run.add_argument(
        "--end-cutoff", type=parse_date, default=date(2014, 12, 1)
    )

    holdout = subparsers.add_parser(
        "holdout",
        help=(
            "Fit contextual booking calibration on 2013 windows and evaluate "
            "predictions and ranking on untouched 2014 windows."
        ),
    )
    add_connection_arguments(holdout)
    add_backtest_arguments(holdout, smoke=False)
    holdout.add_argument(
        "--train-start-cutoff",
        type=parse_date,
        default=date(2013, 5, 1),
    )
    holdout.add_argument(
        "--train-end-cutoff",
        type=parse_date,
        default=date(2013, 12, 1),
    )
    holdout.add_argument(
        "--start-cutoff",
        type=parse_date,
        default=date(2014, 1, 1),
    )
    holdout.add_argument(
        "--end-cutoff",
        type=parse_date,
        default=date(2014, 12, 1),
    )
    return parser.parse_args()


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--clickhouse-url", default=None)
    parser.add_argument("--clickhouse-database", default=None)
    parser.add_argument("--clickhouse-username", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)


def add_backtest_arguments(
    parser: argparse.ArgumentParser,
    *,
    smoke: bool,
) -> None:
    parser.add_argument("--lookback-days", type=positive_int, default=90)
    parser.add_argument("--outcome-days", type=positive_int, default=30)
    parser.add_argument(
        "--max-scenarios",
        type=positive_int,
        default=2 if smoke else 3,
    )
    parser.add_argument("--max-segments", type=positive_int, default=3)
    parser.add_argument("--min-sample-size", type=positive_int, default=2)
    parser.add_argument("--profile-pool-limit", type=positive_int, default=1000)
    parser.add_argument(
        "--min-scenario-users", type=positive_int, default=10 if smoke else 20
    )
    parser.add_argument(
        "--user-sample-modulo",
        type=positive_int,
        default=20,
        help="20 uses a deterministic 5%% user sample; 1 uses all users.",
    )
    parser.add_argument("--user-sample-remainder", type=nonnegative_int, default=0)
    parser.add_argument(
        "--season",
        choices=("none", "spring", "summer", "fall", "winter"),
        default="none",
    )
    parser.add_argument("--output-dir", type=Path, default=None)


def resolve_connection(args: argparse.Namespace) -> dict[str, str]:
    return {
        "url": args.clickhouse_url
        or os.environ.get("LOOPAD_CLICKHOUSE_URL", "http://localhost:18123"),
        "database": args.clickhouse_database
        or os.environ.get("LOOPAD_CLICKHOUSE_DATABASE", "default"),
        "username": args.clickhouse_username
        or os.environ.get("LOOPAD_CLICKHOUSE_USERNAME", "default"),
        "password": args.clickhouse_password
        if args.clickhouse_password is not None
        else os.environ.get("LOOPAD_CLICKHOUSE_PASSWORD", ""),
    }


def backtest_config(args: argparse.Namespace) -> ExpediaBacktestConfig:
    return ExpediaBacktestConfig(
        lookback_days=args.lookback_days,
        outcome_days=args.outcome_days,
        max_scenarios_per_cutoff=args.max_scenarios,
        max_suggested_segments=args.max_segments,
        min_sample_size=args.min_sample_size,
        profile_pool_limit=args.profile_pool_limit,
        min_scenario_users=args.min_scenario_users,
        user_sample_modulo=args.user_sample_modulo,
        user_sample_remainder=args.user_sample_remainder,
        season=None if args.season == "none" else args.season,
    )


def resolve_cutoffs(args: argparse.Namespace) -> list[datetime]:
    if args.command == "smoke":
        return [datetime.combine(args.cutoff, datetime.min.time(), tzinfo=UTC)]
    return monthly_cutoffs(args.start_cutoff, args.end_cutoff)


def default_output_dir(command: str) -> Path:
    run_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "expedia-segment-backtest" / f"{command}-{run_at}"


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


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


def _logging_settings(connection: dict[str, str]) -> Settings:
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
        clickhouse_url=connection["url"],
        clickhouse_database=connection["database"],
        clickhouse_username=connection["username"],
        clickhouse_password=connection["password"],
        data_storage_bucket="unused",
        genai_assets_base_prefix="unused",
        openai_api_key="unused",
        gemini_api_key="unused",
    )


if __name__ == "__main__":
    sys.exit(main())
