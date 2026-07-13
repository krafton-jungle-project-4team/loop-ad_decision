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
    python3 scripts/backtest_expedia_segments.py validation \
        --user-sample-modulo 1
    python3 scripts/backtest_expedia_segments.py seal-final-test
    python3 scripts/backtest_expedia_segments.py run-final-test \
        --confirm RUN_FINAL_TEST_<manifest-id-prefix>
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

from offline_evaluation.expedia_backtest import (  # noqa: E402
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
from offline_evaluation.expedia_final_test import (  # noqa: E402
    ExpediaFinalTestCriteria,
    ExpediaSealedFinalTestManifest,
    build_sealed_final_test_manifest,
    load_sealed_final_test_manifest,
    reserve_sealed_final_test_execution,
    run_sealed_final_test,
    sealed_final_test_cutoffs,
    verify_sealed_final_test_runtime,
    write_sealed_final_test_artifacts,
    write_sealed_final_test_manifest,
)
from app.analysis.segment_performance import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    load_segment_performance_model,
)
from app.config import DECISION_SERVICE_ID, Settings  # noqa: E402
from app.logging import (  # noqa: E402
    configure_logging,
    duration_ms,
    log,
    log_context_scope,
    now_ms,
)
from offline_evaluation.git_state import (  # noqa: E402
    inspect_clean_git_identity,
)
from offline_evaluation.sealed_execution import (  # noqa: E402
    STATUS_RESULT_STAGED,
    mark_execution_failure,
    mark_outcomes_opened,
    mark_result_staged,
    prepare_staging_output,
    publish_staged_result,
    sealed_execution_attempt,
)


DEFAULT_SOURCE_TABLE = "expedia_hotel_events"
DEFAULT_PROJECT_ID = "expedia_backtest"
DEFAULT_FINAL_TEST_MANIFEST = Path(
    "artifacts/expedia-segment-backtest/sealed-final-test-manifest.json"
)


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
    if args.command == "seal-final-test":
        return seal_final_test(args, repository, started_at=started_at)
    if args.command == "run-final-test":
        return execute_final_test(args, repository, started_at=started_at)

    config = backtest_config(args)
    if args.command in {"validation", "holdout"}:
        training_cutoffs = monthly_cutoffs(
            args.train_start_cutoff,
            args.train_end_cutoff,
        )
        validation_cutoffs = monthly_cutoffs(
            args.start_cutoff,
            args.end_cutoff,
        )
        stats = repository.source_stats()
        validate_source_window(
            stats,
            cutoffs=[*training_cutoffs, *validation_cutoffs],
            lookback_days=config.lookback_days,
            outcome_days=config.outcome_days,
        )
        temporal_run = run_temporal_holdout_backtest(
            repository,
            config=config,
            training_cutoffs=training_cutoffs,
            holdout_cutoffs=validation_cutoffs,
        )
        output_dir = args.output_dir or default_output_dir(args.command)
        artifacts = write_temporal_holdout_artifacts(
            temporal_run,
            output_dir=output_dir,
            source_stats=stats,
            config=config,
        )
        log.info(
            "temporal_validation_artifacts_created",
            {
                "trainingResultCount": len(temporal_run.training_run.results),
                "validationResultCount": len(temporal_run.holdout_run.results),
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
                "validationResultCount": len(temporal_run.holdout_run.results),
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


def seal_final_test(
    args: argparse.Namespace,
    repository: ClickHouseExpediaBacktestRepository,
    *,
    started_at: float,
) -> int:
    code_commit, code_tree = sealing_git_identity()
    config = backtest_config(args)
    development_cutoffs = monthly_cutoffs(
        args.development_start_cutoff,
        args.development_end_cutoff,
    )
    final_cutoffs = monthly_cutoffs(
        args.final_start_cutoff,
        args.final_end_cutoff,
    )
    stats = repository.source_stats()
    validate_source_window(
        stats,
        cutoffs=[*development_cutoffs, *final_cutoffs],
        lookback_days=config.lookback_days,
        outcome_days=config.outcome_days,
    )
    model_path = args.model_path.expanduser().resolve()
    model = load_segment_performance_model(model_path)
    manifest = build_sealed_final_test_manifest(
        repository,
        source_table=args.source_table,
        source_stats=stats,
        source_checksum=repository.source_checksum(),
        model_path=model_path,
        model=model,
        config=config,
        development_cutoffs=development_cutoffs,
        final_cutoffs=final_cutoffs,
        development_scenarios_per_cutoff=(
            args.development_scenarios_per_cutoff
        ),
        code_commit=code_commit,
        code_tree=code_tree,
        criteria=ExpediaFinalTestCriteria(
            rank_one_beats_baseline_rate_min=(
                args.min_rank_one_beats_baseline_rate
            ),
            rank_one_is_best_rate_min=args.min_rank_one_is_best_rate,
            all_candidate_mae_percentage_points_max=(
                args.max_all_candidate_mae_percentage_points
            ),
            absolute_prediction_bias_percentage_points_max=(
                args.max_absolute_prediction_bias_percentage_points
            ),
            brier_skill_score_min_exclusive=args.min_brier_skill_score,
        ),
    )
    write_sealed_final_test_manifest(manifest, args.manifest)
    log.info(
        "sealed_final_test_manifest_created",
        {
            "manifestId": manifest.manifest_id,
            "manifestPath": args.manifest,
            "scenarioCount": len(manifest.final_test["scenarios"]),
            "excludedDestinationCount": len(
                manifest.development_validation["excluded_destination_ids"]
            ),
            "requiredConfirmation": manifest.required_confirmation,
        },
    )
    log.info(
        "completed",
        {
            "mode": args.command,
            "manifestId": manifest.manifest_id,
            "manifestPath": args.manifest,
            "requiredConfirmation": manifest.required_confirmation,
            "durationMs": duration_ms(started_at),
        },
    )
    print(f"manifest={args.manifest}")
    print(f"confirmation={manifest.required_confirmation}")
    return 0


def execute_final_test(
    args: argparse.Namespace,
    repository: ClickHouseExpediaBacktestRepository,
    *,
    started_at: float,
) -> int:
    manifest = load_sealed_final_test_manifest(args.manifest)
    if args.confirm != manifest.required_confirmation:
        raise ValueError(
            "final test confirmation does not match the sealed manifest; "
            f"use {manifest.required_confirmation!r} only after code freeze"
        )
    code_commit, code_tree = execution_git_identity()
    model_path = args.model_path.expanduser().resolve()
    model = load_segment_performance_model(model_path)
    stats = repository.source_stats()
    source_checksum = repository.source_checksum()
    verify_sealed_final_test_runtime(
        manifest,
        source_table=args.source_table,
        source_stats=stats,
        source_checksum=source_checksum,
        model_path=model_path,
        model=model,
        code_commit=code_commit,
        code_tree=code_tree,
    )
    config = ExpediaBacktestConfig(**{
        **dict(manifest.config),
        "excluded_destination_ids": tuple(
            manifest.config.get("excluded_destination_ids", ())
        ),
    })
    validate_source_window(
        stats,
        cutoffs=sealed_final_test_cutoffs(manifest),
        lookback_days=config.lookback_days,
        outcome_days=config.outcome_days,
    )
    output_dir = args.output_dir or default_final_test_output_dir(manifest)
    execution = reserve_sealed_final_test_execution(
        args.manifest,
        manifest,
        code_commit=code_commit,
        output_dir=output_dir,
        resume_execution_id=args.resume_execution_id,
    )
    log.assign_context({"executionId": execution.execution_id})
    with sealed_execution_attempt(execution):
        if execution.status == STATUS_RESULT_STAGED:
            completed = publish_staged_result(execution)
            log.info(
                "sealed_result_publication_resumed",
                {
                    "outputDir": completed.output_dir,
                    "executionJournalPath": completed.journal_path,
                },
            )
            return 0
        staging_dir = prepare_staging_output(execution)
        try:
            result = run_sealed_final_test(
                repository,
                manifest=manifest,
                model=model,
                on_outcomes_opened=lambda: mark_outcomes_opened(execution),
            )
            write_sealed_final_test_artifacts(
                result,
                manifest=manifest,
                output_dir=staging_dir,
                source_stats=stats,
            )
            mark_result_staged(execution)
            completed = publish_staged_result(execution)
        except Exception as exc:
            failed = mark_execution_failure(execution, exc)
            log.info(
                "sealed_execution_state_updated",
                {
                    "executionStatus": failed.status,
                    "executionJournalPath": failed.journal_path,
                },
            )
            raise
    log.info(
        "sealed_final_test_completed",
        {
            "manifestId": manifest.manifest_id,
            "verdict": result.verdict,
            "passed": result.passed,
            "scenarioResultCount": len(result.run.results),
            "skippedScenarioCount": len(result.run.skipped_scenarios),
            "executionJournalPath": completed.journal_path,
            "outputDir": completed.output_dir,
            "reportPath": completed.output_dir
            / "sealed_final_test_report.md",
            "summaryPath": completed.output_dir
            / "sealed_final_test_summary.json",
        },
    )
    log.info(
        "completed",
        {
            "mode": args.command,
            "manifestId": manifest.manifest_id,
            "verdict": result.verdict,
            "passed": result.passed,
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
        "validation",
        aliases=["holdout"],
        help=(
            "Fit contextual booking calibration on 2013 windows and evaluate "
            "predictions and ranking on the repeatedly inspected 2014 "
            "development-validation windows."
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

    seal = subparsers.add_parser(
        "seal-final-test",
        help=(
            "Select unseen destination scenarios without reading future outcomes "
            "and write an immutable final-test manifest."
        ),
    )
    add_connection_arguments(seal)
    add_backtest_arguments(
        seal,
        smoke=False,
        default_user_sample_modulo=1,
        include_output_dir=False,
    )
    seal.add_argument("--manifest", type=Path, default=DEFAULT_FINAL_TEST_MANIFEST)
    seal.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    seal.add_argument(
        "--development-start-cutoff",
        type=parse_date,
        default=date(2014, 1, 1),
    )
    seal.add_argument(
        "--development-end-cutoff",
        type=parse_date,
        default=date(2014, 12, 1),
    )
    seal.add_argument(
        "--development-scenarios-per-cutoff",
        type=positive_int,
        default=3,
    )
    seal.add_argument(
        "--final-start-cutoff",
        type=parse_date,
        default=date(2014, 7, 1),
    )
    seal.add_argument(
        "--final-end-cutoff",
        type=parse_date,
        default=date(2014, 12, 1),
    )
    seal.add_argument(
        "--min-rank-one-beats-baseline-rate",
        type=unit_interval,
        default=0.70,
    )
    seal.add_argument(
        "--min-rank-one-is-best-rate",
        type=unit_interval,
        default=0.50,
    )
    seal.add_argument(
        "--max-all-candidate-mae-percentage-points",
        type=nonnegative_float,
        default=3.50,
    )
    seal.add_argument(
        "--max-absolute-prediction-bias-percentage-points",
        type=nonnegative_float,
        default=1.50,
    )
    seal.add_argument(
        "--min-brier-skill-score",
        type=float,
        default=0.0,
    )

    final_test = subparsers.add_parser(
        "run-final-test",
        help=(
            "Open future outcomes for one sealed manifest exactly once and write "
            "the pre-registered verdict."
        ),
    )
    add_connection_arguments(final_test)
    final_test.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_FINAL_TEST_MANIFEST,
    )
    final_test.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    final_test.add_argument("--confirm", required=True)
    final_test.add_argument("--output-dir", type=Path, default=None)
    final_test.add_argument(
        "--resume-execution-id",
        help=(
            "Resume the same execution after a pre-outcome or publication "
            "failure. The ID must match the execution journal."
        ),
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
    default_user_sample_modulo: int = 20,
    include_output_dir: bool = True,
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
        default=default_user_sample_modulo,
        help="20 uses a deterministic 5%% user sample; 1 uses all users.",
    )
    parser.add_argument("--user-sample-remainder", type=nonnegative_int, default=0)
    parser.add_argument(
        "--season",
        choices=("none", "spring", "summer", "fall", "winter"),
        default="none",
    )
    if include_output_dir:
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


def default_final_test_output_dir(
    manifest: ExpediaSealedFinalTestManifest,
) -> Path:
    return (
        Path("artifacts")
        / "expedia-segment-backtest"
        / f"sealed-final-{manifest.manifest_id[:12]}"
    )


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


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def sealing_git_identity() -> tuple[str, str]:
    identity = inspect_clean_git_identity(
        REPOSITORY_ROOT,
        required_branch="dev",
    )
    return identity.commit, identity.tree


def execution_git_identity() -> tuple[str, str]:
    identity = inspect_clean_git_identity(REPOSITORY_ROOT)
    return identity.commit, identity.tree


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
