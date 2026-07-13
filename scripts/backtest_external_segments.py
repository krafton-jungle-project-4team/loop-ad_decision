#!/usr/bin/env python3
"""Run repeatable diagnostics or one-time sealed external evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.analysis.segment_performance import (  # noqa: E402
    DEFAULT_MODEL_PATH,
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
    EXTERNAL_DEVELOPMENT_ROLE,
    EXTERNAL_SEALED_FINAL_ROLE,
    ExternalAdapterConfig,
    load_external_dataset,
)
from offline_evaluation.external_final_test import (  # noqa: E402
    EXTERNAL_COHORT_MODULO,
    EXTERNAL_DEVELOPMENT_REMAINDERS,
    EXTERNAL_FINAL_REMAINDERS,
    SYNERISE_DEVELOPMENT_CUTOFFS,
    SYNERISE_FINAL_CUTOFF,
    build_external_sealed_final_test_manifest,
    load_external_sealed_final_test_manifest,
    reserve_external_sealed_final_test_execution,
    run_external_sealed_final_test,
    verify_external_sealed_final_test_runtime,
    write_external_sealed_final_test_artifacts,
    write_external_sealed_final_test_manifest,
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


DEFAULT_SOURCE_DIRS = {
    "airbnb": Path(
        "artifacts/external-datasets/airbnb-recruiting-new-user-bookings"
    ),
    "booking-com": Path("artifacts/external-datasets/booking-com"),
    "synerise": Path("artifacts/external-datasets/synerise_dataset"),
}
DATASET_IDS = tuple(DEFAULT_SOURCE_DIRS)


def main() -> int:
    args = parse_args()
    configure_logging(_logging_settings())
    try:
        if args.command == "development":
            return run_development_diagnostic(args)
        if args.command == "seal-final-test":
            return seal_final_test(args)
        if args.command == "run-final-test":
            return execute_final_test(args)
        raise ValueError(f"unsupported command: {args.command}")
    except (ExternalBacktestError, ValueError) as exc:
        log.warn(
            "external_evaluation_invalid",
            {
                "command": args.command,
                "datasetId": getattr(args, "dataset", None),
                "err": exc,
            },
        )
        return 2


@log_context_scope
def run_development_diagnostic(args: argparse.Namespace) -> int:
    started_at = now_ms()
    dataset_id = args.dataset
    source_dir = args.source_dir or DEFAULT_SOURCE_DIRS[dataset_id]
    output_dir = args.output_dir or _default_development_output_dir(dataset_id)
    predictor = build_segment_performance_predictor(args.model_path)
    cutoffs = _development_cutoffs(args)
    base_config = _development_adapter_config(args)
    log.assign_context(
        {
            "datasetId": dataset_id,
            "evaluationMode": EXTERNAL_DEVELOPMENT_ROLE,
        }
    )
    log.info(
        "started",
        {
            "sourceDir": source_dir,
            "profilePoolLimit": args.profile_pool_limit,
            "sampleModulo": base_config.sample_modulo,
            "sampleRemainders": base_config.effective_sample_remainders,
            "cutoffCount": len(cutoffs),
        },
    )

    entries: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        adapter_config = replace(base_config, cutoff=cutoff)
        bundle = load_external_dataset(
            dataset_id,
            source_dir,
            config=adapter_config,
        )
        backtest_config = _backtest_config(args, bundle.manifest)
        run = run_external_backtest(
            bundle.cases,
            config=backtest_config,
            performance_predictor=predictor,
        )
        if not run.results:
            raise ExternalBacktestError(
                "external development diagnostic produced no candidates; "
                "increase the profile pool or lower sample thresholds"
            )
        run_output_dir = (
            output_dir
            if len(cutoffs) == 1
            else output_dir / f"cutoff-{cutoff:%Y%m%d}"
        )
        artifacts = write_external_backtest_artifacts(
            run,
            manifest=bundle.manifest,
            output_dir=run_output_dir,
            model_metadata=predictor.metadata(),
        )
        entries.append(
            {
                "cutoff": cutoff.isoformat(),
                "metrics": dict(run.summary),
                "summary_path": str(artifacts["summary"]),
                "report_path": str(artifacts["report"]),
            }
        )

    aggregate_path = output_dir / "development_diagnostic_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(
            {
                "role": EXTERNAL_DEVELOPMENT_ROLE,
                "dataset_id": dataset_id,
                "repeatable": True,
                "updates_model_parameters": False,
                "runs": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info(
        "completed",
        {
            "datasetId": dataset_id,
            "evaluationMode": EXTERNAL_DEVELOPMENT_ROLE,
            "runCount": len(entries),
            "outputDir": output_dir,
            "summaryPath": aggregate_path,
            "durationMs": duration_ms(started_at),
        },
    )
    return 0


@log_context_scope
def seal_final_test(args: argparse.Namespace) -> int:
    started_at = now_ms()
    dataset_id = args.dataset
    source_dir = args.source_dir or DEFAULT_SOURCE_DIRS[dataset_id]
    model_path = _model_path(args.model_path)
    predictor = build_segment_performance_predictor(model_path)
    code_commit, code_tree = frozen_git_identity()
    adapter_config = _final_adapter_config(args)
    backtest_config = ExternalBacktestConfig(
        max_suggested_segments=args.max_segments,
        min_sample_size=args.min_sample_size,
        prediction_error_comparable=False,
        prediction_error_comparability_reason=_prediction_reason(dataset_id),
    )
    manifest_path = args.manifest or _default_manifest_path(dataset_id)
    log.assign_context(
        {
            "datasetId": dataset_id,
            "evaluationMode": "seal_external_final",
        }
    )
    log.info(
        "started",
        {
            "sourceDir": source_dir,
            "manifestPath": manifest_path,
            "codeCommit": code_commit,
        },
    )
    manifest = build_external_sealed_final_test_manifest(
        dataset_id=dataset_id,
        source_dir=source_dir,
        model_path=model_path,
        model_metadata=predictor.metadata(),
        adapter_config=adapter_config,
        backtest_config=backtest_config,
        code_commit=code_commit,
        code_tree=code_tree,
    )
    write_external_sealed_final_test_manifest(manifest, manifest_path)
    log.info(
        "completed",
        {
            "datasetId": dataset_id,
            "manifestId": manifest.manifest_id,
            "manifestPath": manifest_path,
            "requiredConfirmation": manifest.required_confirmation,
            "durationMs": duration_ms(started_at),
        },
    )
    print(f"manifest={manifest_path}")
    print(f"confirmation={manifest.required_confirmation}")
    return 0


@log_context_scope
def execute_final_test(args: argparse.Namespace) -> int:
    started_at = now_ms()
    manifest = load_external_sealed_final_test_manifest(args.manifest)
    if args.confirm != manifest.required_confirmation:
        raise ValueError(
            "external final confirmation does not match the sealed manifest; "
            f"use {manifest.required_confirmation!r} only after code freeze"
        )
    source_dir = args.source_dir or DEFAULT_SOURCE_DIRS[manifest.dataset_id]
    model_path = _model_path(args.model_path)
    predictor = build_segment_performance_predictor(model_path)
    code_commit, code_tree = frozen_git_identity()
    output_dir = args.output_dir or _default_final_output_dir(manifest)
    log.assign_context(
        {
            "datasetId": manifest.dataset_id,
            "evaluationMode": EXTERNAL_SEALED_FINAL_ROLE,
            "manifestId": manifest.manifest_id,
        }
    )
    log.info(
        "started",
        {
            "sourceDir": source_dir,
            "manifestPath": args.manifest,
            "outputDir": output_dir,
            "codeCommit": code_commit,
        },
    )
    verify_external_sealed_final_test_runtime(
        manifest,
        source_dir=source_dir,
        model_path=model_path,
        model_metadata=predictor.metadata(),
        code_commit=code_commit,
        code_tree=code_tree,
    )
    execution = reserve_external_sealed_final_test_execution(
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
            result = run_external_sealed_final_test(
                manifest=manifest,
                source_dir=source_dir,
                performance_predictor=predictor,
                on_outcomes_opened=lambda: mark_outcomes_opened(execution),
            )
            write_external_sealed_final_test_artifacts(
                result,
                manifest=manifest,
                output_dir=staging_dir,
                model_metadata=predictor.metadata(),
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
        "completed",
        {
            "datasetId": manifest.dataset_id,
            "manifestId": manifest.manifest_id,
            "verdict": result.verdict,
            "passed": result.passed,
            "candidateResultCount": len(result.run.results),
            "executionJournalPath": completed.journal_path,
            "summaryPath": completed.output_dir
            / "sealed_final_test_summary.json",
            "reportPath": completed.output_dir
            / "sealed_final_test_report.md",
            "durationMs": duration_ms(started_at),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate production segment recommendations on external datasets "
            "without fitting the Expedia performance model."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    development = subparsers.add_parser(
        "development",
        help="Run repeatable external diagnostics for development decisions.",
    )
    development.add_argument("dataset", choices=DATASET_IDS)
    _add_source_and_model_arguments(development)
    _add_evaluation_size_arguments(development)
    development.add_argument("--output-dir", type=Path)
    development.add_argument("--sample-modulo", type=positive_int)
    development.add_argument("--sample-remainders", type=parse_remainders)
    development.add_argument(
        "--cutoff",
        action="append",
        type=parse_datetime,
        help="Repeat for multiple Synerise development cutoffs.",
    )
    development.add_argument("--lookback-days", type=positive_int, default=90)
    development.add_argument("--outcome-days", type=positive_int, default=28)
    development.add_argument("--skip-checksum", action="store_true")

    seal = subparsers.add_parser(
        "seal-final-test",
        help="Freeze source, model, code and the external final partition.",
    )
    seal.add_argument("dataset", choices=DATASET_IDS)
    _add_source_and_model_arguments(seal)
    _add_evaluation_size_arguments(seal)
    seal.add_argument("--manifest", type=Path)
    seal.add_argument("--lookback-days", type=positive_int, default=90)
    seal.add_argument("--outcome-days", type=positive_int, default=28)

    final = subparsers.add_parser(
        "run-final-test",
        help="Open a sealed external outcome partition exactly once.",
    )
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--confirm", required=True)
    final.add_argument("--source-dir", type=Path)
    final.add_argument("--model-path", type=Path)
    final.add_argument("--output-dir", type=Path)
    final.add_argument(
        "--resume-execution-id",
        help=(
            "Resume the same execution after a pre-outcome or publication "
            "failure. The ID must match the execution journal."
        ),
    )
    return parser.parse_args()


def _add_source_and_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--model-path", type=Path)


def _add_evaluation_size_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile-pool-limit", type=positive_int, default=1000)
    parser.add_argument("--max-scenarios", type=positive_int, default=3)
    parser.add_argument("--max-segments", type=positive_int, default=3)
    parser.add_argument("--min-sample-size", type=positive_int, default=20)
    parser.add_argument("--min-scenario-users", type=positive_int, default=20)


def _development_adapter_config(
    args: argparse.Namespace,
) -> ExternalAdapterConfig:
    if args.dataset == "booking-com":
        default_modulo = 1
        default_remainders = (0,)
    else:
        default_modulo = EXTERNAL_COHORT_MODULO
        default_remainders = EXTERNAL_DEVELOPMENT_REMAINDERS
    modulo = args.sample_modulo or default_modulo
    remainders = args.sample_remainders or (
        tuple(range(modulo))
        if args.sample_modulo is not None
        else default_remainders
    )
    return ExternalAdapterConfig(
        profile_pool_limit=args.profile_pool_limit,
        max_scenarios=args.max_scenarios,
        min_scenario_users=args.min_scenario_users,
        sample_modulo=modulo,
        sample_remainder=remainders[0],
        sample_remainders=remainders,
        evaluation_role=EXTERNAL_DEVELOPMENT_ROLE,
        include_checksum=not args.skip_checksum,
        cutoff=_development_cutoffs(args)[0],
        lookback_days=args.lookback_days,
        outcome_days=args.outcome_days,
    )


def _final_adapter_config(args: argparse.Namespace) -> ExternalAdapterConfig:
    if args.dataset == "booking-com":
        modulo = 1
        remainders = (0,)
    else:
        modulo = EXTERNAL_COHORT_MODULO
        remainders = EXTERNAL_FINAL_REMAINDERS
    return ExternalAdapterConfig(
        profile_pool_limit=args.profile_pool_limit,
        max_scenarios=args.max_scenarios,
        min_scenario_users=args.min_scenario_users,
        sample_modulo=modulo,
        sample_remainder=remainders[0],
        sample_remainders=remainders,
        evaluation_role=EXTERNAL_SEALED_FINAL_ROLE,
        include_checksum=True,
        cutoff=(
            SYNERISE_FINAL_CUTOFF
            if args.dataset == "synerise"
            else datetime(2022, 11, 10, tzinfo=UTC)
        ),
        lookback_days=args.lookback_days,
        outcome_days=args.outcome_days,
    )


def _development_cutoffs(args: argparse.Namespace) -> tuple[datetime, ...]:
    if args.cutoff:
        return tuple(args.cutoff)
    if args.dataset == "synerise":
        return SYNERISE_DEVELOPMENT_CUTOFFS
    return (datetime(2022, 11, 10, tzinfo=UTC),)


def _backtest_config(
    args: argparse.Namespace,
    manifest: Any,
) -> ExternalBacktestConfig:
    return ExternalBacktestConfig(
        max_suggested_segments=args.max_segments,
        min_sample_size=args.min_sample_size,
        prediction_error_comparable=manifest.prediction_error_comparable,
        prediction_error_comparability_reason=(
            manifest.prediction_error_comparability_reason
        ),
    )


def _prediction_reason(dataset_id: str) -> str:
    reasons = {
        "booking-com": (
            "next itinerary city differs from future same-destination booking"
        ),
        "airbnb": (
            "static first-booking label has no matching prediction window"
        ),
        "synerise": (
            "retail category purchase is a cross-domain proxy outcome"
        ),
    }
    return reasons[dataset_id]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_remainders(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated integers"
        ) from exc
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("remainders must be non-negative")
    return parsed


def parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def frozen_git_identity() -> tuple[str, str]:
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "dev":
        raise ValueError(
            "external sealed final test must be created and executed from dev"
        )
    tracked_status = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise ValueError(
            "external sealed final test requires a clean tracked working tree"
        )
    return (
        _git_output("rev-parse", "HEAD"),
        _git_output("rev-parse", "HEAD^{tree}"),
    )


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"failed to inspect frozen git state: {' '.join(args)}"
        ) from exc
    return completed.stdout.strip()


def _model_path(value: Path | None) -> Path:
    return (value or DEFAULT_MODEL_PATH).expanduser().resolve()


def _default_development_output_dir(dataset_id: str) -> Path:
    run_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path("artifacts")
        / "external-segment-backtest"
        / dataset_id
        / f"development-{run_at}"
    )


def _default_manifest_path(dataset_id: str) -> Path:
    return (
        Path("artifacts")
        / "external-segment-backtest"
        / "sealed"
        / f"{dataset_id}-final.manifest.json"
    )


def _default_final_output_dir(
    manifest: Any,
) -> Path:
    return (
        Path("artifacts")
        / "external-segment-backtest"
        / manifest.dataset_id
        / f"sealed-final-{manifest.manifest_id[:12]}"
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
