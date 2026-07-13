from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.analysis.segment_performance import LogisticSegmentPerformanceModel
from offline_evaluation.expedia_backtest import (
    ExpediaBacktestConfig,
    ExpediaBacktestError,
    ExpediaBacktestRepository,
    ExpediaBacktestRun,
    ExpediaBacktestScenario,
    ExpediaSegmentBacktestService,
    ExpediaSourceStats,
    summarize_backtest,
    write_backtest_artifacts,
)
from offline_evaluation.rank_quality import (
    CRITERION_EVIDENCE,
    CRITERION_QUALITY,
    VERDICT_PASSED,
    criterion_result,
    determine_final_verdict,
)
from offline_evaluation.sealed_execution import (
    SealedExecution,
    SealedExecutionError,
    reserve_sealed_execution,
)


SEALED_FINAL_TEST_VERSION = "expedia.sealed-final-test.v2"


@dataclass(frozen=True, slots=True)
class ExpediaFinalTestCriteria:
    rank_one_beats_baseline_rate_min: float = 0.70
    rank_one_is_best_rate_min: float = 0.50
    rank_two_beats_baseline_rate_min: float = 0.50
    mean_rank_two_lift_percentage_points_min: float = 0.0
    rank_three_beats_baseline_rate_min: float = 0.50
    mean_rank_three_lift_percentage_points_min: float = 0.0
    pairwise_rank_accuracy_min: float = 0.55
    observed_outcome_scenario_count_min: int = 6
    rank_comparable_scenario_count_min: int = 6
    rank_comparable_scenario_rate_min: float = 0.70
    rank_two_result_count_min: int = 6
    rank_three_result_count_min: int = 6
    three_rank_scenario_count_min: int = 6
    pairwise_rank_comparison_count_min: int = 12
    pairwise_rank_tie_rate_max: float = 0.50
    all_candidate_mae_percentage_points_max: float = 3.50
    absolute_prediction_bias_percentage_points_max: float = 1.50
    brier_skill_score_min_exclusive: float = 0.0

    def __post_init__(self) -> None:
        rates = (
            self.rank_one_beats_baseline_rate_min,
            self.rank_one_is_best_rate_min,
            self.rank_two_beats_baseline_rate_min,
            self.rank_three_beats_baseline_rate_min,
            self.pairwise_rank_accuracy_min,
            self.rank_comparable_scenario_rate_min,
            self.pairwise_rank_tie_rate_max,
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("final test rate criteria must be between 0 and 1")
        if self.all_candidate_mae_percentage_points_max < 0:
            raise ValueError("final test MAE criterion must not be negative")
        if self.absolute_prediction_bias_percentage_points_max < 0:
            raise ValueError("final test bias criterion must not be negative")
        counts = (
            self.observed_outcome_scenario_count_min,
            self.rank_comparable_scenario_count_min,
            self.rank_two_result_count_min,
            self.rank_three_result_count_min,
            self.three_rank_scenario_count_min,
            self.pairwise_rank_comparison_count_min,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("final test count criteria must be positive")


@dataclass(frozen=True, slots=True)
class ExpediaSealedFinalTestManifest:
    manifest_id: str
    integrity_sha256: str
    created_at: str
    code_commit: str
    code_tree: str
    source_table: str
    source: Mapping[str, Any]
    model: Mapping[str, Any]
    config: Mapping[str, Any]
    development_validation: Mapping[str, Any]
    final_test: Mapping[str, Any]
    acceptance_criteria: Mapping[str, Any]
    version: str = SEALED_FINAL_TEST_VERSION

    @property
    def required_confirmation(self) -> str:
        return f"RUN_FINAL_TEST_{self.manifest_id[:12]}"

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest_id": self.manifest_id,
            "integrity_sha256": self.integrity_sha256,
            "created_at": self.created_at,
            "code_commit": self.code_commit,
            "code_tree": self.code_tree,
            "source_table": self.source_table,
            "source": dict(self.source),
            "model": dict(self.model),
            "config": dict(self.config),
            "development_validation": dict(self.development_validation),
            "final_test": dict(self.final_test),
            "acceptance_criteria": dict(self.acceptance_criteria),
        }


@dataclass(frozen=True, slots=True)
class ExpediaSealedFinalTestResult:
    run: ExpediaBacktestRun
    metrics: Mapping[str, Any]
    criteria_results: Mapping[str, Any]
    verdict: str
    passed: bool


def build_sealed_final_test_manifest(
    repository: ExpediaBacktestRepository,
    *,
    source_table: str,
    source_stats: ExpediaSourceStats,
    source_checksum: str,
    model_path: Path,
    model: LogisticSegmentPerformanceModel,
    config: ExpediaBacktestConfig,
    development_cutoffs: Sequence[datetime],
    final_cutoffs: Sequence[datetime],
    development_scenarios_per_cutoff: int,
    code_commit: str,
    code_tree: str,
    criteria: ExpediaFinalTestCriteria | None = None,
    created_at: datetime | None = None,
) -> ExpediaSealedFinalTestManifest:
    if not development_cutoffs or not final_cutoffs:
        raise ValueError("development and final test cutoffs must not be empty")
    if development_scenarios_per_cutoff <= 0:
        raise ValueError("development scenarios per cutoff must be positive")
    if not model_path.exists():
        raise ValueError(f"segment performance model not found: {model_path}")

    development_destination_ids: set[int] = set()
    for cutoff in sorted(development_cutoffs):
        observation_start = cutoff - timedelta(days=config.lookback_days)
        scenarios = repository.list_scenarios(
            observation_start=observation_start,
            cutoff=cutoff,
            limit=development_scenarios_per_cutoff,
            min_users=config.min_scenario_users,
            user_sample_modulo=config.user_sample_modulo,
            user_sample_remainder=config.user_sample_remainder,
            season=config.season,
            excluded_destination_ids=(),
        )
        development_destination_ids.update(
            scenario.target_destination_id for scenario in scenarios
        )

    reserved_destination_ids = set(development_destination_ids)
    final_scenarios: list[ExpediaBacktestScenario] = []
    for cutoff in sorted(final_cutoffs):
        observation_start = cutoff - timedelta(days=config.lookback_days)
        scenarios = repository.list_scenarios(
            observation_start=observation_start,
            cutoff=cutoff,
            limit=config.max_scenarios_per_cutoff,
            min_users=config.min_scenario_users,
            user_sample_modulo=config.user_sample_modulo,
            user_sample_remainder=config.user_sample_remainder,
            season=config.season,
            excluded_destination_ids=tuple(sorted(reserved_destination_ids)),
        )
        if len(scenarios) != config.max_scenarios_per_cutoff:
            raise ExpediaBacktestError(
                "not enough unseen destinations to seal the requested final test"
            )
        final_scenarios.extend(scenarios)
        reserved_destination_ids.update(
            scenario.target_destination_id for scenario in scenarios
        )

    source_payload = _source_payload(
        source_table=source_table,
        stats=source_stats,
        checksum=source_checksum,
    )
    model_payload = {
        "file_name": model_path.name,
        "sha256": _file_sha256(model_path),
        "version": model.version,
        "training_end_cutoff": model.training_metadata.get(
            "training_end_cutoff"
        ),
        "training_target": model.training_metadata.get("target"),
    }
    criteria_payload = asdict(criteria or ExpediaFinalTestCriteria())
    base_payload: dict[str, Any] = {
        "version": SEALED_FINAL_TEST_VERSION,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "source_table": source_table,
        "source": source_payload,
        "model": model_payload,
        "config": _config_payload(config),
        "development_validation": {
            "role": "adaptive_development_validation",
            "cutoffs": [value.isoformat() for value in development_cutoffs],
            "scenarios_per_cutoff": development_scenarios_per_cutoff,
            "excluded_destination_ids": sorted(development_destination_ids),
        },
        "final_test": {
            "role": "internal_sealed_destination_holdout",
            "selection_uses_future_outcomes": False,
            "cutoffs": [value.isoformat() for value in final_cutoffs],
            "scenarios": [_scenario_payload(value) for value in final_scenarios],
        },
        "acceptance_criteria": criteria_payload,
    }
    manifest_id = _json_sha256(base_payload)
    created_at_value = (created_at or datetime.now(UTC)).isoformat()
    integrity_payload = {
        **base_payload,
        "manifest_id": manifest_id,
        "created_at": created_at_value,
    }
    return ExpediaSealedFinalTestManifest(
        manifest_id=manifest_id,
        integrity_sha256=_json_sha256(integrity_payload),
        created_at=created_at_value,
        code_commit=code_commit,
        code_tree=code_tree,
        source_table=source_table,
        source=source_payload,
        model=model_payload,
        config=base_payload["config"],
        development_validation=base_payload["development_validation"],
        final_test=base_payload["final_test"],
        acceptance_criteria=criteria_payload,
    )


def write_sealed_final_test_manifest(
    manifest: ExpediaSealedFinalTestManifest,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as destination:
            json.dump(
                manifest.to_json(),
                destination,
                ensure_ascii=False,
                indent=2,
            )
            destination.write("\n")
    except FileExistsError as exc:
        raise ExpediaBacktestError(
            "sealed final test manifest already exists; do not overwrite a sealed set"
        ) from exc


def load_sealed_final_test_manifest(path: Path) -> ExpediaSealedFinalTestManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("sealed final test manifest must be a JSON object")
    if payload.get("version") != SEALED_FINAL_TEST_VERSION:
        raise ValueError("unsupported sealed final test manifest version")

    manifest = ExpediaSealedFinalTestManifest(
        manifest_id=str(payload.get("manifest_id", "")),
        integrity_sha256=str(payload.get("integrity_sha256", "")),
        created_at=str(payload.get("created_at", "")),
        code_commit=str(payload.get("code_commit", "")),
        code_tree=str(payload.get("code_tree", "")),
        source_table=str(payload.get("source_table", "")),
        source=_mapping(payload, "source"),
        model=_mapping(payload, "model"),
        config=_mapping(payload, "config"),
        development_validation=_mapping(payload, "development_validation"),
        final_test=_mapping(payload, "final_test"),
        acceptance_criteria=_mapping(payload, "acceptance_criteria"),
    )
    serialized = manifest.to_json()
    expected_integrity = str(serialized.pop("integrity_sha256"))
    actual_integrity = _json_sha256(serialized)
    if actual_integrity != expected_integrity:
        raise ValueError("sealed final test manifest integrity check failed")
    stable_payload = dict(serialized)
    expected_manifest_id = str(stable_payload.pop("manifest_id"))
    stable_payload.pop("created_at")
    actual_manifest_id = _json_sha256(stable_payload)
    if actual_manifest_id != expected_manifest_id:
        raise ValueError("sealed final test manifest ID check failed")
    return manifest


def verify_sealed_final_test_runtime(
    manifest: ExpediaSealedFinalTestManifest,
    *,
    source_table: str,
    source_stats: ExpediaSourceStats,
    source_checksum: str,
    model_path: Path,
    model: LogisticSegmentPerformanceModel,
    code_commit: str,
    code_tree: str,
) -> None:
    if source_table != manifest.source_table:
        raise ValueError("sealed final test source table changed")
    current_source = _source_payload(
        source_table=source_table,
        stats=source_stats,
        checksum=source_checksum,
    )
    if current_source != dict(manifest.source):
        raise ValueError("sealed final test source data changed after sealing")
    if _file_sha256(model_path) != manifest.model.get("sha256"):
        raise ValueError("sealed final test model changed after sealing")
    if model.version != manifest.model.get("version"):
        raise ValueError("sealed final test model version changed after sealing")
    if code_commit != manifest.code_commit or code_tree != manifest.code_tree:
        raise ValueError("sealed final test code changed after sealing")


def reserve_sealed_final_test_execution(
    manifest_path: Path,
    manifest: ExpediaSealedFinalTestManifest,
    *,
    code_commit: str,
    output_dir: Path,
    resume_execution_id: str | None = None,
) -> SealedExecution:
    try:
        return reserve_sealed_execution(
            manifest_path,
            manifest_id=manifest.manifest_id,
            manifest_integrity_sha256=manifest.integrity_sha256,
            code_commit=code_commit,
            output_dir=output_dir,
            resume_execution_id=resume_execution_id,
        )
    except SealedExecutionError as exc:
        raise ExpediaBacktestError(str(exc)) from exc


def run_sealed_final_test(
    repository: ExpediaBacktestRepository,
    *,
    manifest: ExpediaSealedFinalTestManifest,
    model: LogisticSegmentPerformanceModel,
    on_outcomes_opened: Callable[[], None] | None = None,
) -> ExpediaSealedFinalTestResult:
    scenarios = _manifest_scenarios(manifest)
    config = _config_from_payload(manifest.config)
    run = ExpediaSegmentBacktestService(
        repository,
        config=config,
        performance_predictor=model,
    ).run_scenarios(
        scenarios,
        on_outcomes_opened=on_outcomes_opened,
    )
    metrics = dict(summarize_backtest(run))
    training_rate = float(
        model.training_metadata.get(
            "training_contextual_booking_observation_rate",
            0.0,
        )
        or 0.0
    )
    constant_brier = _constant_probability_brier_score(run, training_rate)
    model_brier = float(metrics.get("all_candidate_brier_score", 0.0) or 0.0)
    metrics["constant_training_rate_brier_score"] = constant_brier
    metrics["all_candidate_brier_skill_score"] = (
        1.0 - model_brier / constant_brier if constant_brier > 0 else 0.0
    )
    criteria_results = _evaluate_criteria(
        metrics,
        manifest.acceptance_criteria,
    )
    verdict = determine_final_verdict(criteria_results)
    return ExpediaSealedFinalTestResult(
        run=run,
        metrics=metrics,
        criteria_results=criteria_results,
        verdict=verdict,
        passed=verdict == VERDICT_PASSED,
    )


def write_sealed_final_test_artifacts(
    result: ExpediaSealedFinalTestResult,
    *,
    manifest: ExpediaSealedFinalTestManifest,
    output_dir: Path,
    source_stats: ExpediaSourceStats,
) -> dict[str, Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ExpediaBacktestError(
            "sealed final test output already exists; final results cannot be overwritten"
        ) from exc
    detail_paths = write_backtest_artifacts(
        result.run,
        output_dir=output_dir / "details",
        source_stats=source_stats,
        config=_config_from_payload(manifest.config),
    )
    summary_path = output_dir / "sealed_final_test_summary.json"
    report_path = output_dir / "sealed_final_test_report.md"
    summary = {
        "manifest_id": manifest.manifest_id,
        "manifest_integrity_sha256": manifest.integrity_sha256,
        "scope": "internal_sealed_destination_holdout",
        "verdict": result.verdict,
        "passed": result.passed,
        "metrics": dict(result.metrics),
        "criteria_results": dict(result.criteria_results),
        "limitations": [
            "This test holds out unseen Expedia destinations, not a new calendar year.",
            "It validates future same-destination booking, not causal advertising lift.",
            "After this result is inspected, this manifest must not be reused for tuning.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _sealed_final_test_report(summary),
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "report": report_path,
        "results": detail_paths["results"],
        "skipped": detail_paths["skipped"],
    }


def sealed_final_test_cutoffs(
    manifest: ExpediaSealedFinalTestManifest,
) -> tuple[datetime, ...]:
    return tuple(
        sorted({scenario.cutoff for scenario in _manifest_scenarios(manifest)})
    )


def _source_payload(
    *,
    source_table: str,
    stats: ExpediaSourceStats,
    checksum: str,
) -> dict[str, Any]:
    values = {
        "table": source_table,
        "row_count": stats.row_count,
        "user_count": stats.user_count,
        "booking_row_count": stats.booking_row_count,
        "first_event_at": stats.first_event_at.isoformat()
        if stats.first_event_at
        else None,
        "last_event_at": stats.last_event_at.isoformat()
        if stats.last_event_at
        else None,
        "row_checksum": checksum,
    }
    return {**values, "fingerprint": _json_sha256(values)}


def _config_payload(config: ExpediaBacktestConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["excluded_destination_ids"] = list(config.excluded_destination_ids)
    return payload


def _config_from_payload(payload: Mapping[str, Any]) -> ExpediaBacktestConfig:
    values = dict(payload)
    values["excluded_destination_ids"] = tuple(
        int(value) for value in values.get("excluded_destination_ids", ())
    )
    return ExpediaBacktestConfig(**values)


def _scenario_payload(scenario: ExpediaBacktestScenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "cutoff": scenario.cutoff.isoformat(),
        "target_destination_id": scenario.target_destination_id,
        "historical_user_count": scenario.historical_user_count,
        "historical_event_count": scenario.historical_event_count,
        "season": scenario.season,
    }


def _manifest_scenarios(
    manifest: ExpediaSealedFinalTestManifest,
) -> tuple[ExpediaBacktestScenario, ...]:
    raw_scenarios = manifest.final_test.get("scenarios", ())
    if not isinstance(raw_scenarios, Sequence) or isinstance(
        raw_scenarios,
        (str, bytes),
    ):
        raise ValueError("sealed final test scenarios must be an array")
    scenarios: list[ExpediaBacktestScenario] = []
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping):
            raise ValueError("sealed final test scenario must be an object")
        scenarios.append(
            ExpediaBacktestScenario(
                scenario_id=str(raw["scenario_id"]),
                cutoff=datetime.fromisoformat(str(raw["cutoff"])),
                target_destination_id=int(raw["target_destination_id"]),
                historical_user_count=int(raw["historical_user_count"]),
                historical_event_count=int(raw["historical_event_count"]),
                season=str(raw["season"]) if raw.get("season") else None,
            )
        )
    if not scenarios:
        raise ValueError("sealed final test scenario list must not be empty")
    return tuple(scenarios)


def _constant_probability_brier_score(
    run: ExpediaBacktestRun,
    probability: float,
) -> float:
    clipped = max(0.0, min(1.0, probability))
    total_users = sum(result.sample_size for result in run.results)
    if total_users <= 0:
        return 0.0
    total_error = 0.0
    for result in run.results:
        successes = result.contextual_booking_user_count
        failures = max(result.sample_size - successes, 0)
        total_error += successes * (1.0 - clipped) ** 2
        total_error += failures * clipped**2
    return total_error / total_users


def _evaluate_criteria(
    metrics: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "observed_outcome_scenario_count": criterion_result(
            _int_metric(metrics, "observed_outcome_scenario_count"),
            ">=",
            int(criteria["observed_outcome_scenario_count_min"]),
            category=CRITERION_EVIDENCE,
        ),
        "rank_comparable_scenario_count": criterion_result(
            _int_metric(metrics, "rank_comparable_scenario_count"),
            ">=",
            int(criteria["rank_comparable_scenario_count_min"]),
            category=CRITERION_EVIDENCE,
        ),
        "rank_comparable_scenario_rate": criterion_result(
            _float_metric(metrics, "rank_comparable_scenario_rate"),
            ">=",
            float(criteria["rank_comparable_scenario_rate_min"]),
            category=CRITERION_EVIDENCE,
        ),
        "rank_two_result_count": criterion_result(
            _int_metric(metrics, "rank_two_result_count"),
            ">=",
            int(criteria["rank_two_result_count_min"]),
            category=CRITERION_EVIDENCE,
        ),
        "rank_three_result_count": criterion_result(
            _int_metric(metrics, "rank_three_result_count"),
            ">=",
            int(criteria["rank_three_result_count_min"]),
            category=CRITERION_EVIDENCE,
        ),
        "three_rank_scenario_count": criterion_result(
            _int_metric(metrics, "three_rank_scenario_count"),
            ">=",
            int(criteria["three_rank_scenario_count_min"]),
            category=CRITERION_EVIDENCE,
        ),
        "pairwise_rank_comparison_count": criterion_result(
            _int_metric(metrics, "pairwise_rank_comparison_count"),
            ">=",
            int(criteria["pairwise_rank_comparison_count_min"]),
            category=CRITERION_EVIDENCE,
        ),
        "pairwise_rank_tie_rate": criterion_result(
            _float_metric(metrics, "pairwise_rank_tie_rate"),
            "<=",
            float(criteria["pairwise_rank_tie_rate_max"]),
            category=CRITERION_EVIDENCE,
        ),
        "rank_one_beats_baseline_rate": criterion_result(
            _float_metric(metrics, "rank_one_beats_baseline_rate"),
            ">=",
            float(criteria["rank_one_beats_baseline_rate_min"]),
            category=CRITERION_QUALITY,
        ),
        "rank_one_is_best_rate": criterion_result(
            _float_metric(metrics, "rank_one_is_best_rate"),
            ">=",
            float(criteria["rank_one_is_best_rate_min"]),
            category=CRITERION_QUALITY,
        ),
        "rank_two_beats_baseline_rate": criterion_result(
            _float_metric(metrics, "rank_two_beats_baseline_rate"),
            ">=",
            float(criteria["rank_two_beats_baseline_rate_min"]),
            category=CRITERION_QUALITY,
        ),
        "mean_rank_two_lift_percentage_points": criterion_result(
            _float_metric(metrics, "mean_rank_two_lift_percentage_points"),
            ">=",
            float(criteria["mean_rank_two_lift_percentage_points_min"]),
            category=CRITERION_QUALITY,
        ),
        "rank_three_beats_baseline_rate": criterion_result(
            _float_metric(metrics, "rank_three_beats_baseline_rate"),
            ">=",
            float(criteria["rank_three_beats_baseline_rate_min"]),
            category=CRITERION_QUALITY,
        ),
        "mean_rank_three_lift_percentage_points": criterion_result(
            _float_metric(metrics, "mean_rank_three_lift_percentage_points"),
            ">=",
            float(criteria["mean_rank_three_lift_percentage_points_min"]),
            category=CRITERION_QUALITY,
        ),
        "pairwise_rank_accuracy": criterion_result(
            _float_metric(metrics, "pairwise_rank_accuracy"),
            ">=",
            float(criteria["pairwise_rank_accuracy_min"]),
            category=CRITERION_QUALITY,
        ),
        "all_candidate_mean_absolute_error_percentage_points": criterion_result(
            _float_metric(
                metrics,
                "all_candidate_mean_absolute_error_percentage_points",
            ),
            "<=",
            float(criteria["all_candidate_mae_percentage_points_max"]),
            category=CRITERION_QUALITY,
        ),
        "absolute_prediction_bias_percentage_points": criterion_result(
            _absolute_float_metric(
                metrics,
                "all_candidate_prediction_bias_percentage_points",
            ),
            "<=",
            float(
                criteria[
                    "absolute_prediction_bias_percentage_points_max"
                ]
            ),
            category=CRITERION_QUALITY,
        ),
        "all_candidate_brier_skill_score": criterion_result(
            _float_metric(metrics, "all_candidate_brier_skill_score"),
            ">",
            float(criteria["brier_skill_score_min_exclusive"]),
            category=CRITERION_QUALITY,
        ),
    }


def _sealed_final_test_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    criteria = summary["criteria_results"]
    lines = [
        "# Expedia 봉인 최종 테스트",
        "",
        f"- Manifest ID: `{summary['manifest_id']}`",
        f"- 최종 판정: {_verdict_label(str(summary['verdict']))}",
        "- 범위: 기존 개발 검증에서 제외한 목적지 기반 내부 봉인 테스트",
        "",
        "## 핵심 지표",
        "",
        "- Rank 1 기준선 승률: "
        f"{_format_optional_percent(metrics['rank_one_beats_baseline_rate'])}",
        "- Rank 1 엄격한 실제 최고 후보 비율: "
        f"{_format_optional_percent(metrics['rank_one_is_best_rate'])}",
        "- Rank 1 실제 최고 동률 비율: "
        f"{_format_optional_percent(metrics['rank_one_tied_best_rate'])}",
        "- Rank 2 기준선 승률: "
        f"{_format_optional_percent(metrics['rank_two_beats_baseline_rate'])}",
        "- Rank 3 기준선 승률: "
        f"{_format_optional_percent(metrics['rank_three_beats_baseline_rate'])}",
        "- 후보 쌍 순서 적중률: "
        f"{_format_optional_percent(metrics['pairwise_rank_accuracy'])}",
        "- 후보 쌍 동률 비율: "
        f"{_format_optional_percent(metrics['pairwise_rank_tie_rate'])}",
        "- 비교 가능한 시나리오: "
        f"{metrics['rank_comparable_scenario_count']}개",
        "- Rank 1·2·3이 모두 생성된 시나리오: "
        f"{metrics['three_rank_scenario_count']}개",
        "- 전체 후보 평균 절대 오차: "
        f"{metrics['all_candidate_mean_absolute_error_percentage_points']:.2f}%p",
        "- 전체 후보 예측 편향: "
        f"{metrics['all_candidate_prediction_bias_percentage_points']:.2f}%p",
        "- Brier skill score: "
        f"{metrics['all_candidate_brier_skill_score']:.6f}",
        "",
        "## 사전 등록 기준",
        "",
    ]
    for name, result in criteria.items():
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"- [{result['category']}] {name}: "
            f"{_format_criterion_value(result['actual'])} "
            f"{result['operator']} {result['threshold']} · {status}"
        )
    lines.extend(
        [
            "",
            "## 한계",
            "",
            "- 이 결과는 새로운 연도가 아니라 아직 평가하지 않은 Expedia "
            "목적지를 봉인한 내부 테스트입니다.",
            "- 실제 광고 노출 대조군이 없으므로 광고의 인과적 증분 효과를 "
            "검증하지 않습니다.",
            "- 결과를 확인한 뒤 추천 로직을 수정하면 이 manifest는 다시 "
            "최종 테스트로 사용할 수 없습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _float_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def _int_metric(metrics: Mapping[str, Any], key: str) -> int:
    return int(metrics.get(key, 0) or 0)


def _absolute_float_metric(
    metrics: Mapping[str, Any],
    key: str,
) -> float | None:
    value = _float_metric(metrics, key)
    return abs(value) if value is not None else None


def _verdict_label(verdict: str) -> str:
    return {
        "passed": "PASS",
        "failed": "FAIL",
        "inconclusive": "INCONCLUSIVE (근거 부족)",
    }.get(verdict, verdict.upper())


def _format_optional_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _format_criterion_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"sealed final test manifest field {key!r} must be an object")
    return dict(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
