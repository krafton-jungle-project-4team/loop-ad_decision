from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from app.analysis.segment_performance import SegmentPerformancePredictor
from offline_evaluation.external_backtest import (
    ExternalBacktestConfig,
    ExternalBacktestError,
    ExternalBacktestRun,
    ExternalDatasetManifest,
    run_external_backtest,
    source_file_descriptor,
    write_external_backtest_artifacts,
)
from offline_evaluation.external_datasets import (
    EXTERNAL_SEALED_FINAL_ROLE,
    ExternalAdapterConfig,
    external_source_paths,
    load_external_dataset,
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


EXTERNAL_SEALED_FINAL_TEST_VERSION = "external.sealed-final-test.v2"
EXTERNAL_COHORT_MODULO = 5
EXTERNAL_DEVELOPMENT_REMAINDERS = (0, 1, 2, 3)
EXTERNAL_FINAL_REMAINDERS = (4,)
SYNERISE_DEVELOPMENT_CUTOFFS = (
    datetime(2022, 9, 29, tzinfo=UTC),
    datetime(2022, 10, 13, tzinfo=UTC),
)
SYNERISE_FINAL_CUTOFF = datetime(2022, 11, 10, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ExternalFinalTestCriteria:
    rank_one_beats_baseline_rate_min: float = 0.50
    mean_rank_one_lift_percentage_points_min: float = 0.0
    rank_one_is_best_rate_min: float = 0.50
    rank_two_beats_baseline_rate_min: float = 0.50
    mean_rank_two_lift_percentage_points_min: float = 0.0
    rank_three_beats_baseline_rate_min: float = 0.50
    mean_rank_three_lift_percentage_points_min: float = 0.0
    pairwise_rank_accuracy_min: float = 0.55
    scenario_with_observed_outcome_count_min: int = 3
    rank_comparable_scenario_count_min: int = 3
    rank_comparable_scenario_rate_min: float = 0.80
    rank_two_result_count_min: int = 3
    rank_three_result_count_min: int = 3
    three_rank_scenario_count_min: int = 3
    pairwise_rank_comparison_count_min: int = 6
    pairwise_rank_tie_rate_max: float = 0.50
    candidate_type_count_min: int = 2
    mean_non_first_rank_overlap_max: float = 0.90
    maximum_non_first_rank_overlap_max: float = 0.95

    def __post_init__(self) -> None:
        rates = (
            self.rank_one_beats_baseline_rate_min,
            self.rank_one_is_best_rate_min,
            self.rank_two_beats_baseline_rate_min,
            self.rank_three_beats_baseline_rate_min,
            self.pairwise_rank_accuracy_min,
            self.rank_comparable_scenario_rate_min,
            self.pairwise_rank_tie_rate_max,
            self.mean_non_first_rank_overlap_max,
            self.maximum_non_first_rank_overlap_max,
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("external final rate criteria must be between 0 and 1")
        counts = (
            self.scenario_with_observed_outcome_count_min,
            self.rank_comparable_scenario_count_min,
            self.rank_two_result_count_min,
            self.rank_three_result_count_min,
            self.three_rank_scenario_count_min,
            self.pairwise_rank_comparison_count_min,
            self.candidate_type_count_min,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("external final count criteria must be positive")


@dataclass(frozen=True, slots=True)
class ExternalSealedFinalTestManifest:
    manifest_id: str
    integrity_sha256: str
    created_at: str
    dataset_id: str
    code_commit: str
    code_tree: str
    source: Mapping[str, Any]
    model: Mapping[str, Any]
    adapter_config: Mapping[str, Any]
    backtest_config: Mapping[str, Any]
    partition_contract: Mapping[str, Any]
    outcome_contract: Mapping[str, Any]
    acceptance_criteria: Mapping[str, Any]
    version: str = EXTERNAL_SEALED_FINAL_TEST_VERSION

    @property
    def required_confirmation(self) -> str:
        return f"RUN_EXTERNAL_FINAL_{self.manifest_id[:12]}"

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest_id": self.manifest_id,
            "integrity_sha256": self.integrity_sha256,
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "code_commit": self.code_commit,
            "code_tree": self.code_tree,
            "source": dict(self.source),
            "model": dict(self.model),
            "adapter_config": dict(self.adapter_config),
            "backtest_config": dict(self.backtest_config),
            "partition_contract": dict(self.partition_contract),
            "outcome_contract": dict(self.outcome_contract),
            "acceptance_criteria": dict(self.acceptance_criteria),
        }


@dataclass(frozen=True, slots=True)
class ExternalSealedFinalTestResult:
    run: ExternalBacktestRun
    dataset_manifest: ExternalDatasetManifest
    criteria_results: Mapping[str, Any]
    verdict: str
    passed: bool


def external_partition_contract(dataset_id: str) -> dict[str, Any]:
    if dataset_id == "booking-com":
        return {
            "development": {
                "role": "repeatable_development_diagnostic",
                "source": "train_set.csv",
                "design": "hide_last_city_within_each_train_trip",
            },
            "final": {
                "role": "one_time_sealed_external_evaluation",
                "source": "test_set.csv + ground_truth.csv",
                "design": "official_test_ground_truth_holdout",
                "disjoint_from_development": True,
            },
        }
    if dataset_id == "airbnb":
        return {
            "development": {
                "role": "repeatable_development_diagnostic",
                "cohort_hash_modulo": EXTERNAL_COHORT_MODULO,
                "cohort_hash_remainders": list(
                    EXTERNAL_DEVELOPMENT_REMAINDERS
                ),
            },
            "final": {
                "role": "one_time_sealed_external_evaluation",
                "cohort_hash_modulo": EXTERNAL_COHORT_MODULO,
                "cohort_hash_remainders": list(EXTERNAL_FINAL_REMAINDERS),
                "design": "static_label_disjoint_user_holdout",
                "disjoint_from_development": True,
            },
        }
    if dataset_id == "synerise":
        return {
            "development": {
                "role": "repeatable_development_diagnostic",
                "cohort_hash_modulo": EXTERNAL_COHORT_MODULO,
                "cohort_hash_remainders": list(
                    EXTERNAL_DEVELOPMENT_REMAINDERS
                ),
                "cutoffs": [
                    cutoff.isoformat()
                    for cutoff in SYNERISE_DEVELOPMENT_CUTOFFS
                ],
            },
            "final": {
                "role": "one_time_sealed_external_evaluation",
                "cohort_hash_modulo": EXTERNAL_COHORT_MODULO,
                "cohort_hash_remainders": list(EXTERNAL_FINAL_REMAINDERS),
                "cutoff": SYNERISE_FINAL_CUTOFF.isoformat(),
                "design": "later_window_disjoint_user_holdout",
                "disjoint_from_development": True,
            },
        }
    raise ValueError(f"unsupported external dataset: {dataset_id}")


def build_external_sealed_final_test_manifest(
    *,
    dataset_id: str,
    source_dir: Path,
    model_path: Path,
    model_metadata: Mapping[str, Any],
    adapter_config: ExternalAdapterConfig,
    backtest_config: ExternalBacktestConfig,
    code_commit: str,
    code_tree: str,
    criteria: ExternalFinalTestCriteria | None = None,
    created_at: datetime | None = None,
) -> ExternalSealedFinalTestManifest:
    _validate_final_adapter_config(dataset_id, adapter_config)
    if not model_path.is_file():
        raise ValueError(f"segment performance model not found: {model_path}")
    source_files = external_source_paths(
        dataset_id,
        source_dir,
        evaluation_role=EXTERNAL_SEALED_FINAL_ROLE,
    )
    source_descriptors = [
        source_file_descriptor(path, include_checksum=True)
        for path in source_files
    ]
    source_payload = {
        "files": source_descriptors,
        "fingerprint": _source_fingerprint(source_descriptors),
        "selection_uses_outcomes": False,
    }
    model_payload = {
        "file_name": model_path.name,
        "sha256": _file_sha256(model_path),
        "metadata": dict(model_metadata),
        "trained_on_external_data": False,
    }
    adapter_payload = _adapter_config_payload(adapter_config)
    backtest_payload = asdict(backtest_config)
    outcome_contract = {
        "prediction_error_comparable": (
            backtest_config.prediction_error_comparable
        ),
        "prediction_error_comparability_reason": (
            backtest_config.prediction_error_comparability_reason
        ),
        "external_data_updates_model_parameters": False,
        "primary_metrics": [
            "rank_one_beats_baseline_rate",
            "mean_rank_one_lift_percentage_points",
            "rank_one_is_best_rate",
            "rank_two_beats_baseline_rate",
            "rank_three_beats_baseline_rate",
            "pairwise_rank_accuracy",
            "pairwise_rank_tie_rate",
            "candidate_type_count",
            "mean_non_first_rank_overlap",
        ],
    }
    criteria_payload = asdict(criteria or ExternalFinalTestCriteria())
    base_payload: dict[str, Any] = {
        "version": EXTERNAL_SEALED_FINAL_TEST_VERSION,
        "dataset_id": dataset_id,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "source": source_payload,
        "model": model_payload,
        "adapter_config": adapter_payload,
        "backtest_config": backtest_payload,
        "partition_contract": external_partition_contract(dataset_id),
        "outcome_contract": outcome_contract,
        "acceptance_criteria": criteria_payload,
    }
    manifest_id = _json_sha256(base_payload)
    created_at_value = (created_at or datetime.now(UTC)).isoformat()
    integrity_payload = {
        **base_payload,
        "manifest_id": manifest_id,
        "created_at": created_at_value,
    }
    return ExternalSealedFinalTestManifest(
        manifest_id=manifest_id,
        integrity_sha256=_json_sha256(integrity_payload),
        created_at=created_at_value,
        dataset_id=dataset_id,
        code_commit=code_commit,
        code_tree=code_tree,
        source=source_payload,
        model=model_payload,
        adapter_config=adapter_payload,
        backtest_config=backtest_payload,
        partition_contract=base_payload["partition_contract"],
        outcome_contract=outcome_contract,
        acceptance_criteria=criteria_payload,
    )


def write_external_sealed_final_test_manifest(
    manifest: ExternalSealedFinalTestManifest,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as destination:
            json.dump(manifest.to_json(), destination, ensure_ascii=False, indent=2)
            destination.write("\n")
    except FileExistsError as exc:
        raise ExternalBacktestError(
            "external sealed manifest already exists; do not overwrite it"
        ) from exc


def load_external_sealed_final_test_manifest(
    path: Path,
) -> ExternalSealedFinalTestManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("external sealed manifest must be a JSON object")
    if payload.get("version") != EXTERNAL_SEALED_FINAL_TEST_VERSION:
        raise ValueError("unsupported external sealed manifest version")
    manifest = ExternalSealedFinalTestManifest(
        manifest_id=str(payload.get("manifest_id", "")),
        integrity_sha256=str(payload.get("integrity_sha256", "")),
        created_at=str(payload.get("created_at", "")),
        dataset_id=str(payload.get("dataset_id", "")),
        code_commit=str(payload.get("code_commit", "")),
        code_tree=str(payload.get("code_tree", "")),
        source=_mapping(payload, "source"),
        model=_mapping(payload, "model"),
        adapter_config=_mapping(payload, "adapter_config"),
        backtest_config=_mapping(payload, "backtest_config"),
        partition_contract=_mapping(payload, "partition_contract"),
        outcome_contract=_mapping(payload, "outcome_contract"),
        acceptance_criteria=_mapping(payload, "acceptance_criteria"),
    )
    serialized = manifest.to_json()
    expected_integrity = str(serialized.pop("integrity_sha256"))
    if _json_sha256(serialized) != expected_integrity:
        raise ValueError("external sealed manifest integrity check failed")
    stable_payload = dict(serialized)
    expected_manifest_id = str(stable_payload.pop("manifest_id"))
    stable_payload.pop("created_at")
    if _json_sha256(stable_payload) != expected_manifest_id:
        raise ValueError("external sealed manifest ID check failed")
    return manifest


def verify_external_sealed_final_test_runtime(
    manifest: ExternalSealedFinalTestManifest,
    *,
    source_dir: Path,
    model_path: Path,
    model_metadata: Mapping[str, Any],
    code_commit: str,
    code_tree: str,
) -> None:
    source_files = external_source_paths(
        manifest.dataset_id,
        source_dir,
        evaluation_role=EXTERNAL_SEALED_FINAL_ROLE,
    )
    source_descriptors = [
        source_file_descriptor(path, include_checksum=True)
        for path in source_files
    ]
    current_source = {
        "files": source_descriptors,
        "fingerprint": _source_fingerprint(source_descriptors),
        "selection_uses_outcomes": False,
    }
    if current_source != dict(manifest.source):
        raise ValueError("external sealed source data changed after sealing")
    if _file_sha256(model_path) != manifest.model.get("sha256"):
        raise ValueError("external sealed model changed after sealing")
    if dict(model_metadata) != manifest.model.get("metadata"):
        raise ValueError("external sealed model metadata changed after sealing")
    if code_commit != manifest.code_commit or code_tree != manifest.code_tree:
        raise ValueError("external sealed code changed after sealing")


def reserve_external_sealed_final_test_execution(
    manifest_path: Path,
    manifest: ExternalSealedFinalTestManifest,
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
        raise ExternalBacktestError(str(exc)) from exc


def run_external_sealed_final_test(
    *,
    manifest: ExternalSealedFinalTestManifest,
    source_dir: Path,
    performance_predictor: SegmentPerformancePredictor,
    on_outcomes_opened: Callable[[], None] | None = None,
) -> ExternalSealedFinalTestResult:
    adapter_config = _adapter_config_from_payload(manifest.adapter_config)
    backtest_config = ExternalBacktestConfig(**dict(manifest.backtest_config))
    bundle = load_external_dataset(
        manifest.dataset_id,
        source_dir,
        config=adapter_config,
        on_outcomes_opened=on_outcomes_opened,
    )
    run = run_external_backtest(
        bundle.cases,
        config=backtest_config,
        performance_predictor=performance_predictor,
    )
    if not run.results:
        raise ExternalBacktestError(
            "external sealed final test produced no segment candidates"
        )
    criteria_results = _evaluate_criteria(
        run.summary,
        manifest.acceptance_criteria,
    )
    verdict = determine_final_verdict(criteria_results)
    return ExternalSealedFinalTestResult(
        run=run,
        dataset_manifest=bundle.manifest,
        criteria_results=criteria_results,
        verdict=verdict,
        passed=verdict == VERDICT_PASSED,
    )


def write_external_sealed_final_test_artifacts(
    result: ExternalSealedFinalTestResult,
    *,
    manifest: ExternalSealedFinalTestManifest,
    output_dir: Path,
    model_metadata: Mapping[str, Any],
) -> dict[str, Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ExternalBacktestError(
            "external sealed final output already exists; it cannot be overwritten"
        ) from exc
    details = write_external_backtest_artifacts(
        result.run,
        manifest=result.dataset_manifest,
        output_dir=output_dir / "details",
        model_metadata=model_metadata,
    )
    summary_path = output_dir / "sealed_final_test_summary.json"
    report_path = output_dir / "sealed_final_test_report.md"
    summary = {
        "manifest_id": manifest.manifest_id,
        "manifest_integrity_sha256": manifest.integrity_sha256,
        "dataset_id": manifest.dataset_id,
        "scope": "one_time_sealed_external_evaluation",
        "verdict": result.verdict,
        "passed": result.passed,
        "metrics": dict(result.run.summary),
        "criteria_results": dict(result.criteria_results),
        "outcome_contract": dict(manifest.outcome_contract),
        "limitations": [
            "External outcomes do not update the Expedia-trained model.",
            "External lift is observational and does not prove causal ad lift.",
            "After inspection, this manifest must not be reused for tuning.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_sealed_report(summary), encoding="utf-8")
    return {
        "summary": summary_path,
        "report": report_path,
        "results": details["results"],
        "dataset_manifest": details["manifest"],
    }


def _validate_final_adapter_config(
    dataset_id: str,
    config: ExternalAdapterConfig,
) -> None:
    if config.evaluation_role != EXTERNAL_SEALED_FINAL_ROLE:
        raise ValueError("external final adapter must use the sealed final role")
    if dataset_id in {"airbnb", "synerise"}:
        if config.sample_modulo != EXTERNAL_COHORT_MODULO or (
            config.effective_sample_remainders != EXTERNAL_FINAL_REMAINDERS
        ):
            raise ValueError("external final cohort must use hash remainder 4 of 5")
    if dataset_id == "synerise" and config.cutoff != SYNERISE_FINAL_CUTOFF:
        raise ValueError("Synerise final cutoff must remain sealed at 2022-11-10")
    external_partition_contract(dataset_id)


def _evaluate_criteria(
    metrics: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_with_observed_outcome_count": criterion_result(
            _int_metric(metrics, "scenario_with_observed_outcome_count"),
            ">=",
            int(criteria["scenario_with_observed_outcome_count_min"]),
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
        "mean_rank_one_lift_percentage_points": criterion_result(
            _float_metric(metrics, "mean_rank_one_lift_percentage_points"),
            ">=",
            float(criteria["mean_rank_one_lift_percentage_points_min"]),
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
        "candidate_type_count": criterion_result(
            _int_metric(metrics, "candidate_type_count"),
            ">=",
            int(criteria["candidate_type_count_min"]),
            category=CRITERION_QUALITY,
        ),
        "mean_non_first_rank_overlap": criterion_result(
            _float_metric(metrics, "mean_non_first_rank_overlap"),
            "<=",
            float(criteria["mean_non_first_rank_overlap_max"]),
            category=CRITERION_QUALITY,
        ),
        "maximum_non_first_rank_overlap": criterion_result(
            _float_metric(metrics, "maximum_non_first_rank_overlap"),
            "<=",
            float(criteria["maximum_non_first_rank_overlap_max"]),
            category=CRITERION_QUALITY,
        ),
    }


def _adapter_config_payload(config: ExternalAdapterConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["sample_remainders"] = list(config.sample_remainders)
    payload["cutoff"] = config.cutoff.isoformat()
    return payload


def _adapter_config_from_payload(
    payload: Mapping[str, Any],
) -> ExternalAdapterConfig:
    values = dict(payload)
    values["sample_remainders"] = tuple(
        int(value) for value in values.get("sample_remainders", ())
    )
    values["cutoff"] = datetime.fromisoformat(str(values["cutoff"]))
    return ExternalAdapterConfig(**values)


def _source_fingerprint(
    descriptors: list[Mapping[str, Any]],
) -> str:
    return _json_sha256({"files": descriptors})


def _sealed_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        f"# {summary['dataset_id']} 외부 봉인 최종 평가",
        "",
        f"- Manifest ID: `{summary['manifest_id']}`",
        f"- 최종 판정: {_verdict_label(str(summary['verdict']))}",
        "- 외부 outcome으로 모델을 학습하거나 보정하지 않음",
        "",
        "## 핵심 지표",
        "",
        "- Rank 1 baseline 초과 비율: "
        f"{_format_optional_percent(metrics['rank_one_beats_baseline_rate'])}",
        "- Rank 1 평균 lift: "
        f"{_format_optional_percentage_points(metrics['mean_rank_one_lift_percentage_points'])}",
        "- Rank 2 baseline 초과 비율: "
        f"{_format_optional_percent(metrics['rank_two_beats_baseline_rate'])}",
        "- Rank 3 baseline 초과 비율: "
        f"{_format_optional_percent(metrics['rank_three_beats_baseline_rate'])}",
        "- Rank 1 엄격한 실제 최고 비율: "
        f"{_format_optional_percent(metrics['rank_one_is_best_rate'])}",
        "- 후보 쌍 순서 적중률: "
        f"{_format_optional_percent(metrics['pairwise_rank_accuracy'])}",
        "- 후보 쌍 동률 비율: "
        f"{_format_optional_percent(metrics['pairwise_rank_tie_rate'])}",
        "- 비교 가능한 시나리오: "
        f"{metrics['rank_comparable_scenario_count']}개",
        "- Rank 1·2·3이 모두 생성된 시나리오: "
        f"{metrics['three_rank_scenario_count']}개",
        "- 비교 가능한 예상값 오차: N/A",
        "",
        "## 사전 등록 기준",
        "",
    ]
    for name, criterion in summary["criteria_results"].items():
        status = "PASS" if criterion["passed"] else "FAIL"
        lines.append(
            f"- [{criterion['category']}] {name}: "
            f"{_format_criterion_value(criterion['actual'])} "
            f"{criterion['operator']} {criterion['threshold']} · {status}"
        )
    lines.extend(
        [
            "",
            "## 한계",
            "",
            "- 외부 결과는 관측 데이터 기반이며 광고의 인과적 증분 효과를 "
            "증명하지 않습니다.",
            "- 결과를 확인한 뒤 추천 로직을 수정하면 이 manifest는 최종 "
            "평가로 재사용할 수 없습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _float_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def _int_metric(metrics: Mapping[str, Any], key: str) -> int:
    return int(metrics.get(key, 0) or 0)


def _verdict_label(verdict: str) -> str:
    return {
        "passed": "PASS",
        "failed": "FAIL",
        "inconclusive": "INCONCLUSIVE (근거 부족)",
    }.get(verdict, verdict.upper())


def _format_optional_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _format_optional_percentage_points(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%p"


def _format_criterion_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"external sealed manifest field {key!r} must be an object")
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
