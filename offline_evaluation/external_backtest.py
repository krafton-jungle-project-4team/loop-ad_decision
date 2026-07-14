from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from app.analysis.raw_event_segments import (
    PromotionIntent,
    compile_raw_event_intent,
    generate_raw_event_segment_definitions,
)
from app.analysis.repositories import PromotionRecord, RawEventUserSignalRecord
from app.analysis.segment_performance import SegmentPerformancePredictor
from offline_evaluation.rank_quality import RankedOutcome, summarize_rank_quality


EXTERNAL_BACKTEST_VERSION = "external.segment-backtest.v3"


class ExternalBacktestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalBacktestConfig:
    max_suggested_segments: int = 3
    min_sample_size: int = 20
    prediction_error_comparable: bool = False
    prediction_error_comparability_reason: str = (
        "external outcome semantics differ from the Expedia calibration target"
    )

    def __post_init__(self) -> None:
        if self.max_suggested_segments <= 0:
            raise ValueError("max_suggested_segments must be positive")
        if self.min_sample_size <= 0:
            raise ValueError("min_sample_size must be positive")


@dataclass(frozen=True, slots=True)
class ExternalDatasetManifest:
    dataset_id: str
    source_version: str
    evaluation_design: str
    outcome_name: str
    supports_temporal_holdout: bool
    supported_claims: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    signal_mappings: Mapping[str, Mapping[str, str]]
    source_files: tuple[Mapping[str, Any], ...]
    supported_claim_ids: tuple[str, ...] = ()
    unsupported_claim_ids: tuple[str, ...] = ()
    verdict_scope: str = "supporting_evidence"
    evaluation_role: str = "development_diagnostic"
    prediction_error_comparable: bool = False
    prediction_error_comparability_reason: str = (
        "external outcome semantics differ from the Expedia calibration target"
    )
    notes: tuple[str, ...] = ()
    version: str = EXTERNAL_BACKTEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_id": self.dataset_id,
            "source_version": self.source_version,
            "evaluation_design": self.evaluation_design,
            "outcome_name": self.outcome_name,
            "supports_temporal_holdout": self.supports_temporal_holdout,
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "supported_claim_ids": list(self.supported_claim_ids),
            "unsupported_claim_ids": list(self.unsupported_claim_ids),
            "verdict_scope": self.verdict_scope,
            "signal_mappings": {
                key: dict(value) for key, value in self.signal_mappings.items()
            },
            "source_files": [dict(value) for value in self.source_files],
            "evaluation_role": self.evaluation_role,
            "prediction_error_comparable": self.prediction_error_comparable,
            "prediction_error_comparability_reason": (
                self.prediction_error_comparability_reason
            ),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ExternalEvaluationCase:
    dataset_id: str
    scenario_id: str
    target_value: str
    target_label: str
    outcome_name: str
    evaluation_design: str
    profiles: tuple[RawEventUserSignalRecord, ...]
    positive_user_ids: frozenset[str]
    promotion: PromotionRecord
    intent: PromotionIntent

    def __post_init__(self) -> None:
        user_ids = [profile.user_id for profile in self.profiles]
        if not user_ids:
            raise ValueError("external evaluation case requires profiles")
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("external evaluation profile user IDs must be unique")
        unknown_positive_ids = self.positive_user_ids - set(user_ids)
        if unknown_positive_ids:
            raise ValueError("positive outcome IDs must belong to the profile pool")


@dataclass(frozen=True, slots=True)
class ExternalBacktestResult:
    dataset_id: str
    evaluation_design: str
    outcome_name: str
    scenario_id: str
    target_value: str
    target_label: str
    rank: int
    segment_id: str
    candidate_type: str
    rank_role: str
    sample_size: int
    total_eligible_user_count: int
    positive_user_count: int
    predicted_outcome_rate: float
    actual_outcome_rate: float
    baseline_outcome_rate: float
    absolute_lift_percentage_points: float
    relative_lift: float | None
    prediction_error_comparable: bool
    absolute_prediction_error_percentage_points: float | None
    recommendation_score: float
    maximum_prior_rank_overlap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExternalBacktestRun:
    results: tuple[ExternalBacktestResult, ...]
    skipped_scenarios: tuple[Mapping[str, str], ...]
    summary: Mapping[str, Any]


def run_external_backtest(
    cases: Sequence[ExternalEvaluationCase],
    *,
    config: ExternalBacktestConfig,
    performance_predictor: SegmentPerformancePredictor,
) -> ExternalBacktestRun:
    results: list[ExternalBacktestResult] = []
    skipped: list[Mapping[str, str]] = []
    for case in cases:
        if len(case.profiles) < config.min_sample_size:
            skipped.append(
                {
                    "dataset_id": case.dataset_id,
                    "scenario_id": case.scenario_id,
                    "reason": "insufficient_profiles",
                }
            )
            continue
        compilation = compile_raw_event_intent(case.intent)
        segments = generate_raw_event_segment_definitions(
            promotion=case.promotion,
            intent=case.intent,
            compilation=compilation,
            profiles=case.profiles,
            max_suggested_segments=config.max_suggested_segments,
            min_sample_size=config.min_sample_size,
            performance_predictor=performance_predictor,
        )
        if not segments:
            skipped.append(
                {
                    "dataset_id": case.dataset_id,
                    "scenario_id": case.scenario_id,
                    "reason": "no_segment_candidates",
                }
            )
            continue
        results.extend(_evaluate_case(case, segments, config=config))
    response_results = tuple(results)
    return ExternalBacktestRun(
        results=response_results,
        skipped_scenarios=tuple(skipped),
        summary=summarize_external_backtest(
            response_results,
            skipped,
            config=config,
        ),
    )


def _evaluate_case(
    case: ExternalEvaluationCase,
    segments: Sequence[Any],
    *,
    config: ExternalBacktestConfig,
) -> list[ExternalBacktestResult]:
    eligible_ids = {profile.user_id for profile in case.profiles}
    baseline_rate = _safe_rate(len(case.positive_user_ids), len(eligible_ids))
    prior_user_sets: list[set[str]] = []
    results: list[ExternalBacktestResult] = []
    for rank, segment in enumerate(segments, start=1):
        candidate_ids = set(segment.rule_json.get("candidate_user_ids", ()))
        candidate_ids &= eligible_ids
        positive_count = len(candidate_ids & set(case.positive_user_ids))
        actual_rate = _safe_rate(positive_count, len(candidate_ids))
        performance_estimate = segment.profile_json.get("performance_estimate", {})
        predicted_rate = float(performance_estimate.get("value", 0.0) or 0.0)
        maximum_overlap = max(
            (_jaccard(candidate_ids, previous) for previous in prior_user_sets),
            default=0.0,
        )
        prior_user_sets.append(candidate_ids)
        relative_lift = (
            actual_rate / baseline_rate - 1.0 if baseline_rate > 0 else None
        )
        results.append(
            ExternalBacktestResult(
                dataset_id=case.dataset_id,
                evaluation_design=case.evaluation_design,
                outcome_name=case.outcome_name,
                scenario_id=case.scenario_id,
                target_value=case.target_value,
                target_label=case.target_label,
                rank=rank,
                segment_id=segment.segment_id,
                candidate_type=str(
                    segment.profile_json.get("candidate_type", "unknown")
                ),
                rank_role=str(segment.profile_json.get("rank_role", "")),
                sample_size=len(candidate_ids),
                total_eligible_user_count=len(eligible_ids),
                positive_user_count=positive_count,
                predicted_outcome_rate=predicted_rate,
                actual_outcome_rate=actual_rate,
                baseline_outcome_rate=baseline_rate,
                absolute_lift_percentage_points=(
                    actual_rate - baseline_rate
                )
                * 100.0,
                relative_lift=relative_lift,
                prediction_error_comparable=config.prediction_error_comparable,
                absolute_prediction_error_percentage_points=(
                    abs(predicted_rate - actual_rate) * 100.0
                    if config.prediction_error_comparable
                    else None
                ),
                recommendation_score=float(
                    segment.profile_json.get("recommendation_score", 0.0) or 0.0
                ),
                maximum_prior_rank_overlap=maximum_overlap,
            )
        )
    return results


def summarize_external_backtest(
    results: Sequence[ExternalBacktestResult],
    skipped_scenarios: Sequence[Mapping[str, str]] = (),
    *,
    config: ExternalBacktestConfig | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ExternalBacktestResult]] = {}
    for result in results:
        grouped.setdefault((result.dataset_id, result.scenario_id), []).append(result)
    rank_quality = summarize_rank_quality(
        [
            [
                RankedOutcome(
                    rank=result.rank,
                    actual_rate=result.actual_outcome_rate,
                    baseline_rate=result.baseline_outcome_rate,
                )
                for result in scenario_results
            ]
            for scenario_results in grouped.values()
        ]
    )
    candidate_types = {result.candidate_type for result in results}
    non_first_overlaps = [
        result.maximum_prior_rank_overlap for result in results if result.rank > 1
    ]
    comparable_errors = [
        result.absolute_prediction_error_percentage_points
        for result in results
        if result.prediction_error_comparable
        and result.absolute_prediction_error_percentage_points is not None
    ]
    prediction_error_comparable = bool(comparable_errors)
    mean_portfolio_candidate_overlap = _mean(non_first_overlaps)
    maximum_portfolio_candidate_overlap = max(non_first_overlaps, default=0.0)
    return {
        "version": EXTERNAL_BACKTEST_VERSION,
        "scenario_count": len(grouped),
        "candidate_result_count": len(results),
        "skipped_scenario_count": len(skipped_scenarios),
        "scenario_with_observed_outcome_count": rank_quality[
            "observed_outcome_scenario_count"
        ],
        **rank_quality,
        "prediction_error_comparable": prediction_error_comparable,
        "prediction_error_comparability_reason": (
            config.prediction_error_comparability_reason
            if config is not None and not prediction_error_comparable
            else None
        ),
        "mean_absolute_prediction_error_percentage_points": (
            _mean(comparable_errors) if comparable_errors else None
        ),
        "mean_portfolio_candidate_overlap": mean_portfolio_candidate_overlap,
        "maximum_portfolio_candidate_overlap": maximum_portfolio_candidate_overlap,
        # Deprecated diagnostic aliases retained for existing artifacts.
        "mean_non_first_rank_overlap": mean_portfolio_candidate_overlap,
        "maximum_non_first_rank_overlap": maximum_portfolio_candidate_overlap,
        "candidate_type_count": len(candidate_types),
        "candidate_type_diversity_rate": _safe_rate(
            len(candidate_types),
            len(results),
        ),
    }


def write_external_backtest_artifacts(
    run: ExternalBacktestRun,
    *,
    manifest: ExternalDatasetManifest,
    output_dir: Path,
    model_metadata: Mapping[str, Any],
) -> Mapping[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_results_csv(results_path, run.results)
    summary_payload = {
        "dataset": manifest.to_dict(),
        "model": dict(model_metadata),
        "metrics": dict(run.summary),
        "skipped_scenarios": [dict(value) for value in run.skipped_scenarios],
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _markdown_report(run, manifest=manifest, model_metadata=model_metadata),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "results": results_path,
        "summary": summary_path,
        "report": report_path,
    }


def source_file_descriptor(path: Path, *, include_checksum: bool) -> dict[str, Any]:
    if not path.is_file():
        raise ExternalBacktestError(f"source file not found: {path}")
    descriptor: dict[str, Any] = {
        "name": path.name,
        "size_bytes": path.stat().st_size,
    }
    if include_checksum:
        descriptor["sha256"] = _file_sha256(path)
    return descriptor


def stable_bucket(value: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("sample modulo must be positive")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def stable_score(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _write_results_csv(
    path: Path,
    results: Sequence[ExternalBacktestResult],
) -> None:
    fieldnames = [field.name for field in ExternalBacktestResult.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)


def _markdown_report(
    run: ExternalBacktestRun,
    *,
    manifest: ExternalDatasetManifest,
    model_metadata: Mapping[str, Any],
) -> str:
    metrics = run.summary
    lines = [
        f"# {manifest.dataset_id} 외부 세그먼트 추천 검증",
        "",
        "## 검증 계약",
        "",
        f"- 평가 설계: `{manifest.evaluation_design}`",
        f"- 실제 결과 지표: `{manifest.outcome_name}`",
        f"- 시간 기반 홀드아웃: `{'yes' if manifest.supports_temporal_holdout else 'no'}`",
        f"- 예측 모델: `{model_metadata.get('model_version', 'unknown')}`",
        f"- 모델 학습 데이터: `{model_metadata.get('training_dataset', 'unknown')}`",
        f"- 모델 적용 범위: `{model_metadata.get('applicability_scope', 'unknown')}`",
        "",
        "## 결과",
        "",
        f"- 평가 시나리오: {metrics['scenario_count']}",
        f"- 후보 결과: {metrics['candidate_result_count']}",
        "- 전체 추천 후보 baseline 초과 비율: "
        f"{_format_optional_percent(metrics['portfolio_candidate_beats_baseline_rate'])}",
        "- 유용한 후보가 하나 이상인 시나리오 비율: "
        f"{_format_optional_percent(metrics['portfolio_scenario_any_candidate_beats_baseline_rate'])}",
        "- 모든 후보가 baseline을 넘은 시나리오 비율: "
        f"{_format_optional_percent(metrics['portfolio_scenario_all_candidates_beat_baseline_rate'])}",
        "- 추천 후보 평균 lift: "
        f"{_format_optional_lift(metrics['portfolio_mean_candidate_lift_percentage_points'])}",
        "- 시나리오별 최저 성과 후보의 평균 lift: "
        f"{_format_optional_lift(metrics['portfolio_mean_worst_candidate_lift_percentage_points'])}",
        "- 후보가 2개 이상인 시나리오: "
        f"{metrics['portfolio_multi_candidate_scenario_count']}",
        "- 후보가 3개인 시나리오: "
        f"{metrics['portfolio_three_candidate_scenario_count']}",
        "- 예상값과 외부 outcome의 평균 절대 차이: "
        f"{_format_optional_percentage_points(metrics['mean_absolute_prediction_error_percentage_points'])}",
        "- 예상값 오차 비교 가능 여부: "
        f"{'yes' if metrics['prediction_error_comparable'] else 'no'}",
        "- 후보 간 평균 사용자 중복도: "
        f"{_format_percent(metrics['mean_portfolio_candidate_overlap'])}",
        "",
        "## 이 데이터로 평가할 수 있는 항목",
        "",
        *[f"- {value}" for value in manifest.supported_claims],
        "",
        "## 이 데이터로 검증할 수 없는 주장",
        "",
        *[f"- {value}" for value in manifest.unsupported_claims],
        "",
    ]
    if manifest.notes:
        lines.extend(
            ["## 주의", "", *[f"- {value}" for value in manifest.notes], ""]
        )
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return mean(materialized) if materialized else 0.0


def _format_optional_percentage_points(value: object) -> str:
    if value is None:
        return "N/A (outcome definition differs from the calibration target)"
    return f"{float(value):.2f}%p"


def _format_optional_lift(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}%p"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return _format_percent(value)
