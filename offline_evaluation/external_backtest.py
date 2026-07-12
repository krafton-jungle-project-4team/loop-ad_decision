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


EXTERNAL_BACKTEST_VERSION = "external.segment-backtest.v1"


class ExternalBacktestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalBacktestConfig:
    max_suggested_segments: int = 3
    min_sample_size: int = 20

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
            "signal_mappings": {
                key: dict(value) for key, value in self.signal_mappings.items()
            },
            "source_files": [dict(value) for value in self.source_files],
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
    absolute_prediction_error_percentage_points: float
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
        results.extend(_evaluate_case(case, segments))
    response_results = tuple(results)
    return ExternalBacktestRun(
        results=response_results,
        skipped_scenarios=tuple(skipped),
        summary=summarize_external_backtest(response_results, skipped),
    )


def _evaluate_case(
    case: ExternalEvaluationCase,
    segments: Sequence[Any],
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
                absolute_prediction_error_percentage_points=abs(
                    predicted_rate - actual_rate
                )
                * 100.0,
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
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ExternalBacktestResult]] = {}
    for result in results:
        grouped.setdefault((result.dataset_id, result.scenario_id), []).append(result)
    rank_one_results: list[ExternalBacktestResult] = []
    rank_comparable_scenarios: list[list[ExternalBacktestResult]] = []
    rank_one_best_count = 0
    for scenario_results in grouped.values():
        ordered = sorted(scenario_results, key=lambda value: value.rank)
        rank_one = ordered[0]
        rank_one_results.append(rank_one)
        if len(ordered) < 2 or not any(
            value.positive_user_count > 0 for value in ordered
        ):
            continue
        rank_comparable_scenarios.append(ordered)
        if rank_one.actual_outcome_rate >= max(
            value.actual_outcome_rate for value in ordered[1:]
        ):
            rank_one_best_count += 1
    candidate_types = {result.candidate_type for result in results}
    non_first_overlaps = [
        result.maximum_prior_rank_overlap for result in results if result.rank > 1
    ]
    return {
        "version": EXTERNAL_BACKTEST_VERSION,
        "scenario_count": len(grouped),
        "candidate_result_count": len(results),
        "skipped_scenario_count": len(skipped_scenarios),
        "rank_comparable_scenario_count": len(rank_comparable_scenarios),
        "scenario_with_observed_outcome_count": sum(
            result.baseline_outcome_rate > 0 for result in rank_one_results
        ),
        "rank_one_beats_baseline_rate": _safe_rate(
            sum(
                result.actual_outcome_rate > result.baseline_outcome_rate
                for result in rank_one_results
            ),
            len(rank_one_results),
        ),
        "rank_one_is_best_rate": (
            _safe_rate(rank_one_best_count, len(rank_comparable_scenarios))
            if rank_comparable_scenarios
            else None
        ),
        "mean_rank_one_lift_percentage_points": _mean(
            result.absolute_lift_percentage_points for result in rank_one_results
        ),
        "mean_absolute_prediction_error_percentage_points": _mean(
            result.absolute_prediction_error_percentage_points for result in results
        ),
        "mean_non_first_rank_overlap": _mean(non_first_overlaps),
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
        "- Rank 1 baseline 초과 비율: "
        f"{_format_percent(metrics['rank_one_beats_baseline_rate'])}",
        f"- Rank 비교 가능 시나리오: {metrics['rank_comparable_scenario_count']}",
        "- Rank 1 실제 최고 성과 비율: "
        f"{_format_optional_percent(metrics['rank_one_is_best_rate'])}",
        "- Rank 1 평균 lift: "
        f"{metrics['mean_rank_one_lift_percentage_points']:.2f}%p",
        "- 예상값과 외부 outcome의 평균 절대 차이: "
        f"{metrics['mean_absolute_prediction_error_percentage_points']:.2f}%p",
        "- 후순위 평균 사용자 중복도: "
        f"{_format_percent(metrics['mean_non_first_rank_overlap'])}",
        "",
        "## 해석 가능한 주장",
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


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return _format_percent(value)
