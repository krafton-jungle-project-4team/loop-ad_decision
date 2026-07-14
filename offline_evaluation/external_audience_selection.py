from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.analysis.raw_event_segments import (
    compile_raw_event_intent,
    generate_raw_event_segment_candidate_pool,
    generate_raw_event_segment_definitions,
)
from app.analysis.segment_performance import SegmentPerformancePredictor
from offline_evaluation.audience_selection import (
    AudienceSelectionEvaluationConfig,
    AudienceSelectionOutcome,
    evaluate_audience_selection_ratios,
    summarize_selection_ratios,
)
from offline_evaluation.external_backtest import (
    ExternalBacktestConfig,
    ExternalDatasetManifest,
    ExternalEvaluationCase,
)


EXTERNAL_AUDIENCE_SELECTION_DIAGNOSTIC_VERSION = (
    "external.audience-selection-diagnostic.v1"
)


@dataclass(frozen=True, slots=True)
class ExternalAudienceSelectionDiagnostic:
    dataset_id: str
    outcome_name: str
    evaluation_designs: tuple[str, ...]
    evaluation_keys: tuple[str, ...]
    current_runtime_ratio: float
    outcomes: tuple[AudienceSelectionOutcome, ...]
    summaries: tuple[Mapping[str, Any], ...]
    assessment: Mapping[str, Any]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "version": EXTERNAL_AUDIENCE_SELECTION_DIAGNOSTIC_VERSION,
            "role": "development_diagnostic",
            "dataset_id": self.dataset_id,
            "outcome_name": self.outcome_name,
            "evaluation_designs": list(self.evaluation_designs),
            "evaluation_keys": list(self.evaluation_keys),
            "ratio_grid": [
                float(summary["selection_ratio"])
                for summary in self.summaries
            ],
            "current_runtime_ratio": self.current_runtime_ratio,
            "updates_model_parameters": False,
            "updates_runtime_policy": False,
            "accesses_sealed_final_partition": False,
            "assessment": dict(self.assessment),
            "ratio_summaries": [dict(summary) for summary in self.summaries],
        }


def evaluate_external_audience_selection_cases(
    cases: Sequence[ExternalEvaluationCase],
    *,
    evaluation_key: str,
    backtest_config: ExternalBacktestConfig,
    selection_config: AudienceSelectionEvaluationConfig,
    performance_predictor: SegmentPerformancePredictor,
) -> tuple[AudienceSelectionOutcome, ...]:
    outcomes: list[AudienceSelectionOutcome] = []
    for case in cases:
        if len(case.profiles) < backtest_config.min_sample_size:
            continue
        compilation = compile_raw_event_intent(case.intent)
        all_matching_segments = generate_raw_event_segment_definitions(
            promotion=case.promotion,
            intent=case.intent,
            compilation=compilation,
            profiles=case.profiles,
            max_suggested_segments=backtest_config.max_suggested_segments,
            min_sample_size=backtest_config.min_sample_size,
            performance_predictor=performance_predictor,
        )
        if not all_matching_segments:
            continue
        all_matching_pool = generate_raw_event_segment_candidate_pool(
            promotion=case.promotion,
            intent=case.intent,
            compilation=compilation,
            profiles=case.profiles,
            min_sample_size=backtest_config.min_sample_size,
            performance_predictor=performance_predictor,
        )
        outcomes.extend(
            evaluate_audience_selection_ratios(
                evaluation_key=evaluation_key,
                scenario_id=case.scenario_id,
                positive_user_ids=case.positive_user_ids,
                promotion=case.promotion,
                intent=case.intent,
                compilation=compilation,
                profiles=case.profiles,
                all_matching_segments=all_matching_segments,
                all_matching_pool=all_matching_pool,
                performance_predictor=performance_predictor,
                config=selection_config,
                max_suggested_segments=(
                    backtest_config.max_suggested_segments
                ),
                min_sample_size=backtest_config.min_sample_size,
            )
        )
    return tuple(outcomes)


def build_external_audience_selection_diagnostic(
    *,
    dataset_id: str,
    outcome_name: str,
    evaluation_designs: Sequence[str],
    outcomes: Sequence[AudienceSelectionOutcome],
    selection_config: AudienceSelectionEvaluationConfig,
    current_runtime_ratio: float,
) -> ExternalAudienceSelectionDiagnostic:
    if current_runtime_ratio not in selection_config.ratios:
        raise ValueError("current_runtime_ratio must be included in the ratio grid")
    summaries = summarize_selection_ratios(
        outcomes,
        ratios=selection_config.ratios,
    )
    current = _summary_for_ratio(summaries, current_runtime_ratio)
    all_matching = _summary_for_ratio(summaries, 1.0)
    assessment = _assess_current_runtime_ratio(
        selected=current,
        all_matching=all_matching,
        config=selection_config,
    )
    return ExternalAudienceSelectionDiagnostic(
        dataset_id=dataset_id,
        outcome_name=outcome_name,
        evaluation_designs=tuple(dict.fromkeys(evaluation_designs)),
        evaluation_keys=tuple(
            dict.fromkeys(outcome.cutoff for outcome in outcomes)
        ),
        current_runtime_ratio=current_runtime_ratio,
        outcomes=tuple(outcomes),
        summaries=summaries,
        assessment=assessment,
    )


def write_external_audience_selection_diagnostic_artifacts(
    diagnostic: ExternalAudienceSelectionDiagnostic,
    *,
    manifest: ExternalDatasetManifest,
    model_metadata: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ratio_results.csv"
    summary_path = output_dir / "ratio_summary.json"
    report_path = output_dir / "report.md"

    _write_csv(
        results_path,
        [outcome.to_dict() for outcome in diagnostic.outcomes],
    )
    summary_path.write_text(
        json.dumps(
            {
                "dataset": manifest.to_dict(),
                "model": dict(model_metadata),
                "diagnostic": diagnostic.to_summary_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _markdown_report(
            diagnostic,
            manifest=manifest,
            model_metadata=model_metadata,
        ),
        encoding="utf-8",
    )
    return {
        "results": results_path,
        "summary": summary_path,
        "report": report_path,
    }


def _assess_current_runtime_ratio(
    *,
    selected: Mapping[str, Any],
    all_matching: Mapping[str, Any],
    config: AudienceSelectionEvaluationConfig,
) -> dict[str, Any]:
    applied = _mapping(selected.get("applied_only"))
    lift = _optional_float(
        applied.get("lift_vs_all_matching_percentage_points")
    )
    capture = _optional_float(applied.get("positive_capture_rate"))
    selected_rank_quality = _mapping(selected.get("rank_quality"))
    all_matching_rank_quality = _mapping(all_matching.get("rank_quality"))
    selected_pairwise = _optional_float(
        selected_rank_quality.get("pairwise_rank_accuracy")
    )
    all_matching_pairwise = _optional_float(
        all_matching_rank_quality.get("pairwise_rank_accuracy")
    )
    rank_check = (
        None
        if selected_pairwise is None or all_matching_pairwise is None
        else selected_pairwise
        + config.maximum_pairwise_rank_accuracy_drop
        >= all_matching_pairwise
    )
    checks: dict[str, bool | None] = {
        "minimum_policy_applied_results": (
            int(selected.get("policy_applied_result_count", 0) or 0)
            >= config.minimum_policy_applied_result_count
        ),
        "nonnegative_lift_vs_all_matching": (
            None if lift is None else lift >= 0.0
        ),
        "minimum_positive_capture": (
            None
            if capture is None
            else capture >= config.minimum_positive_capture_rate
        ),
        "rank_accuracy_not_materially_worse": rank_check,
    }
    mandatory = (
        checks["minimum_policy_applied_results"],
        checks["nonnegative_lift_vs_all_matching"],
        checks["minimum_positive_capture"],
    )
    if any(value is None for value in mandatory) or not mandatory[0]:
        status = "insufficient_evidence"
    elif any(value is False for value in checks.values()):
        status = "caution"
    else:
        status = "supported"
    return {
        "status": status,
        "interpretation": (
            "외부 개발 outcome에서 현재 비율의 강건성을 진단한 결과이며 "
            "운영 비율을 자동 변경하지 않습니다."
        ),
        "checks": checks,
        "minimum_policy_applied_result_count": (
            config.minimum_policy_applied_result_count
        ),
        "minimum_positive_capture_rate": config.minimum_positive_capture_rate,
        "maximum_pairwise_rank_accuracy_drop": (
            config.maximum_pairwise_rank_accuracy_drop
        ),
        "selected_ratio_metrics": dict(selected),
        "all_matching_metrics": dict(all_matching),
    }


def _summary_for_ratio(
    summaries: Sequence[Mapping[str, Any]],
    ratio: float,
) -> Mapping[str, Any]:
    for summary in summaries:
        if float(summary["selection_ratio"]) == ratio:
            return summary
    raise ValueError(f"selection ratio summary not found: {ratio}")


def _markdown_report(
    diagnostic: ExternalAudienceSelectionDiagnostic,
    *,
    manifest: ExternalDatasetManifest,
    model_metadata: Mapping[str, Any],
) -> str:
    lines = [
        f"# {diagnostic.dataset_id} 추천 대상 비율 개발 진단",
        "",
        "## 진단 계약",
        "",
        f"- 외부 outcome: `{diagnostic.outcome_name}`",
        f"- 평가 설계: `{', '.join(diagnostic.evaluation_designs)}`",
        f"- 예측 모델: `{model_metadata.get('model_version', 'unknown')}`",
        "- 사용자 정렬 입력: 관찰 구간 행동 신호만 사용",
        "- 모델 파라미터 갱신: 아니오",
        "- 운영 비율 자동 변경: 아니오",
        "- 봉인 final partition 접근: 아니오",
        "",
        "## 현재 운영 비율 진단",
        "",
        f"- 비교 비율: {diagnostic.current_runtime_ratio * 100:g}%",
        f"- 판정: `{diagnostic.assessment['status']}`",
        "- Rank 순서 판정: `"
        + (
            "not_evaluable"
            if diagnostic.assessment["checks"][
                "rank_accuracy_not_materially_worse"
            ]
            is None
            else (
                "supported"
                if diagnostic.assessment["checks"][
                    "rank_accuracy_not_materially_worse"
                ]
                else "caution"
            )
        )
        + "`",
        f"- 해석: {diagnostic.assessment['interpretation']}",
        "",
        "## 비율별 결과",
        "",
        "| 비율 | 적용 후보 | 선택/조건 일치 | 실제 outcome | 전체 일치 outcome | lift | positive 포착률 | reach | 안정 표본 | 후보/시나리오 | Rank pairwise 정확도 | 후순위 중복도 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in diagnostic.summaries:
        displayed_metrics = _display_metrics(summary)
        rank_quality = _mapping(summary.get("rank_quality"))
        lines.append(
            "| "
            + " | ".join(
                [
                    _percent(summary.get("selection_ratio")),
                    str(summary.get("policy_applied_result_count", 0)),
                    (
                        f"{summary.get('selected_user_count', 0)}/"
                        f"{summary.get('matching_user_count', 0)}"
                    ),
                    _percent(displayed_metrics.get("actual_goal_rate")),
                    _percent(
                        displayed_metrics.get("all_matching_goal_rate")
                    ),
                    _points(
                        displayed_metrics.get(
                            "lift_vs_all_matching_percentage_points"
                        )
                    ),
                    _percent(displayed_metrics.get("positive_capture_rate")),
                    _percent(displayed_metrics.get("reach_within_matching")),
                    _percent(summary.get("sample_stable_result_rate")),
                    _number(summary.get("mean_candidate_count_per_scenario")),
                    _percent(rank_quality.get("pairwise_rank_accuracy")),
                    _percent(summary.get("mean_non_first_rank_overlap")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 해석 주의",
            "",
            "- 외부 데이터셋마다 outcome 정의가 달라 절대 전환율을 서로 합산하거나 직접 비교하지 않습니다.",
            "- 이 결과는 추천 대상 비율의 일반화 가능성을 살피는 개발 진단이며 Expedia 모델을 재학습하지 않습니다.",
            "- 운영 비율 변경은 Expedia 개발·검증 결과와 외부 진단을 함께 검토한 별도 변경으로 진행해야 합니다.",
            f"- 데이터셋 지원 주장: {', '.join(manifest.supported_claims)}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _display_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if int(summary.get("policy_applied_result_count", 0) or 0) > 0:
        return _mapping(summary.get("applied_only"))
    return {
        "actual_goal_rate": summary.get("pooled_actual_goal_rate"),
        "all_matching_goal_rate": summary.get(
            "pooled_all_matching_goal_rate"
        ),
        "lift_vs_all_matching_percentage_points": summary.get(
            "lift_vs_all_matching_percentage_points"
        ),
        "positive_capture_rate": summary.get("positive_capture_rate"),
        "reach_within_matching": summary.get("reach_within_matching"),
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent(value: object) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{parsed * 100:.2f}%"


def _points(value: object) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{parsed:+.2f}%p"


def _number(value: object) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{parsed:.2f}"
