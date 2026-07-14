from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.analysis.audience_selection import (
    AUDIENCE_SELECTION_ARTIFACT_SCHEMA_VERSION,
    write_audience_selection_artifact,
)
from offline_evaluation.rank_quality import RankedOutcome, summarize_rank_quality


DEFAULT_SELECTION_RATIOS = (0.2, 0.4, 0.6, 0.8, 1.0)
AUDIENCE_SELECTION_EVALUATION_VERSION = (
    "offline.expedia-audience-selection.v1"
)


@dataclass(frozen=True, slots=True)
class AudienceSelectionEvaluationConfig:
    ratios: tuple[float, ...] = DEFAULT_SELECTION_RATIOS
    minimum_selected_user_count: int = 30
    minimum_policy_applied_result_count: int = 3
    minimum_positive_capture_rate: float = 0.8
    goal_metric: str = "booking_conversion_rate"

    def __post_init__(self) -> None:
        if not self.ratios:
            raise ValueError("audience selection ratios must not be empty")
        if tuple(sorted(set(self.ratios))) != self.ratios:
            raise ValueError("audience selection ratios must be sorted and unique")
        if any(not 0.0 < ratio <= 1.0 for ratio in self.ratios):
            raise ValueError("audience selection ratios must be in (0, 1]")
        if self.ratios[-1] != 1.0:
            raise ValueError("audience selection ratios must include 1.0")
        if self.minimum_selected_user_count <= 0:
            raise ValueError("minimum_selected_user_count must be positive")
        if self.minimum_policy_applied_result_count <= 0:
            raise ValueError(
                "minimum_policy_applied_result_count must be positive"
            )
        if not 0.0 <= self.minimum_positive_capture_rate <= 1.0:
            raise ValueError("minimum_positive_capture_rate must be in [0, 1]")
        if not self.goal_metric.strip():
            raise ValueError("goal_metric must not be empty")


@dataclass(frozen=True, slots=True)
class AudienceSelectionOutcome:
    cutoff: str
    scenario_id: str
    selection_ratio: float
    rank: int
    candidate_type: str
    matching_user_count: int
    selected_user_count: int
    matching_positive_user_count: int
    selected_positive_user_count: int
    baseline_user_count: int
    baseline_positive_user_count: int
    predicted_goal_rate: float
    actual_goal_rate: float
    all_matching_goal_rate: float
    baseline_goal_rate: float
    lift_vs_all_matching_percentage_points: float
    lift_vs_baseline_percentage_points: float
    positive_capture_rate: float | None
    reach_within_matching: float
    policy_applied: bool
    sample_stable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AudienceSelectionPolicyEvaluation:
    artifact: Mapping[str, Any]
    development_summaries: tuple[Mapping[str, Any], ...]
    validation_summaries: tuple[Mapping[str, Any], ...]


def summarize_selection_ratios(
    outcomes: Sequence[AudienceSelectionOutcome],
    *,
    ratios: Sequence[float],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _summarize_ratio(
            [outcome for outcome in outcomes if outcome.selection_ratio == ratio],
            ratio=ratio,
        )
        for ratio in ratios
    )


def build_audience_selection_policy_evaluation(
    *,
    development_outcomes: Sequence[AudienceSelectionOutcome],
    validation_outcomes: Sequence[AudienceSelectionOutcome],
    config: AudienceSelectionEvaluationConfig,
    development_split: str,
    validation_split: str,
) -> AudienceSelectionPolicyEvaluation:
    development_summaries = summarize_selection_ratios(
        development_outcomes,
        ratios=config.ratios,
    )
    validation_summaries = summarize_selection_ratios(
        validation_outcomes,
        ratios=config.ratios,
    )
    development_choice = _choose_development_ratio(
        development_summaries,
        config=config,
    )
    chosen_ratio = float(development_choice["selection_ratio"])
    validation_choice = _summary_for_ratio(validation_summaries, chosen_ratio)
    validation_all_matching = _summary_for_ratio(validation_summaries, 1.0)
    validation_criteria = _validation_criteria(
        selected=validation_choice,
        all_matching=validation_all_matching,
        config=config,
    )
    validation_passed = all(validation_criteria.values())
    selected_ratio = chosen_ratio if validation_passed else 1.0
    calibration_status = (
        "validated"
        if validation_passed and selected_ratio < 1.0
        else "fallback_all_matching"
    )
    input_hash = _evaluation_input_hash(
        development_outcomes=development_outcomes,
        validation_outcomes=validation_outcomes,
        config=config,
        development_split=development_split,
        validation_split=validation_split,
    )
    policy_version = f"expedia-booking-audience-selection.{input_hash[:12]}"
    artifact: dict[str, Any] = {
        "schema_version": AUDIENCE_SELECTION_ARTIFACT_SCHEMA_VERSION,
        "policy_version": policy_version,
        "calibration_status": calibration_status,
        "selection_method": "top_behavior_strength_ratio",
        "rules": [
            {
                "goal_metric": config.goal_metric,
                "selected_ratio": selected_ratio,
                "minimum_selected_user_count": (
                    config.minimum_selected_user_count
                ),
                "candidate_types": [],
            }
        ],
        "provenance": {
            "evaluation_version": AUDIENCE_SELECTION_EVALUATION_VERSION,
            "dataset": "expedia_hotel_recommendations_train",
            "development_split": development_split,
            "validation_split": validation_split,
            "final_test": "not_run",
            "candidate_scope": "all_raw_event_candidate_types",
            "ordering_input": "observation_window_behavior_strength_only",
            "outcome_target": "future_contextual_booking_rate",
            "selection_uses_future_outcomes_at_runtime": False,
            "input_hash": input_hash,
        },
        "selection": {
            "development_chosen_ratio": chosen_ratio,
            "runtime_selected_ratio": selected_ratio,
            "validation_passed": validation_passed,
            "validation_criteria": validation_criteria,
            "minimum_positive_capture_rate": (
                config.minimum_positive_capture_rate
            ),
            "minimum_policy_applied_result_count": (
                config.minimum_policy_applied_result_count
            ),
        },
        "development_metrics": development_choice,
        "validation_metrics": validation_choice,
        "validation_all_matching_metrics": validation_all_matching,
        "ratio_grid": list(config.ratios),
    }
    return AudienceSelectionPolicyEvaluation(
        artifact=artifact,
        development_summaries=development_summaries,
        validation_summaries=validation_summaries,
    )


def write_audience_selection_evaluation_artifacts(
    evaluation: AudienceSelectionPolicyEvaluation,
    *,
    development_outcomes: Sequence[AudienceSelectionOutcome],
    validation_outcomes: Sequence[AudienceSelectionOutcome],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    development_path = output_dir / "development_ratio_results.csv"
    validation_path = output_dir / "validation_ratio_results.csv"
    summary_path = output_dir / "ratio_summary.json"
    report_path = output_dir / "report.md"
    policy_path = output_dir / "audience_selection_policy_v1.json"

    _write_csv(development_path, [item.to_dict() for item in development_outcomes])
    _write_csv(validation_path, [item.to_dict() for item in validation_outcomes])
    finalized_artifact = write_audience_selection_artifact(
        evaluation.artifact,
        policy_path,
    )
    summary_path.write_text(
        json.dumps(
            {
                "development": list(evaluation.development_summaries),
                "validation": list(evaluation.validation_summaries),
                "policy": finalized_artifact,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _selection_markdown_report(
            evaluation=evaluation,
            finalized_artifact=finalized_artifact,
        ),
        encoding="utf-8",
    )
    return {
        "development_results": development_path,
        "validation_results": validation_path,
        "summary": summary_path,
        "report": report_path,
        "policy": policy_path,
    }


def _summarize_ratio(
    outcomes: Sequence[AudienceSelectionOutcome],
    *,
    ratio: float,
) -> dict[str, Any]:
    applied = [item for item in outcomes if item.policy_applied]
    grouped: dict[tuple[str, str], list[AudienceSelectionOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[(outcome.cutoff, outcome.scenario_id)].append(outcome)
    rank_quality = summarize_rank_quality(
        [
            [
                RankedOutcome(
                    rank=item.rank,
                    actual_rate=item.actual_goal_rate,
                    baseline_rate=item.baseline_goal_rate,
                )
                for item in scenario
            ]
            for scenario in grouped.values()
        ]
    )
    selected_users = sum(item.selected_user_count for item in outcomes)
    matching_users = sum(item.matching_user_count for item in outcomes)
    selected_positives = sum(
        item.selected_positive_user_count for item in outcomes
    )
    matching_positives = sum(
        item.matching_positive_user_count for item in outcomes
    )
    applied_selected_users = sum(item.selected_user_count for item in applied)
    applied_matching_users = sum(item.matching_user_count for item in applied)
    applied_selected_positives = sum(
        item.selected_positive_user_count for item in applied
    )
    applied_matching_positives = sum(
        item.matching_positive_user_count for item in applied
    )
    pooled_actual_rate = _optional_rate(selected_positives, selected_users)
    pooled_all_matching_rate = _optional_rate(matching_positives, matching_users)
    applied_actual_rate = _optional_rate(
        applied_selected_positives,
        applied_selected_users,
    )
    applied_all_matching_rate = _optional_rate(
        applied_matching_positives,
        applied_matching_users,
    )
    return {
        "selection_ratio": ratio,
        "result_count": len(outcomes),
        "scenario_count": len(grouped),
        "policy_applied_result_count": len(applied),
        "sample_stable_result_rate": _optional_rate(
            sum(item.sample_stable for item in outcomes),
            len(outcomes),
        ),
        "selected_user_count": selected_users,
        "matching_user_count": matching_users,
        "pooled_actual_goal_rate": pooled_actual_rate,
        "pooled_all_matching_goal_rate": pooled_all_matching_rate,
        "lift_vs_all_matching_percentage_points": _rate_difference_points(
            pooled_actual_rate,
            pooled_all_matching_rate,
        ),
        "positive_capture_rate": _optional_rate(
            selected_positives,
            matching_positives,
        ),
        "reach_within_matching": _optional_rate(selected_users, matching_users),
        "applied_only": {
            "selected_user_count": applied_selected_users,
            "matching_user_count": applied_matching_users,
            "actual_goal_rate": applied_actual_rate,
            "all_matching_goal_rate": applied_all_matching_rate,
            "lift_vs_all_matching_percentage_points": _rate_difference_points(
                applied_actual_rate,
                applied_all_matching_rate,
            ),
            "positive_capture_rate": _optional_rate(
                applied_selected_positives,
                applied_matching_positives,
            ),
            "reach_within_matching": _optional_rate(
                applied_selected_users,
                applied_matching_users,
            ),
        },
        "rank_quality": rank_quality,
    }


def _choose_development_ratio(
    summaries: Sequence[Mapping[str, Any]],
    *,
    config: AudienceSelectionEvaluationConfig,
) -> Mapping[str, Any]:
    eligible: list[Mapping[str, Any]] = []
    for summary in summaries:
        ratio = float(summary["selection_ratio"])
        if ratio >= 1.0:
            continue
        applied = _mapping(summary.get("applied_only"))
        lift = _optional_float(
            applied.get("lift_vs_all_matching_percentage_points")
        )
        capture = _optional_float(applied.get("positive_capture_rate"))
        actual_rate = _optional_float(applied.get("actual_goal_rate"))
        if (
            int(summary.get("policy_applied_result_count", 0) or 0)
            < config.minimum_policy_applied_result_count
            or lift is None
            or lift <= 0.0
            or capture is None
            or capture < config.minimum_positive_capture_rate
            or actual_rate is None
        ):
            continue
        eligible.append(summary)
    if not eligible:
        return _summary_for_ratio(summaries, 1.0)
    return max(
        eligible,
        key=lambda item: (
            float(_mapping(item.get("applied_only")).get("actual_goal_rate", 0.0)),
            float(item["selection_ratio"]),
        ),
    )


def _validation_criteria(
    *,
    selected: Mapping[str, Any],
    all_matching: Mapping[str, Any],
    config: AudienceSelectionEvaluationConfig,
) -> dict[str, bool]:
    ratio = float(selected["selection_ratio"])
    if ratio >= 1.0:
        return {
            "development_selected_reduction": False,
            "minimum_applied_results": False,
            "nonnegative_lift_vs_all_matching": False,
            "minimum_positive_capture": False,
        }
    applied = _mapping(selected.get("applied_only"))
    lift = _optional_float(applied.get("lift_vs_all_matching_percentage_points"))
    capture = _optional_float(applied.get("positive_capture_rate"))
    selected_rank_quality = _mapping(selected.get("rank_quality"))
    all_rank_quality = _mapping(all_matching.get("rank_quality"))
    selected_pairwise = _optional_float(
        selected_rank_quality.get("pairwise_rank_accuracy")
    )
    all_pairwise = _optional_float(all_rank_quality.get("pairwise_rank_accuracy"))
    return {
        "development_selected_reduction": True,
        "minimum_applied_results": int(
            selected.get("policy_applied_result_count", 0) or 0
        )
        >= config.minimum_policy_applied_result_count,
        "nonnegative_lift_vs_all_matching": lift is not None and lift >= 0.0,
        "minimum_positive_capture": capture is not None
        and capture >= config.minimum_positive_capture_rate,
        "rank_accuracy_not_materially_worse": (
            selected_pairwise is None
            or all_pairwise is None
            or selected_pairwise + 0.05 >= all_pairwise
        ),
    }


def _evaluation_input_hash(
    *,
    development_outcomes: Sequence[AudienceSelectionOutcome],
    validation_outcomes: Sequence[AudienceSelectionOutcome],
    config: AudienceSelectionEvaluationConfig,
    development_split: str,
    validation_split: str,
) -> str:
    payload = {
        "evaluation_version": AUDIENCE_SELECTION_EVALUATION_VERSION,
        "config": asdict(config),
        "development_split": development_split,
        "validation_split": validation_split,
        "development_outcomes": [
            item.to_dict()
            for item in sorted(
                development_outcomes,
                key=_outcome_sort_key,
            )
        ],
        "validation_outcomes": [
            item.to_dict()
            for item in sorted(validation_outcomes, key=_outcome_sort_key)
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _selection_markdown_report(
    *,
    evaluation: AudienceSelectionPolicyEvaluation,
    finalized_artifact: Mapping[str, Any],
) -> str:
    selection = _mapping(finalized_artifact.get("selection"))
    lines = [
        "# 추천 대상 선택 비율 개발·검증 보고서",
        "",
        "## 판정",
        "",
        f"- 개발 구간 선택 비율: {float(selection.get('development_chosen_ratio', 1.0)) * 100:g}%",
        f"- 런타임 적용 비율: {float(selection.get('runtime_selected_ratio', 1.0)) * 100:g}%",
        f"- validation 통과: {'예' if selection.get('validation_passed') else '아니오'}",
        f"- 보정 상태: {finalized_artifact.get('calibration_status')}",
        "",
        "## 비율별 결과",
        "",
        "| 구간 | 비율 | 실제 성과율 | 전체 일치자 대비 lift | positive 포착률 | reach | Rank pairwise 정확도 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_label, summaries in (
        ("development", evaluation.development_summaries),
        ("validation", evaluation.validation_summaries),
    ):
        for summary in summaries:
            rank_quality = _mapping(summary.get("rank_quality"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        split_label,
                        _percent(summary.get("selection_ratio")),
                        _percent(summary.get("pooled_actual_goal_rate")),
                        _points(
                            summary.get(
                                "lift_vs_all_matching_percentage_points"
                            )
                        ),
                        _percent(summary.get("positive_capture_rate")),
                        _percent(summary.get("reach_within_matching")),
                        _percent(rank_quality.get("pairwise_rank_accuracy")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## 해석 주의",
            "",
            "- 비율 선택에는 development outcome만 사용하고 validation은 통과 여부 확인에 사용합니다.",
            "- 런타임 사용자 정렬에는 미래 outcome이 포함되지 않습니다.",
            "- 봉인된 final test는 이 과정에서 실행하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_for_ratio(
    summaries: Sequence[Mapping[str, Any]],
    ratio: float,
) -> Mapping[str, Any]:
    for summary in summaries:
        if float(summary["selection_ratio"]) == ratio:
            return summary
    raise ValueError(f"selection ratio summary not found: {ratio}")


def _outcome_sort_key(item: AudienceSelectionOutcome) -> tuple[Any, ...]:
    return (
        item.cutoff,
        item.scenario_id,
        item.selection_ratio,
        item.rank,
        item.candidate_type,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _rate_difference_points(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return (left - right) * 100.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
