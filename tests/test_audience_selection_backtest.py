from __future__ import annotations

from pathlib import Path

from app.analysis.audience_selection import load_audience_selection_policy
from offline_evaluation.audience_selection import (
    AudienceSelectionEvaluationConfig,
    AudienceSelectionOutcome,
    build_audience_selection_policy_evaluation,
    write_audience_selection_evaluation_artifacts,
)


def test_policy_evaluation_selects_development_ratio_confirmed_by_validation(
    tmp_path: Path,
) -> None:
    config = AudienceSelectionEvaluationConfig(
        minimum_selected_user_count=1,
        minimum_policy_applied_result_count=2,
        minimum_positive_capture_rate=0.8,
    )
    development = outcomes_by_ratio(
        selected_positives={0.2: 3, 0.4: 6, 0.6: 8, 0.8: 9, 1.0: 10}
    )
    validation = outcomes_by_ratio(
        selected_positives={0.2: 3, 0.4: 6, 0.6: 8, 0.8: 9, 1.0: 10}
    )

    evaluation = build_audience_selection_policy_evaluation(
        development_outcomes=development,
        validation_outcomes=validation,
        config=config,
        development_split="2013",
        validation_split="2014",
    )

    assert evaluation.artifact["calibration_status"] == "validated"
    assert evaluation.artifact["selection"]["development_chosen_ratio"] == 0.6
    assert evaluation.artifact["selection"]["runtime_selected_ratio"] == 0.6
    assert evaluation.artifact["selection"]["validation_passed"] is True

    artifacts = write_audience_selection_evaluation_artifacts(
        evaluation,
        development_outcomes=development,
        validation_outcomes=validation,
        output_dir=tmp_path,
    )
    policy = load_audience_selection_policy(artifacts["policy"])
    decision = policy.decide(
        goal_metric="booking_conversion_rate",
        candidate_type="intent_matched",
        matching_user_count=100,
    )
    assert decision.selected_user_count == 60
    assert artifacts["development_results"].exists()
    assert artifacts["validation_results"].exists()
    assert "후보 baseline 초과율" in artifacts["report"].read_text(
        encoding="utf-8"
    )


def test_policy_evaluation_falls_back_when_validation_does_not_confirm_choice() -> None:
    config = AudienceSelectionEvaluationConfig(
        minimum_selected_user_count=1,
        minimum_policy_applied_result_count=2,
        minimum_positive_capture_rate=0.8,
    )
    development = outcomes_by_ratio(
        selected_positives={0.2: 3, 0.4: 6, 0.6: 8, 0.8: 9, 1.0: 10}
    )
    validation = outcomes_by_ratio(
        selected_positives={0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1.0: 10}
    )

    evaluation = build_audience_selection_policy_evaluation(
        development_outcomes=development,
        validation_outcomes=validation,
        config=config,
        development_split="2013",
        validation_split="2014",
    )

    assert evaluation.artifact["calibration_status"] == "fallback_all_matching"
    assert evaluation.artifact["selection"]["development_chosen_ratio"] == 0.6
    assert evaluation.artifact["selection"]["runtime_selected_ratio"] == 1.0
    assert evaluation.artifact["selection"]["validation_passed"] is False


def outcomes_by_ratio(
    *,
    selected_positives: dict[float, int],
) -> list[AudienceSelectionOutcome]:
    outcomes: list[AudienceSelectionOutcome] = []
    for ratio in (0.2, 0.4, 0.6, 0.8, 1.0):
        for scenario_index in range(2):
            selected_user_count = int(100 * ratio)
            positive_count = selected_positives[ratio]
            actual_rate = positive_count / selected_user_count
            outcomes.append(
                AudienceSelectionOutcome(
                    cutoff=f"201{scenario_index + 3}-07-01T00:00:00+00:00",
                    scenario_id=f"scenario_{scenario_index}",
                    selection_ratio=ratio,
                    rank=1,
                    candidate_type="intent_matched",
                    matching_user_count=100,
                    selected_user_count=selected_user_count,
                    matching_positive_user_count=10,
                    selected_positive_user_count=positive_count,
                    baseline_user_count=200,
                    baseline_positive_user_count=10,
                    predicted_goal_rate=actual_rate,
                    actual_goal_rate=actual_rate,
                    all_matching_goal_rate=0.1,
                    baseline_goal_rate=0.05,
                    lift_vs_all_matching_percentage_points=(
                        actual_rate - 0.1
                    )
                    * 100,
                    lift_vs_baseline_percentage_points=(actual_rate - 0.05) * 100,
                    positive_capture_rate=positive_count / 10,
                    reach_within_matching=ratio,
                    policy_applied=ratio < 1.0,
                    sample_stable=True,
                )
            )
    return outcomes
