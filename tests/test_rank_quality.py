from __future__ import annotations

from dataclasses import asdict

import pytest

from offline_evaluation.external_final_test import (
    ExternalFinalTestCriteria,
    _evaluate_criteria,
)
from offline_evaluation.rank_quality import (
    CRITERION_EVIDENCE,
    CRITERION_QUALITY,
    VERDICT_FAILED,
    VERDICT_INCONCLUSIVE,
    VERDICT_PASSED,
    RankedOutcome,
    criterion_result,
    determine_final_verdict,
    summarize_rank_quality,
)


def test_rank_quality_does_not_count_second_best_rank_one_as_best() -> None:
    metrics = summarize_rank_quality(
        [[_outcome(1, 0.14), _outcome(2, 0.18), _outcome(3, 0.11)]]
    )

    assert metrics["rank_one_is_best_rate"] == 0.0
    assert metrics["pairwise_rank_accuracy"] == pytest.approx(2 / 3)
    rank_gap = metrics[
        "mean_rank_one_minus_rank_two_outcome_rate_percentage_points"
    ]
    assert rank_gap == pytest.approx(-4.0)


def test_rank_quality_reports_best_tie_without_counting_it_as_win() -> None:
    metrics = summarize_rank_quality(
        [[_outcome(1, 0.20), _outcome(2, 0.20), _outcome(3, 0.10)]]
    )

    assert metrics["rank_one_is_best_rate"] == 0.0
    assert metrics["rank_one_tied_best_rate"] == 1.0
    assert metrics["pairwise_rank_accuracy"] == 1.0
    assert metrics["pairwise_rank_tie_rate"] == pytest.approx(1 / 3)
    assert metrics["pairwise_rank_comparison_count"] == 2


def test_rank_quality_treats_all_ties_as_insufficient_ordering_evidence() -> None:
    metrics = summarize_rank_quality(
        [[_outcome(1, 0.20), _outcome(2, 0.20), _outcome(3, 0.20)]]
    )

    assert metrics["rank_comparable_scenario_count"] == 0
    assert metrics["rank_one_is_best_rate"] is None
    assert metrics["pairwise_rank_accuracy"] is None
    assert metrics["pairwise_rank_tie_rate"] == 1.0


def test_external_final_criteria_pass_only_with_top_three_ordering_quality() -> None:
    criteria = ExternalFinalTestCriteria()
    results = _evaluate_criteria(_passing_external_metrics(), asdict(criteria))

    assert determine_final_verdict(results) == VERDICT_PASSED

    wrong_order_metrics = {
        **_passing_external_metrics(),
        "rank_one_is_best_rate": 0.0,
        "pairwise_rank_accuracy": 0.40,
    }
    wrong_order_results = _evaluate_criteria(
        wrong_order_metrics,
        asdict(criteria),
    )

    assert determine_final_verdict(wrong_order_results) == VERDICT_FAILED

    weak_rank_three_metrics = {
        **_passing_external_metrics(),
        "rank_three_beats_baseline_rate": 0.0,
        "mean_rank_three_lift_percentage_points": -2.0,
    }
    weak_rank_three_results = _evaluate_criteria(
        weak_rank_three_metrics,
        asdict(criteria),
    )

    assert determine_final_verdict(weak_rank_three_results) == VERDICT_FAILED


def test_external_final_criteria_are_inconclusive_without_rank_three() -> None:
    criteria = ExternalFinalTestCriteria()
    metrics = {
        **_passing_external_metrics(),
        "rank_three_result_count": 0,
        "three_rank_scenario_count": 0,
    }

    results = _evaluate_criteria(metrics, asdict(criteria))

    assert determine_final_verdict(results) == VERDICT_INCONCLUSIVE


def test_final_verdict_prioritizes_evidence_before_quality() -> None:
    insufficient = {
        "sample": criterion_result(
            1,
            ">=",
            3,
            category=CRITERION_EVIDENCE,
        ),
        "quality": criterion_result(
            0.0,
            ">=",
            0.5,
            category=CRITERION_QUALITY,
        ),
    }
    sufficient_but_bad = {
        **insufficient,
        "sample": criterion_result(
            3,
            ">=",
            3,
            category=CRITERION_EVIDENCE,
        ),
    }

    assert determine_final_verdict(insufficient) == VERDICT_INCONCLUSIVE
    assert determine_final_verdict(sufficient_but_bad) == VERDICT_FAILED


def _outcome(rank: int, actual_rate: float) -> RankedOutcome:
    return RankedOutcome(
        rank=rank,
        actual_rate=actual_rate,
        baseline_rate=0.10,
    )


def _passing_external_metrics() -> dict[str, int | float]:
    return {
        "scenario_with_observed_outcome_count": 3,
        "rank_comparable_scenario_count": 3,
        "rank_comparable_scenario_rate": 1.0,
        "rank_two_result_count": 3,
        "rank_three_result_count": 3,
        "three_rank_scenario_count": 3,
        "pairwise_rank_comparison_count": 9,
        "pairwise_rank_tie_rate": 0.0,
        "rank_one_beats_baseline_rate": 1.0,
        "mean_rank_one_lift_percentage_points": 5.0,
        "rank_one_is_best_rate": 1.0,
        "rank_two_beats_baseline_rate": 1.0,
        "mean_rank_two_lift_percentage_points": 3.0,
        "rank_three_beats_baseline_rate": 1.0,
        "mean_rank_three_lift_percentage_points": 1.0,
        "pairwise_rank_accuracy": 1.0,
        "candidate_type_count": 3,
        "mean_non_first_rank_overlap": 0.20,
        "maximum_non_first_rank_overlap": 0.30,
    }
