from __future__ import annotations

import pytest

from offline_evaluation.external_evaluation_contract import (
    CLAIM_HOSPITALITY_DOMAIN_PERFORMANCE,
    CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
    EXTERNAL_EVALUATION_CONTRACT_VERSION,
    SCOPE_CROSS_DOMAIN_DIAGNOSTIC_ONLY,
    determine_external_evaluation_verdict,
    evaluate_external_criteria,
    external_evaluation_contract,
    validate_external_evaluation_contract,
)
from offline_evaluation.rank_quality import (
    VERDICT_FAILED,
    VERDICT_INCONCLUSIVE,
    VERDICT_PASSED,
)


def test_dataset_contracts_only_apply_supported_claim_criteria() -> None:
    booking = external_evaluation_contract("booking-com").to_json()
    airbnb = external_evaluation_contract("airbnb").to_json()
    synerise = external_evaluation_contract("synerise").to_json()

    assert booking["version"] == EXTERNAL_EVALUATION_CONTRACT_VERSION
    assert CLAIM_STRATEGY_PORTFOLIO_DIVERSITY in booking[
        "unsupported_claim_ids"
    ]
    assert booking["criteria"][
        "portfolio_multi_candidate_scenario_count"
    ]["applicable"] is False
    assert booking["criteria"][
        "scenario_with_observed_outcome_count"
    ]["threshold"] == 3

    assert airbnb["criteria"][
        "scenario_with_observed_outcome_count"
    ]["threshold"] == 1
    assert airbnb["criteria"]["portfolio_candidate_result_count"][
        "threshold"
    ] == 1

    assert synerise["verdict_scope"] == SCOPE_CROSS_DOMAIN_DIAGNOSTIC_ONLY
    assert CLAIM_HOSPITALITY_DOMAIN_PERFORMANCE in synerise[
        "unsupported_claim_ids"
    ]


def test_not_applicable_criteria_are_reported_without_affecting_verdict() -> None:
    contract = external_evaluation_contract("booking-com").to_json()
    results = evaluate_external_criteria(_passing_booking_metrics(), contract)

    diversity = results["portfolio_multi_candidate_scenario_count"]
    assert diversity["applicable"] is False
    assert diversity["passed"] is None
    assert diversity["operator"] == "not_applicable"
    assert determine_external_evaluation_verdict(results) == VERDICT_PASSED


def test_applicable_evidence_and_quality_remain_strict() -> None:
    contract = external_evaluation_contract("booking-com").to_json()
    insufficient = evaluate_external_criteria(
        {
            **_passing_booking_metrics(),
            "scenario_with_observed_outcome_count": 2,
        },
        contract,
    )
    poor_quality = evaluate_external_criteria(
        {
            **_passing_booking_metrics(),
            "portfolio_candidate_beats_baseline_rate": 0.3,
        },
        contract,
    )

    assert determine_external_evaluation_verdict(insufficient) == VERDICT_INCONCLUSIVE
    assert determine_external_evaluation_verdict(poor_quality) == VERDICT_FAILED


def test_sealed_contract_rejects_changed_claim_support() -> None:
    contract = external_evaluation_contract("booking-com").to_json()
    contract["supported_claim_ids"] = ["strategy_portfolio_diversity"]

    with pytest.raises(
        ValueError,
        match="supported_claim_ids changed after sealing",
    ):
        validate_external_evaluation_contract(
            contract,
            dataset_id="booking-com",
        )


def test_sealed_contract_allows_preregistered_threshold_override() -> None:
    contract = external_evaluation_contract("booking-com").to_json()
    contract["criteria"]["portfolio_candidate_beats_baseline_rate"][
        "threshold"
    ] = 0.75

    validate_external_evaluation_contract(
        contract,
        dataset_id="booking-com",
    )


def _passing_booking_metrics() -> dict[str, int | float]:
    return {
        "scenario_with_observed_outcome_count": 3,
        "portfolio_candidate_result_count": 3,
        "portfolio_multi_candidate_scenario_count": 0,
        "portfolio_three_candidate_scenario_count": 0,
        "portfolio_candidate_beats_baseline_rate": 2 / 3,
        "portfolio_scenario_any_candidate_beats_baseline_rate": 2 / 3,
        "portfolio_scenario_all_candidates_beat_baseline_rate": 2 / 3,
        "portfolio_mean_candidate_lift_percentage_points": 1.0,
        "portfolio_mean_worst_candidate_lift_percentage_points": 1.0,
        "candidate_type_count": 1,
        "mean_portfolio_candidate_overlap": 0.0,
        "maximum_portfolio_candidate_overlap": 0.0,
    }
