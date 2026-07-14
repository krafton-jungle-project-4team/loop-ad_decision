from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from offline_evaluation.rank_quality import (
    CRITERION_EVIDENCE,
    CRITERION_QUALITY,
    criterion_result,
    determine_final_verdict,
)


EXTERNAL_EVALUATION_CONTRACT_VERSION = "external.evaluation-contract.v1"

CLAIM_CANDIDATE_OUTCOME_UTILITY = "candidate_outcome_utility"
CLAIM_DESTINATION_INTEREST_GENERALIZATION = (
    "destination_interest_generalization"
)
CLAIM_REPRESENTATIVE_AUDIENCE_ENRICHMENT = (
    "representative_audience_enrichment"
)
CLAIM_CROSS_DOMAIN_CANDIDATE_ROBUSTNESS = (
    "cross_domain_candidate_robustness"
)
CLAIM_STRATEGY_PORTFOLIO_DIVERSITY = "strategy_portfolio_diversity"
CLAIM_EXPEDIA_PREDICTION_CALIBRATION = "expedia_prediction_calibration"
CLAIM_CAUSAL_INCREMENTAL_LIFT = "causal_incremental_lift"
CLAIM_HOSPITALITY_DOMAIN_PERFORMANCE = "hospitality_domain_performance"

SCOPE_HOSPITALITY_SUPPORTING_EVIDENCE = "hospitality_supporting_evidence"
SCOPE_CROSS_DOMAIN_DIAGNOSTIC_ONLY = "cross_domain_diagnostic_only"


@dataclass(frozen=True, slots=True)
class ExternalFinalTestCriteria:
    portfolio_candidate_beats_baseline_rate_min: float = 0.50
    portfolio_scenario_any_candidate_beats_baseline_rate_min: float = 0.50
    portfolio_scenario_all_candidates_beat_baseline_rate_min: float = 0.50
    portfolio_mean_candidate_lift_percentage_points_min: float = 0.0
    portfolio_mean_worst_candidate_lift_percentage_points_min: float = 0.0
    scenario_with_observed_outcome_count_min: int = 3
    portfolio_candidate_result_count_min: int = 3
    portfolio_multi_candidate_scenario_count_min: int = 3
    portfolio_three_candidate_scenario_count_min: int = 3
    candidate_type_count_min: int = 2
    mean_portfolio_candidate_overlap_max: float = 0.90
    maximum_portfolio_candidate_overlap_max: float = 0.95

    def __post_init__(self) -> None:
        rates = (
            self.portfolio_candidate_beats_baseline_rate_min,
            self.portfolio_scenario_any_candidate_beats_baseline_rate_min,
            self.portfolio_scenario_all_candidates_beat_baseline_rate_min,
            self.mean_portfolio_candidate_overlap_max,
            self.maximum_portfolio_candidate_overlap_max,
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("external final rate criteria must be between 0 and 1")
        counts = (
            self.scenario_with_observed_outcome_count_min,
            self.portfolio_candidate_result_count_min,
            self.portfolio_multi_candidate_scenario_count_min,
            self.portfolio_three_candidate_scenario_count_min,
            self.candidate_type_count_min,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("external final count criteria must be positive")


@dataclass(frozen=True, slots=True)
class ExternalEvaluationContract:
    dataset_id: str
    verdict_scope: str
    supported_claim_ids: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]
    acceptance_criteria: Mapping[str, Mapping[str, Any]]
    version: str = EXTERNAL_EVALUATION_CONTRACT_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_id": self.dataset_id,
            "verdict_scope": self.verdict_scope,
            "supported_claim_ids": list(self.supported_claim_ids),
            "unsupported_claim_ids": list(self.unsupported_claim_ids),
            "criteria": {
                key: dict(value)
                for key, value in self.acceptance_criteria.items()
            },
        }


def external_evaluation_contract(
    dataset_id: str,
    *,
    criteria: ExternalFinalTestCriteria | None = None,
) -> ExternalEvaluationContract:
    thresholds = criteria or ExternalFinalTestCriteria()
    dataset_contract = _dataset_claim_contract(dataset_id)
    applicable_criteria = _applicable_criteria(dataset_id)
    criterion_specs = _criterion_specs(dataset_id, thresholds)
    acceptance_criteria: dict[str, Mapping[str, Any]] = {}
    for criterion_id, spec in criterion_specs.items():
        claim_id = str(spec["claim_id"])
        if criterion_id in applicable_criteria:
            acceptance_criteria[criterion_id] = {
                **spec,
                "applicable": True,
            }
            continue
        acceptance_criteria[criterion_id] = {
            "applicable": False,
            "category": spec["category"],
            "claim_id": claim_id,
            "operator": None,
            "threshold": None,
            "reason": _not_applicable_reason(dataset_id, claim_id),
        }
    return ExternalEvaluationContract(
        dataset_id=dataset_id,
        verdict_scope=dataset_contract["verdict_scope"],
        supported_claim_ids=dataset_contract["supported_claim_ids"],
        unsupported_claim_ids=dataset_contract["unsupported_claim_ids"],
        acceptance_criteria=acceptance_criteria,
    )


def evaluate_external_criteria(
    metrics: Mapping[str, Any],
    acceptance_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if acceptance_contract.get("version") != EXTERNAL_EVALUATION_CONTRACT_VERSION:
        raise ValueError("unsupported external evaluation contract version")
    raw_criteria = acceptance_contract.get("criteria")
    if not isinstance(raw_criteria, Mapping):
        raise ValueError("external evaluation contract criteria must be an object")
    results: dict[str, Any] = {}
    for criterion_id, raw_spec in raw_criteria.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError("external evaluation criterion must be an object")
        category = str(raw_spec.get("category", ""))
        claim_id = str(raw_spec.get("claim_id", ""))
        if raw_spec.get("applicable") is not True:
            results[str(criterion_id)] = {
                "actual": None,
                "operator": "not_applicable",
                "threshold": None,
                "category": category,
                "claim_id": claim_id,
                "applicable": False,
                "passed": None,
                "reason": str(raw_spec.get("reason", "unsupported claim")),
            }
            continue
        operator = str(raw_spec.get("operator", ""))
        threshold = raw_spec.get("threshold")
        if not isinstance(threshold, (int, float)):
            raise ValueError("applicable external criterion requires a threshold")
        actual = metrics.get(str(criterion_id))
        if actual is not None and not isinstance(actual, (int, float)):
            raise ValueError("external evaluation metric must be numeric or null")
        result = criterion_result(
            actual,
            operator,
            threshold,
            category=category,
        )
        results[str(criterion_id)] = {
            **result,
            "claim_id": claim_id,
            "applicable": True,
        }
    return results


def validate_external_evaluation_contract(
    acceptance_contract: Mapping[str, Any],
    *,
    dataset_id: str,
) -> None:
    expected = external_evaluation_contract(dataset_id).to_json()
    for key in (
        "version",
        "dataset_id",
        "verdict_scope",
        "supported_claim_ids",
        "unsupported_claim_ids",
    ):
        if acceptance_contract.get(key) != expected[key]:
            raise ValueError(
                f"external evaluation contract {key} changed after sealing"
            )

    raw_criteria = acceptance_contract.get("criteria")
    expected_criteria = expected["criteria"]
    if not isinstance(raw_criteria, Mapping):
        raise ValueError("external evaluation contract criteria must be an object")
    if set(raw_criteria) != set(expected_criteria):
        raise ValueError(
            "external evaluation contract criteria changed after sealing"
        )

    supported_claim_ids = set(expected["supported_claim_ids"])
    for criterion_id, raw_spec in raw_criteria.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError("external evaluation criterion must be an object")
        expected_spec = expected_criteria[criterion_id]
        for key in ("applicable", "category", "claim_id"):
            if raw_spec.get(key) != expected_spec.get(key):
                raise ValueError(
                    f"external evaluation criterion {criterion_id} {key} "
                    "changed after sealing"
                )
        if raw_spec.get("applicable") is not True:
            if raw_spec.get("operator") is not None or (
                raw_spec.get("threshold") is not None
            ):
                raise ValueError(
                    f"external evaluation criterion {criterion_id} must remain "
                    "not applicable"
                )
            continue
        if raw_spec.get("claim_id") not in supported_claim_ids:
            raise ValueError(
                f"external evaluation criterion {criterion_id} references an "
                "unsupported claim"
            )
        if raw_spec.get("operator") != expected_spec.get("operator"):
            raise ValueError(
                f"external evaluation criterion {criterion_id} operator "
                "changed after sealing"
            )
        if not isinstance(raw_spec.get("threshold"), (int, float)):
            raise ValueError(
                f"external evaluation criterion {criterion_id} requires a "
                "numeric threshold"
            )


def determine_external_evaluation_verdict(
    criteria_results: Mapping[str, Mapping[str, Any]],
) -> str:
    applicable_results = {
        key: value
        for key, value in criteria_results.items()
        if value.get("applicable") is True
    }
    return determine_final_verdict(applicable_results)


def _dataset_claim_contract(dataset_id: str) -> dict[str, Any]:
    if dataset_id == "booking-com":
        return {
            "verdict_scope": SCOPE_HOSPITALITY_SUPPORTING_EVIDENCE,
            "supported_claim_ids": (
                CLAIM_CANDIDATE_OUTCOME_UTILITY,
                CLAIM_DESTINATION_INTEREST_GENERALIZATION,
            ),
            "unsupported_claim_ids": (
                CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
                CLAIM_EXPEDIA_PREDICTION_CALIBRATION,
                CLAIM_CAUSAL_INCREMENTAL_LIFT,
            ),
        }
    if dataset_id == "airbnb":
        return {
            "verdict_scope": SCOPE_HOSPITALITY_SUPPORTING_EVIDENCE,
            "supported_claim_ids": (
                CLAIM_CANDIDATE_OUTCOME_UTILITY,
                CLAIM_REPRESENTATIVE_AUDIENCE_ENRICHMENT,
            ),
            "unsupported_claim_ids": (
                CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
                CLAIM_DESTINATION_INTEREST_GENERALIZATION,
                CLAIM_EXPEDIA_PREDICTION_CALIBRATION,
                CLAIM_CAUSAL_INCREMENTAL_LIFT,
            ),
        }
    if dataset_id == "synerise":
        return {
            "verdict_scope": SCOPE_CROSS_DOMAIN_DIAGNOSTIC_ONLY,
            "supported_claim_ids": (
                CLAIM_CANDIDATE_OUTCOME_UTILITY,
                CLAIM_CROSS_DOMAIN_CANDIDATE_ROBUSTNESS,
            ),
            "unsupported_claim_ids": (
                CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
                CLAIM_HOSPITALITY_DOMAIN_PERFORMANCE,
                CLAIM_EXPEDIA_PREDICTION_CALIBRATION,
                CLAIM_CAUSAL_INCREMENTAL_LIFT,
            ),
        }
    raise ValueError(f"unsupported external dataset: {dataset_id}")


def _applicable_criteria(dataset_id: str) -> frozenset[str]:
    common = {
        "scenario_with_observed_outcome_count",
        "portfolio_candidate_result_count",
        "portfolio_candidate_beats_baseline_rate",
        "portfolio_scenario_any_candidate_beats_baseline_rate",
        "portfolio_mean_candidate_lift_percentage_points",
    }
    if dataset_id == "booking-com":
        return frozenset(common)
    if dataset_id == "airbnb":
        return frozenset(common)
    if dataset_id == "synerise":
        return frozenset(common)
    raise ValueError(f"unsupported external dataset: {dataset_id}")


def _criterion_specs(
    dataset_id: str,
    criteria: ExternalFinalTestCriteria,
) -> dict[str, dict[str, Any]]:
    values = asdict(criteria)
    primary_claim_id = _primary_outcome_claim_id(dataset_id)
    scenario_threshold = (
        1
        if dataset_id == "airbnb"
        else values["scenario_with_observed_outcome_count_min"]
    )
    candidate_threshold = (
        1
        if dataset_id == "airbnb"
        else values["portfolio_candidate_result_count_min"]
    )
    return {
        "scenario_with_observed_outcome_count": _criterion_spec(
            ">=",
            scenario_threshold,
            values,
            "scenario_with_observed_outcome_count_min",
            category=CRITERION_EVIDENCE,
            claim_id=primary_claim_id,
        ),
        "portfolio_candidate_result_count": _criterion_spec(
            ">=",
            candidate_threshold,
            values,
            "portfolio_candidate_result_count_min",
            category=CRITERION_EVIDENCE,
            claim_id=primary_claim_id,
        ),
        "portfolio_multi_candidate_scenario_count": _criterion_spec(
            ">=",
            None,
            values,
            "portfolio_multi_candidate_scenario_count_min",
            category=CRITERION_EVIDENCE,
            claim_id=CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
        ),
        "portfolio_three_candidate_scenario_count": _criterion_spec(
            ">=",
            None,
            values,
            "portfolio_three_candidate_scenario_count_min",
            category=CRITERION_EVIDENCE,
            claim_id=CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
        ),
        "portfolio_candidate_beats_baseline_rate": _criterion_spec(
            ">=",
            None,
            values,
            "portfolio_candidate_beats_baseline_rate_min",
            category=CRITERION_QUALITY,
            claim_id=primary_claim_id,
        ),
        "portfolio_scenario_any_candidate_beats_baseline_rate": _criterion_spec(
            ">=",
            None,
            values,
            "portfolio_scenario_any_candidate_beats_baseline_rate_min",
            category=CRITERION_QUALITY,
            claim_id=primary_claim_id,
        ),
        "portfolio_scenario_all_candidates_beat_baseline_rate": _criterion_spec(
            ">=",
            None,
            values,
            "portfolio_scenario_all_candidates_beat_baseline_rate_min",
            category=CRITERION_QUALITY,
            claim_id=CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
        ),
        "portfolio_mean_candidate_lift_percentage_points": _criterion_spec(
            ">=",
            None,
            values,
            "portfolio_mean_candidate_lift_percentage_points_min",
            category=CRITERION_QUALITY,
            claim_id=primary_claim_id,
        ),
        "portfolio_mean_worst_candidate_lift_percentage_points": _criterion_spec(
            ">=",
            None,
            values,
            "portfolio_mean_worst_candidate_lift_percentage_points_min",
            category=CRITERION_QUALITY,
            claim_id=CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
        ),
        "candidate_type_count": _criterion_spec(
            ">=",
            None,
            values,
            "candidate_type_count_min",
            category=CRITERION_QUALITY,
            claim_id=CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
        ),
        "mean_portfolio_candidate_overlap": _criterion_spec(
            "<=",
            None,
            values,
            "mean_portfolio_candidate_overlap_max",
            category=CRITERION_QUALITY,
            claim_id=CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
        ),
        "maximum_portfolio_candidate_overlap": _criterion_spec(
            "<=",
            None,
            values,
            "maximum_portfolio_candidate_overlap_max",
            category=CRITERION_QUALITY,
            claim_id=CLAIM_STRATEGY_PORTFOLIO_DIVERSITY,
        ),
    }


def _criterion_spec(
    operator: str,
    dataset_threshold: Any,
    values: Mapping[str, Any],
    threshold_key: str,
    *,
    category: str,
    claim_id: str,
) -> dict[str, Any]:
    return {
        "operator": operator,
        "threshold": (
            dataset_threshold
            if dataset_threshold is not None
            else values[threshold_key]
        ),
        "category": category,
        "claim_id": claim_id,
    }


def _not_applicable_reason(dataset_id: str, claim_id: str) -> str:
    if claim_id == CLAIM_STRATEGY_PORTFOLIO_DIVERSITY:
        return (
            f"{dataset_id} adapter does not provide enough independent signals "
            "to pre-register strategy portfolio diversity as a supported claim"
        )
    return f"{claim_id} is not supported by the {dataset_id} dataset contract"


def _primary_outcome_claim_id(dataset_id: str) -> str:
    if dataset_id == "booking-com":
        return CLAIM_DESTINATION_INTEREST_GENERALIZATION
    if dataset_id == "airbnb":
        return CLAIM_REPRESENTATIVE_AUDIENCE_ENRICHMENT
    if dataset_id == "synerise":
        return CLAIM_CROSS_DOMAIN_CANDIDATE_ROBUSTNESS
    raise ValueError(f"unsupported external dataset: {dataset_id}")
