from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.uplift.contracts import UpliftTrainingExample
from app.uplift.model import TransformedOutcomeRidgeModel, validate_model_payload
from app.uplift.split import (
    UpliftDatasetSplitUnavailable,
    split_by_experiment_end_time,
)
from app.uplift.validation import evaluate_validation_policy


REFERENCE_TIME = datetime(2026, 7, 22, tzinfo=UTC)


def test_experiment_time_split_keeps_each_randomized_experiment_together() -> None:
    examples = [
        example("early", "t", 1, days_ago=4),
        example("early", "c", 0, days_ago=4),
        example("middle", "t", 1, days_ago=2),
        example("middle", "c", 0, days_ago=2),
        example("latest", "t", 1, days_ago=1),
        example("latest", "c", 0, days_ago=1),
    ]

    split = split_by_experiment_end_time(examples)

    assert split.train_experiment_ids == ("early", "middle")
    assert split.test_experiment_ids == ("latest",)
    assert {item.ad_experiment_id for item in split.train_examples}.isdisjoint(
        {item.ad_experiment_id for item in split.test_examples}
    )
    assert {item.treatment for item in split.train_examples} == {0, 1}
    assert {item.treatment for item in split.test_examples} == {0, 1}


def test_experiment_time_split_rejects_one_experiment_or_missing_arm() -> None:
    with pytest.raises(UpliftDatasetSplitUnavailable):
        split_by_experiment_end_time(
            [
                example("only", "t", 1, days_ago=1),
                example("only", "c", 0, days_ago=1),
            ]
        )
    with pytest.raises(UpliftDatasetSplitUnavailable, match="incomplete"):
        split_by_experiment_end_time(
            [
                example("early", "t", 1, days_ago=2),
                example("latest", "t", 1, days_ago=1),
                example("latest", "c", 0, days_ago=1),
            ]
        )


def test_model_payload_requires_matching_finite_dimensions() -> None:
    payload = TransformedOutcomeRidgeModel(
        feature_means=(0.0, 1.0),
        feature_scales=(1.0, 2.0),
        coefficients=(0.1, -0.2),
        intercept=0.3,
    ).to_payload()

    restored = TransformedOutcomeRidgeModel.from_payload(payload)

    assert restored.to_payload() == payload
    with pytest.raises(ValueError, match="dimensions"):
        validate_model_payload({**payload, "coefficients": [0.1]})
    with pytest.raises(ValueError, match="finite"):
        validate_model_payload({**payload, "intercept": float("nan")})
    with pytest.raises(ValueError, match="supported"):
        validate_model_payload({**payload, "model_version": "unknown"})


def test_validation_policy_evaluates_every_provisional_guard() -> None:
    policy = {
        "validation_policy_version": "uplift-validation.test",
        "policy_status": "provisional_safety_guard",
        "minimum_completed_experiments": 2,
        "minimum_treatment_observations": 4,
        "minimum_control_observations": 4,
        "minimum_positive_outcomes_per_arm": 1,
        "required_metrics": {
            "qini_above_zero": True,
            "auuc_above_baseline": True,
        },
        "requires_manual_approval": True,
        "statistical_power_derived": False,
    }

    passed = evaluate_validation_policy(
        metrics={"qini": 0.2, "auuc": 0.3, "auuc_baseline": 0.1},
        completed_experiment_count=2,
        treatment_observation_count=4,
        control_observation_count=4,
        positive_treatment_outcome_count=1,
        positive_control_outcome_count=1,
        policy=policy,
    )
    failed = evaluate_validation_policy(
        metrics={"qini": 0.0, "auuc": 0.1, "auuc_baseline": 0.1},
        completed_experiment_count=1,
        treatment_observation_count=3,
        control_observation_count=3,
        positive_treatment_outcome_count=0,
        positive_control_outcome_count=0,
        policy=policy,
    )

    assert passed == {
        "passed": True,
        "failed_rules": [],
        "policy_version": "uplift-validation.test",
        "policy_status": "provisional_safety_guard",
        "statistical_power_derived": False,
        "requires_manual_approval": True,
    }
    assert failed["passed"] is False
    assert set(failed["failed_rules"]) == {
        "minimum_completed_experiments",
        "minimum_treatment_observations",
        "minimum_control_observations",
        "minimum_positive_treatment_outcomes",
        "minimum_positive_control_outcomes",
        "qini_above_zero",
        "auuc_above_baseline",
    }


def example(
    experiment_id: str,
    suffix: str,
    treatment: int,
    *,
    days_ago: int,
) -> UpliftTrainingExample:
    outcome_end = REFERENCE_TIME - timedelta(days=days_ago)
    return UpliftTrainingExample(
        experiment_unit_id=f"{experiment_id}_{suffix}",
        project_id="project",
        promotion_run_id=f"run_{experiment_id}",
        ad_experiment_id=experiment_id,
        segment_id="segment",
        user_id=f"user_{experiment_id}_{suffix}",
        audience_snapshot_id=f"snapshot_{experiment_id}",
        vector_generation_id="generation",
        features=(float(treatment), 1.0),
        treatment=treatment,
        outcome=treatment,
        treatment_probability=0.5,
        assigned_at=outcome_end - timedelta(days=30),
        outcome_window_start=outcome_end - timedelta(days=30),
        outcome_window_end=outcome_end,
        vector_version="hotel_behavior.v2",
        feature_contract_hash="a" * 64,
        outcome_spec_hash="b" * 64,
        outcome_contract_hash="c" * 64,
    )
