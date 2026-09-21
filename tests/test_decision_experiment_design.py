from decimal import Decimal

import pytest

from app.decision.experiment_design import (
    ExperimentAudienceMember,
    ExperimentDesignValidationError,
    RandomizedHoldoutAudienceTooSmallError,
    RandomizedHoldoutConfigurationError,
    allocate_experiment_units,
    build_experiment_design,
)


OUTCOME_HASH = "a" * 64


def test_all_treatment_does_not_require_randomization_salt() -> None:
    design = build_experiment_design(
        mode="all_treatment",
        treatment_ratio=None,
        outcome_window_days=30,
        outcome_spec_hash=OUTCOME_HASH,
        randomization_salt=None,
    )

    allocations, results = allocate_experiment_units(
        project_id="project",
        promotion_run_id="run",
        design=design,
        members=members(3),
        randomization_salt=None,
    )

    assert {allocation.arm for allocation in allocations} == {"treatment"}
    assert results[0].treatment_count == 3
    assert results[0].control_count == 0
    assert results[0].actual_treatment_ratio == Decimal("1")


def test_randomized_holdout_requires_salt() -> None:
    with pytest.raises(RandomizedHoldoutConfigurationError):
        build_experiment_design(
            mode="randomized_holdout",
            treatment_ratio=0.5,
            outcome_window_days=30,
            outcome_spec_hash=OUTCOME_HASH,
            randomization_salt=None,
        )


@pytest.mark.parametrize("ratio", [None, 0, 1, -0.1, 1.1])
def test_randomized_holdout_rejects_invalid_ratio(ratio: float | None) -> None:
    with pytest.raises(ExperimentDesignValidationError):
        build_experiment_design(
            mode="randomized_holdout",
            treatment_ratio=ratio,
            outcome_window_days=30,
            outcome_spec_hash=OUTCOME_HASH,
            randomization_salt="secret",
        )


def test_complete_randomization_is_exact_and_input_order_independent() -> None:
    design = build_experiment_design(
        mode="randomized_holdout",
        treatment_ratio=0.5,
        outcome_window_days=30,
        outcome_spec_hash=OUTCOME_HASH,
        randomization_salt="secret",
    )
    original = members(5)

    first, first_results = allocate_experiment_units(
        project_id="project",
        promotion_run_id="run",
        design=design,
        members=original,
        randomization_salt="secret",
    )
    second, _ = allocate_experiment_units(
        project_id="project",
        promotion_run_id="run",
        design=design,
        members=list(reversed(original)),
        randomization_salt="secret",
    )

    assert [(item.member.user_id, item.arm) for item in first] == [
        (item.member.user_id, item.arm) for item in second
    ]
    assert first_results[0].treatment_count == 3
    assert first_results[0].control_count == 2
    assert all(item.treatment_probability == Decimal("0.6") for item in first)


@pytest.mark.parametrize("ratio", [0.01, 0.99])
def test_complete_randomization_keeps_both_arms_non_empty(ratio: float) -> None:
    design = build_experiment_design(
        mode="randomized_holdout",
        treatment_ratio=ratio,
        outcome_window_days=30,
        outcome_spec_hash=OUTCOME_HASH,
        randomization_salt="secret",
    )

    allocations, results = allocate_experiment_units(
        project_id="project",
        promotion_run_id="run",
        design=design,
        members=members(2),
        randomization_salt="secret",
    )

    assert {item.arm for item in allocations} == {"treatment", "control"}
    assert results[0].treatment_count == 1
    assert results[0].control_count == 1


def test_randomized_holdout_rejects_single_user_experiment() -> None:
    design = build_experiment_design(
        mode="randomized_holdout",
        treatment_ratio=0.5,
        outcome_window_days=30,
        outcome_spec_hash=OUTCOME_HASH,
        randomization_salt="secret",
    )

    with pytest.raises(RandomizedHoldoutAudienceTooSmallError):
        allocate_experiment_units(
            project_id="project",
            promotion_run_id="run",
            design=design,
            members=members(1),
            randomization_salt="secret",
        )


def test_randomized_holdout_rejects_empty_experiment() -> None:
    design = build_experiment_design(
        mode="randomized_holdout",
        treatment_ratio=0.5,
        outcome_window_days=30,
        outcome_spec_hash=OUTCOME_HASH,
        randomization_salt="secret",
    )

    with pytest.raises(RandomizedHoldoutAudienceTooSmallError):
        allocate_experiment_units(
            project_id="project",
            promotion_run_id="run",
            design=design,
            members=[],
            randomization_salt="secret",
            experiment_bindings={"exp": ("segment", "snapshot")},
        )


def test_same_user_can_receive_different_arm_in_another_experiment() -> None:
    design = build_experiment_design(
        mode="randomized_holdout",
        treatment_ratio=0.5,
        outcome_window_days=30,
        outcome_spec_hash=OUTCOME_HASH,
        randomization_salt="secret",
    )
    experiment_a = members(20, experiment_id="exp_a")
    experiment_b = members(20, experiment_id="exp_b")

    allocation_a, _ = allocate_experiment_units(
        project_id="project",
        promotion_run_id="run",
        design=design,
        members=experiment_a,
        randomization_salt="secret",
    )
    allocation_b, _ = allocate_experiment_units(
        project_id="project",
        promotion_run_id="run",
        design=design,
        members=experiment_b,
        randomization_salt="secret",
    )

    arms_a = {item.member.user_id: item.arm for item in allocation_a}
    arms_b = {item.member.user_id: item.arm for item in allocation_b}
    assert any(arms_a[user_id] != arms_b[user_id] for user_id in arms_a)


def members(
    count: int,
    *,
    experiment_id: str = "exp",
) -> list[ExperimentAudienceMember]:
    return [
        ExperimentAudienceMember(
            user_id=f"user_{index:03d}",
            segment_id=f"segment_{experiment_id}",
            ad_experiment_id=experiment_id,
            audience_snapshot_id=f"snapshot_{experiment_id}",
            vector_generation_id="generation",
            behavior_fit_score=Decimal("0.8"),
        )
        for index in range(count)
    ]
