from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.audience_selection import (
    AUDIENCE_SELECTION_ARTIFACT_SCHEMA_VERSION,
    AudienceSelectionPolicyError,
    build_audience_selection_policy,
    finalize_audience_selection_artifact,
    fixed_ratio_audience_selection_policy,
    load_audience_selection_policy,
)


def test_fixed_ratio_policy_selects_ceil_of_matching_users() -> None:
    policy = fixed_ratio_audience_selection_policy(
        goal_metric="booking_conversion_rate",
        selected_ratio=0.6,
        minimum_selected_user_count=10,
    )

    decision = policy.decide(
        goal_metric="booking_conversion_rate",
        candidate_type="promotion_responsive",
        matching_user_count=101,
    )

    assert decision.selected_user_count == 61
    assert decision.selection_limited is True
    assert decision.method == "top_behavior_strength_ratio"
    assert decision.applied_ratio == pytest.approx(61 / 101)


def test_fixed_ratio_policy_keeps_all_users_when_selected_sample_is_too_small() -> None:
    policy = fixed_ratio_audience_selection_policy(
        goal_metric="booking_conversion_rate",
        selected_ratio=0.2,
        minimum_selected_user_count=30,
    )

    decision = policy.decide(
        goal_metric="booking_conversion_rate",
        candidate_type="intent_matched",
        matching_user_count=100,
    )

    assert decision.selected_user_count == 100
    assert decision.method == "all_matching"
    assert decision.fallback_reason == "minimum_selected_user_count_not_met"


def test_fixed_ratio_policy_keeps_all_users_for_unsupported_goal() -> None:
    policy = fixed_ratio_audience_selection_policy(
        goal_metric="booking_conversion_rate",
        selected_ratio=0.6,
        minimum_selected_user_count=10,
    )

    decision = policy.decide(
        goal_metric="inflow_rate",
        candidate_type="promotion_responsive",
        matching_user_count=100,
    )

    assert decision.selected_user_count == 100
    assert decision.fallback_reason == "unsupported_goal_or_candidate_type"


def test_policy_artifact_round_trip_validates_integrity(tmp_path: Path) -> None:
    artifact = finalize_audience_selection_artifact(
        {
            "schema_version": AUDIENCE_SELECTION_ARTIFACT_SCHEMA_VERSION,
            "policy_version": "expedia-selection-test-v1",
            "calibration_status": "validated",
            "rules": [
                {
                    "goal_metric": "booking_conversion_rate",
                    "selected_ratio": 0.8,
                    "minimum_selected_user_count": 30,
                    "candidate_types": [],
                }
            ],
            "provenance": {"development": "2013", "validation": "2014"},
        }
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    policy = load_audience_selection_policy(path)

    assert policy.policy_version == "expedia-selection-test-v1"
    assert policy.artifact_hash == artifact["artifact_hash"]


def test_invalid_artifact_falls_back_to_all_matching(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": AUDIENCE_SELECTION_ARTIFACT_SCHEMA_VERSION,
                "policy_version": "tampered",
                "calibration_status": "validated",
                "rules": [],
                "artifact_hash": "wrong",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AudienceSelectionPolicyError, match="hash mismatch"):
        load_audience_selection_policy(path)

    policy = build_audience_selection_policy(path)
    decision = policy.decide(
        goal_metric="booking_conversion_rate",
        candidate_type="intent_matched",
        matching_user_count=50,
    )
    assert decision.selected_user_count == 50
    assert decision.calibration_status == "invalid_artifact"
    assert decision.fallback_reason == "artifact_invalid"


def test_default_policy_uses_validated_booking_audience_ratio() -> None:
    policy = build_audience_selection_policy()

    decision = policy.decide(
        goal_metric="booking_conversion_rate",
        candidate_type="promotion_responsive",
        matching_user_count=230,
    )

    assert policy.calibration_status == "validated"
    assert decision.configured_ratio == 0.8
    assert decision.selected_user_count == 184
    assert decision.artifact_hash is not None
