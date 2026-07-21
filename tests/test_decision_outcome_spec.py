from app.analysis.segment_audience_templates import RegisteredSegmentAudienceBinder
from app.decision.outcome_spec import (
    build_frozen_outcome_spec,
    outcome_spec_hash,
    require_frozen_outcome_spec,
)


def test_booking_outcome_uses_canonical_promotion_destinations() -> None:
    spec, spec_hash = build_frozen_outcome_spec(
        goal_metric="booking_conversion_rate",
        target_segment_rules=[
            segment_rule("funnel_recovery", ["오키나와", "jeju"]),
            segment_rule("target_destination_affinity", ["jeju", "okinawa"]),
        ],
    )

    assert spec == {
        "outcome_metric": "booking_conversion_rate",
        "outcome_event_name": "booking_complete",
        "outcome_filter": {"destination_ids": ["jeju", "okinawa"]},
        "outcome_definition_version": "booking-outcome.v1",
        "uplift_training_eligible": True,
    }
    assert spec_hash == outcome_spec_hash(spec)


def test_unsupported_goal_is_frozen_but_not_training_eligible() -> None:
    spec, spec_hash = build_frozen_outcome_spec(
        goal_metric="inflow_rate",
        target_segment_rules=[],
    )

    assert spec["outcome_event_name"] is None
    assert spec["outcome_definition_version"] == "unsupported.v1"
    assert spec["uplift_training_eligible"] is False
    assert spec["exclusion_reason"] == "unsupported_goal_metric"
    assert require_frozen_outcome_spec(
        {"outcome_spec": spec, "outcome_spec_hash": spec_hash}
    ) == (spec, spec_hash)


def test_frozen_outcome_hash_detects_mutation() -> None:
    spec, spec_hash = build_frozen_outcome_spec(
        goal_metric="booking_conversion_rate",
        target_segment_rules=[],
    )
    spec["outcome_filter"] = {"destination_ids": ["seoul"]}

    try:
        require_frozen_outcome_spec(
            {"outcome_spec": spec, "outcome_spec_hash": spec_hash}
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mutated frozen outcome spec must be rejected")


def segment_rule(candidate_type: str, destination_ids: list[str]) -> dict[str, object]:
    return {
        "audience_resolution_contract": "segment_audience.v1",
        "segment_audience_spec": dict(
            RegisteredSegmentAudienceBinder().bind(
                candidate_type=candidate_type,
                destination_ids=destination_ids,
            )
        ),
    }
