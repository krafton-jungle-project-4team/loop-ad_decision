from __future__ import annotations

from copy import deepcopy

import pytest

from app.analysis.raw_event_segments import (
    INTENT_EXTRACTOR_VERSION,
    RAW_EVENT_SEGMENT_VERSION,
)
from app.analysis.segment_audience_backfill import (
    SegmentAudienceBackfillError,
    plan_segment_audience_backfill,
)


def _row(
    segment_id: str,
    *,
    candidate_type: str = "funnel_recovery",
    destinations: list[str] | None = None,
    seasons: list[str] | None = None,
    benefits: list[str] | None = None,
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "source": "ai_suggested",
        "rule_json": {
            "version": RAW_EVENT_SEGMENT_VERSION,
            "candidate_type": candidate_type,
            "candidate_user_ids": ["user_1", "user_2"],
        },
        "profile_json": {
            "promotion_intent": {
                "version": INTENT_EXTRACTOR_VERSION,
                "destinations": destinations or [],
                "season": seasons or [],
                "benefits": benefits or [],
            }
        },
    }


def test_backfill_is_explicit_additive_and_idempotent() -> None:
    original = _row(
        "seg_existing",
        candidate_type="intent_matched",
        destinations=["제주", "jeju"],
        seasons=["summer"],
    )
    plans = plan_segment_audience_backfill(
        [original],
        requested_segment_ids=["seg_existing"],
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.segment_id == "seg_existing"
    assert plan.changed is True
    assert plan.rule_json["candidate_user_ids"] == ["user_1", "user_2"]
    spec = plan.rule_json["segment_audience_spec"]
    assert spec["template_id"] == "hotel.intent_matched.v1"
    assert spec["parameters"] == {
        "destination_ids": ["jeju"],
        "season_months": [6, 7, 8],
        "benefit_keys": [],
    }
    assert "audience_resolution_contract" not in original["rule_json"]

    rebound = deepcopy(original)
    rebound["rule_json"] = dict(plan.rule_json)
    repeated = plan_segment_audience_backfill(
        [rebound],
        requested_segment_ids=["seg_existing"],
    )
    assert repeated[0].changed is False
    assert repeated[0].rule_json == plan.rule_json


def test_backfill_rejects_missing_rows_and_conflicting_bindings_as_one_plan() -> None:
    with pytest.raises(SegmentAudienceBackfillError, match="do not exist"):
        plan_segment_audience_backfill(
            [_row("seg_a")],
            requested_segment_ids=["seg_a", "seg_b"],
        )

    first = _row("seg_a")
    conflicting = _row("seg_b")
    conflicting["rule_json"] = {
        **conflicting["rule_json"],
        "audience_resolution_contract": "segment_audience.v1",
        "segment_audience_spec": {"template_id": "unregistered"},
    }
    with pytest.raises(SegmentAudienceBackfillError, match="conflicting"):
        plan_segment_audience_backfill(
            [first, conflicting],
            requested_segment_ids=["seg_a", "seg_b"],
        )
    assert "audience_resolution_contract" not in first["rule_json"]


def test_backfill_never_reconstructs_parameters_from_promotion_text() -> None:
    row = _row(
        "seg_destination",
        candidate_type="target_destination_affinity",
    )
    row["promotion_message"] = "제주 숙소 프로모션"

    with pytest.raises(SegmentAudienceBackfillError, match="destination_ids"):
        plan_segment_audience_backfill(
            [row],
            requested_segment_ids=["seg_destination"],
        )


def test_backfill_rejects_manual_segments() -> None:
    row = _row("seg_manual")
    row["source"] = "manual_rule"

    with pytest.raises(SegmentAudienceBackfillError, match="ai_suggested"):
        plan_segment_audience_backfill(
            [row],
            requested_segment_ids=["seg_manual"],
        )
