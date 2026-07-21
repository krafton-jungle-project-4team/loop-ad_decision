from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from app.audience_contract import SegmentDefinitionAudienceAdapter
from app.analysis.segment_audience_templates import canonical_destination_ids


BOOKING_CONVERSION_RATE = "booking_conversion_rate"
BOOKING_OUTCOME_DEFINITION_VERSION = "booking-outcome.v1"
UNSUPPORTED_OUTCOME_DEFINITION_VERSION = "unsupported.v1"


def build_frozen_outcome_spec(
    *,
    goal_metric: str,
    target_segment_rules: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    destination_ids = _target_destination_ids(target_segment_rules)
    if goal_metric == BOOKING_CONVERSION_RATE:
        outcome_spec: dict[str, Any] = {
            "outcome_metric": BOOKING_CONVERSION_RATE,
            "outcome_event_name": "booking_complete",
            "outcome_filter": {"destination_ids": list(destination_ids)},
            "outcome_definition_version": BOOKING_OUTCOME_DEFINITION_VERSION,
            "uplift_training_eligible": True,
        }
    else:
        outcome_spec = {
            "outcome_metric": goal_metric,
            "outcome_event_name": None,
            "outcome_filter": {"destination_ids": list(destination_ids)},
            "outcome_definition_version": UNSUPPORTED_OUTCOME_DEFINITION_VERSION,
            "uplift_training_eligible": False,
            "exclusion_reason": "unsupported_goal_metric",
        }
    return outcome_spec, outcome_spec_hash(outcome_spec)


def outcome_spec_hash(outcome_spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(outcome_spec).encode("utf-8")).hexdigest()


def require_frozen_outcome_spec(
    goal_snapshot_json: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    raw_spec = goal_snapshot_json.get("outcome_spec")
    raw_hash = goal_snapshot_json.get("outcome_spec_hash")
    if not isinstance(raw_spec, Mapping):
        raise ValueError("promotion run outcome_spec is required")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        raise ValueError("promotion run outcome_spec_hash is invalid")
    normalized = dict(raw_spec)
    if outcome_spec_hash(normalized) != raw_hash:
        raise ValueError("promotion run outcome_spec_hash does not match outcome_spec")
    return normalized, raw_hash


def _target_destination_ids(
    target_segment_rules: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    adapter = SegmentDefinitionAudienceAdapter()
    destination_ids: set[str] = set()
    for index, rule_json in enumerate(target_segment_rules):
        resolution = adapter.resolve(
            segment_id=str(rule_json.get("segment_id") or f"target_{index}"),
            rule_json=rule_json,
        )
        if resolution.spec is not None:
            destination_ids.update(resolution.spec.destination_ids)
            custom_destinations = [
                condition.get("destination")
                for condition in resolution.spec.custom_conditions
                if isinstance(condition.get("destination"), str)
                and condition.get("destination")
            ]
            destination_ids.update(
                canonical_destination_ids(custom_destinations)
            )
    return tuple(sorted(destination_ids))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
