from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.audience_contract import SEGMENT_AUDIENCE_CONTRACT
from app.analysis.raw_event_segments import (
    INTENT_EXTRACTOR_VERSION,
    RAW_EVENT_SEGMENT_VERSION,
    SEASON_MONTHS,
)
from app.analysis.segment_audience_templates import (
    RegisteredSegmentAudienceBinder,
)


@dataclass(frozen=True, slots=True)
class SegmentAudienceBackfillPlan:
    segment_id: str
    rule_json: Mapping[str, Any]
    changed: bool


class SegmentAudienceBackfillError(RuntimeError):
    pass


def plan_segment_audience_backfill(
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_segment_ids: Sequence[str],
    binder: RegisteredSegmentAudienceBinder | None = None,
) -> tuple[SegmentAudienceBackfillPlan, ...]:
    expected = tuple(sorted(set(requested_segment_ids)))
    found = tuple(sorted(str(row["segment_id"]) for row in rows))
    if found != expected:
        missing = sorted(set(expected) - set(found))
        raise SegmentAudienceBackfillError(
            "segment definitions do not exist: " + ", ".join(missing)
        )
    template_binder = binder or RegisteredSegmentAudienceBinder()
    plans: list[SegmentAudienceBackfillPlan] = []
    for row in rows:
        segment_id = str(row["segment_id"])
        if str(row.get("source", "")) != "ai_suggested":
            raise SegmentAudienceBackfillError(
                f"{segment_id}: only ai_suggested segments can be backfilled"
            )
        rule_json = row.get("rule_json")
        profile_json = row.get("profile_json")
        if not isinstance(rule_json, Mapping) or not isinstance(
            profile_json,
            Mapping,
        ):
            raise SegmentAudienceBackfillError(
                f"{segment_id}: rule_json and profile_json must be objects"
            )
        if rule_json.get("version") != RAW_EVENT_SEGMENT_VERSION:
            raise SegmentAudienceBackfillError(
                f"{segment_id}: unsupported raw-event segment version"
            )
        candidate_type = str(rule_json.get("candidate_type", "")).strip()
        promotion_intent = profile_json.get("promotion_intent")
        if (
            not isinstance(promotion_intent, Mapping)
            or promotion_intent.get("version") != INTENT_EXTRACTOR_VERSION
        ):
            raise SegmentAudienceBackfillError(
                f"{segment_id}: versioned promotion_intent is required"
            )
        destinations = _text_values(promotion_intent.get("destinations"))
        benefits = _text_values(promotion_intent.get("benefits"))
        season_months = _season_months(
            _text_values(promotion_intent.get("season")),
            segment_id=segment_id,
        )
        if candidate_type not in {
            "intent_matched",
            "target_destination_affinity",
            "funnel_recovery",
            "benefit_value_seeker",
        }:
            destinations = ()
        if candidate_type != "intent_matched":
            season_months = ()
        if candidate_type != "benefit_value_seeker":
            benefits = ()
        try:
            audience_spec = template_binder.bind(
                candidate_type=candidate_type,
                destination_ids=destinations,
                season_months=season_months,
                benefit_keys=benefits,
            )
        except ValueError as exc:
            raise SegmentAudienceBackfillError(
                f"{segment_id}: {exc}"
            ) from exc
        existing_contract = rule_json.get("audience_resolution_contract")
        existing_spec = rule_json.get("segment_audience_spec")
        if existing_contract == SEGMENT_AUDIENCE_CONTRACT:
            if existing_spec != audience_spec:
                raise SegmentAudienceBackfillError(
                    f"{segment_id}: conflicting V2 audience binding exists"
                )
            plans.append(
                SegmentAudienceBackfillPlan(
                    segment_id=segment_id,
                    rule_json=dict(rule_json),
                    changed=False,
                )
            )
            continue
        if existing_contract not in {None, "legacy"} or existing_spec is not None:
            raise SegmentAudienceBackfillError(
                f"{segment_id}: existing audience contract cannot be overwritten"
            )
        plans.append(
            SegmentAudienceBackfillPlan(
                segment_id=segment_id,
                rule_json={
                    **dict(rule_json),
                    "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
                    "segment_audience_spec": dict(audience_spec),
                },
                changed=True,
            )
        )
    return tuple(plans)


def _text_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SegmentAudienceBackfillError(
            "promotion_intent parameter arrays must contain strings"
        )
    return tuple(value)


def _season_months(
    seasons: Sequence[str],
    *,
    segment_id: str,
) -> tuple[int, ...]:
    months: list[int] = []
    for value in seasons:
        normalized = value.strip().casefold()
        if normalized not in SEASON_MONTHS:
            raise SegmentAudienceBackfillError(
                f"{segment_id}: unregistered season value: {value}"
            )
        months.extend(SEASON_MONTHS[normalized])
    return tuple(sorted(set(months)))
