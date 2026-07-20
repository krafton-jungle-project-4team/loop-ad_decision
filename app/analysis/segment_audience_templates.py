from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from app.analysis.behavior_manifest import (
    canonical_destination_id,
    load_behavior_manifest,
    manifest_intent_benefit_keys,
)


SEGMENT_AUDIENCE_TEMPLATE_VERSION = 1
SEGMENT_AUDIENCE_OBSERVATION_WINDOW_DAYS = 30
MAX_DESTINATION_IDS = 8
MAX_SEASON_MONTHS = 12
MAX_BENEFIT_KEYS = 4

_MANIFEST = load_behavior_manifest()
_REGISTERED_DESTINATION_IDS = frozenset(
    str(value) for value in _MANIFEST["destination_aliases"]
)
_REGISTERED_BENEFIT_KEYS = frozenset(manifest_intent_benefit_keys())
_REGISTERED_DESTINATION_ID_PATTERNS = tuple(
    re.compile(str(value))
    for value in _MANIFEST["canonical_destination_id_patterns"]
)


@dataclass(frozen=True, slots=True)
class SegmentAudienceTemplate:
    template_id: str
    template_version: int
    candidate_type: str
    query_signal_keys: tuple[str, ...]
    base_hard_predicate_keys: tuple[str, ...]
    parameter_policy_id: str
    semantic_selection_policy_id: str
    semantic_anchor_policy_id: str
    destination_min: int = 0
    destination_max: int = 0
    season_min: int = 0
    season_max: int = 0
    benefit_min: int = 0
    benefit_max: int = 0
    destination_adds_recent_predicate: bool = False
    season_adds_match_predicate: bool = False
    observation_window_days: int = SEGMENT_AUDIENCE_OBSERVATION_WINDOW_DAYS

    @property
    def semantic_payload(self) -> Mapping[str, Any]:
        return {
            "template_id": self.template_id,
            "template_version": self.template_version,
            "candidate_type": self.candidate_type,
            "query_signal_keys": list(self.query_signal_keys),
            "base_hard_predicate_keys": list(self.base_hard_predicate_keys),
            "parameter_policy": {
                "policy_id": self.parameter_policy_id,
                "destination_min": self.destination_min,
                "destination_max": self.destination_max,
                "season_min": self.season_min,
                "season_max": self.season_max,
                "benefit_min": self.benefit_min,
                "benefit_max": self.benefit_max,
                "destination_allowlist": "hotel_destination_alias.v1",
                "destination_canonical_id_patterns": list(
                    _MANIFEST["canonical_destination_id_patterns"]
                ),
                "benefit_allowlist": "hotel_booking_behavior.v2",
                "destination_adds_recent_predicate": (
                    self.destination_adds_recent_predicate
                ),
                "season_adds_match_predicate": self.season_adds_match_predicate,
            },
            "semantic_selection_policy_id": self.semantic_selection_policy_id,
            "semantic_anchor_policy_id": self.semantic_anchor_policy_id,
            "observation_window_days": self.observation_window_days,
        }

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(
            self.semantic_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def hard_predicate_keys(
        self,
        *,
        destination_ids: Sequence[str],
        season_months: Sequence[int],
    ) -> tuple[str, ...]:
        values = list(self.base_hard_predicate_keys)
        if destination_ids and self.destination_adds_recent_predicate:
            values.append("recent_destination_search")
        if season_months and self.season_adds_match_predicate:
            values.append("season_match")
        return tuple(values)


def _template(
    candidate_type: str,
    *,
    destination_min: int = 0,
    destination_max: int = 0,
    season_min: int = 0,
    season_max: int = 0,
    benefit_min: int = 0,
    benefit_max: int = 0,
    destination_adds_recent_predicate: bool = False,
    season_adds_match_predicate: bool = False,
) -> SegmentAudienceTemplate:
    return SegmentAudienceTemplate(
        template_id=f"hotel.{candidate_type}.v1",
        template_version=SEGMENT_AUDIENCE_TEMPLATE_VERSION,
        candidate_type=candidate_type,
        query_signal_keys=tuple(
            str(value)
            for value in _MANIFEST["candidate_query_dimensions"][candidate_type]
        ),
        base_hard_predicate_keys=tuple(
            str(value)
            for value in _MANIFEST["candidate_hard_predicates"][candidate_type]
        ),
        parameter_policy_id=f"hotel.{candidate_type}.params.v1",
        semantic_selection_policy_id=f"hotel.{candidate_type}.selection.v1",
        semantic_anchor_policy_id=f"hotel.{candidate_type}.anchors.v1",
        destination_min=destination_min,
        destination_max=destination_max,
        season_min=season_min,
        season_max=season_max,
        benefit_min=benefit_min,
        benefit_max=benefit_max,
        destination_adds_recent_predicate=destination_adds_recent_predicate,
        season_adds_match_predicate=season_adds_match_predicate,
    )


_TEMPLATES = (
    _template(
        "intent_matched",
        destination_max=MAX_DESTINATION_IDS,
        season_max=MAX_SEASON_MONTHS,
        destination_adds_recent_predicate=True,
        season_adds_match_predicate=True,
    ),
    _template(
        "target_destination_affinity",
        destination_min=1,
        destination_max=MAX_DESTINATION_IDS,
        destination_adds_recent_predicate=True,
    ),
    _template(
        "funnel_recovery",
        destination_max=MAX_DESTINATION_IDS,
        destination_adds_recent_predicate=True,
    ),
    _template(
        "benefit_value_seeker",
        destination_max=MAX_DESTINATION_IDS,
        benefit_max=MAX_BENEFIT_KEYS,
        destination_adds_recent_predicate=True,
    ),
    _template("promotion_responsive"),
    _template("general_destination_explorer"),
)

REGISTERED_SEGMENT_AUDIENCE_TEMPLATES: Mapping[str, SegmentAudienceTemplate] = (
    MappingProxyType({template.template_id: template for template in _TEMPLATES})
)
TEMPLATE_ID_BY_CANDIDATE_TYPE: Mapping[str, str] = MappingProxyType(
    {template.candidate_type: template.template_id for template in _TEMPLATES}
)


class RegisteredSegmentAudienceBinder:
    """Serialize a fixed template binding without reading promotion text."""

    def bind(
        self,
        *,
        candidate_type: str,
        destination_ids: Sequence[str] = (),
        season_months: Sequence[int] = (),
        benefit_keys: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        template_id = TEMPLATE_ID_BY_CANDIDATE_TYPE.get(candidate_type)
        if template_id is None:
            raise ValueError(f"unregistered segment audience candidate: {candidate_type}")
        template = REGISTERED_SEGMENT_AUDIENCE_TEMPLATES[template_id]
        destinations = canonical_destination_ids(destination_ids)
        seasons = canonical_season_months(season_months)
        benefits = canonical_benefit_keys(benefit_keys)
        validate_template_parameters(
            template,
            destination_ids=destinations,
            season_months=seasons,
            benefit_keys=benefits,
        )
        hard_predicates = template.hard_predicate_keys(
            destination_ids=destinations,
            season_months=seasons,
        )
        return {
            "schema_version": "hotel_behavior.v2",
            "template_id": template.template_id,
            "template_version": template.template_version,
            "template_semantic_hash": template.semantic_hash,
            "candidate_type": template.candidate_type,
            "condition_keys": list(hard_predicates),
            "query_signal_keys": list(template.query_signal_keys),
            "hard_predicate_keys": list(hard_predicates),
            "parameters": {
                "destination_ids": list(destinations),
                "season_months": list(seasons),
                "benefit_keys": list(benefits),
            },
            "parameter_policy_id": template.parameter_policy_id,
            "semantic_selection_policy_id": (
                template.semantic_selection_policy_id
            ),
            "semantic_anchor_policy_id": template.semantic_anchor_policy_id,
            "observation_window_days": template.observation_window_days,
        }


def require_registered_template(template_id: str) -> SegmentAudienceTemplate:
    try:
        return REGISTERED_SEGMENT_AUDIENCE_TEMPLATES[template_id]
    except KeyError as exc:
        raise ValueError(f"unregistered segment audience template: {template_id}") from exc


def canonical_destination_ids(values: Sequence[str]) -> tuple[str, ...]:
    canonical: set[str] = set()
    for value in values:
        normalized = canonical_destination_id(str(value))
        if not normalized or (
            normalized not in _REGISTERED_DESTINATION_IDS
            and not any(
                pattern.fullmatch(normalized)
                for pattern in _REGISTERED_DESTINATION_ID_PATTERNS
            )
        ):
            raise ValueError(f"unregistered canonical destination: {value}")
        canonical.add(normalized)
    return tuple(sorted(canonical))


def canonical_season_months(values: Sequence[int]) -> tuple[int, ...]:
    months: set[int] = set()
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 12:
            raise ValueError("season months must be integers between 1 and 12")
        months.add(value)
    return tuple(sorted(months))


def canonical_benefit_keys(values: Sequence[str]) -> tuple[str, ...]:
    benefits: set[str] = set()
    for value in values:
        normalized = str(value).strip().casefold()
        if normalized not in _REGISTERED_BENEFIT_KEYS:
            raise ValueError(f"unregistered benefit key: {value}")
        benefits.add(normalized)
    return tuple(sorted(benefits))


def validate_template_parameters(
    template: SegmentAudienceTemplate,
    *,
    destination_ids: Sequence[str],
    season_months: Sequence[int],
    benefit_keys: Sequence[str],
) -> None:
    checks = (
        (
            "destination_ids",
            len(destination_ids),
            template.destination_min,
            template.destination_max,
        ),
        (
            "season_months",
            len(season_months),
            template.season_min,
            template.season_max,
        ),
        (
            "benefit_keys",
            len(benefit_keys),
            template.benefit_min,
            template.benefit_max,
        ),
    )
    for name, count, minimum, maximum in checks:
        if count < minimum or count > maximum:
            raise ValueError(
                f"{template.template_id} requires {name} count between "
                f"{minimum} and {maximum}"
            )
