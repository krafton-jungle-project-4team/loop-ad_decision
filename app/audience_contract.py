from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from app.analysis.behavior_manifest import behavior_manifest_hash
from app.analysis.segment_audience_templates import (
    canonical_benefit_keys,
    canonical_destination_ids,
    canonical_season_months,
    require_registered_template,
    validate_template_parameters,
)


LEGACY_AUDIENCE_CONTRACT = "legacy"
SEGMENT_AUDIENCE_CONTRACT = "segment_audience.v1"
SEGMENT_AUDIENCE_SCHEMA_VERSION = "hotel_behavior.v2"
SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION = "segment_behavior_query.v2"
CUSTOM_STRUCTURED_TEMPLATE_ID = "custom_structured_condition"
CUSTOM_STRUCTURED_TEMPLATE_VERSION = 1
CUSTOM_STRUCTURED_CANDIDATE_TYPE = "custom_structured"
CUSTOM_STRUCTURED_WINDOW_DAYS = 30
CUSTOM_STRUCTURED_MIN_WINDOW_DAYS = 1
CUSTOM_STRUCTURED_MAX_WINDOW_DAYS = 365
CUSTOM_STRUCTURED_PARAMETER_POLICY_ID = "custom_structured_parameters.v1"
CUSTOM_STRUCTURED_SELECTION_POLICY_ID = "exact_predicate_membership.v1"
CUSTOM_STRUCTURED_ANCHOR_POLICY_ID = "structured_conditions_no_anchor.v1"
CUSTOM_STRUCTURED_CONDITION_KEY = "structured_conditions"
CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION = 2
CUSTOM_SOURCE_REFINEMENT_PARAMETER_POLICY_ID = "custom_structured_parameters.v2"
CUSTOM_SOURCE_REFINEMENT_SELECTION_POLICY_ID = "source_refinement_exact_membership.v1"
CUSTOM_SOURCE_REFINEMENT_ANCHOR_POLICY_ID = (
    "source_membership_with_optional_structured_conditions.v1"
)
CUSTOM_SOURCE_MEMBERSHIP_CONDITION_KEY = "source_audience_membership"


def custom_structured_template_hash(
    *,
    template_version: int,
    window_days: int,
) -> str:
    if (
        not isinstance(window_days, int)
        or isinstance(window_days, bool)
        or not CUSTOM_STRUCTURED_MIN_WINDOW_DAYS
        <= window_days
        <= CUSTOM_STRUCTURED_MAX_WINDOW_DAYS
    ):
        raise ValueError("custom structured window must be between 1 and 365 days")
    if template_version == CUSTOM_STRUCTURED_TEMPLATE_VERSION:
        semantics = {
            "candidate_type": CUSTOM_STRUCTURED_CANDIDATE_TYPE,
            "conditions": "allowlisted_event_property_count_conjunction",
            "schema_version": SEGMENT_AUDIENCE_SCHEMA_VERSION,
            "selection": "exact_predicate_membership_vector_tiebreak_only",
            "template_id": CUSTOM_STRUCTURED_TEMPLATE_ID,
            "template_version": CUSTOM_STRUCTURED_TEMPLATE_VERSION,
            "window_days": window_days,
        }
    elif template_version == CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION:
        semantics = {
            "base_membership": "canonical_source_suggestion_user_ids",
            "candidate_type": CUSTOM_STRUCTURED_CANDIDATE_TYPE,
            "conditions": "optional_allowlisted_event_property_count_conjunction",
            "schema_version": SEGMENT_AUDIENCE_SCHEMA_VERSION,
            "selection": "source_membership_with_optional_exact_predicate_membership",
            "template_id": CUSTOM_STRUCTURED_TEMPLATE_ID,
            "template_version": CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION,
            "window_days": window_days,
        }
    else:
        raise ValueError("unsupported custom structured template version")
    return hashlib.sha256(
        json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


CUSTOM_STRUCTURED_TEMPLATE_HASH = custom_structured_template_hash(
    template_version=CUSTOM_STRUCTURED_TEMPLATE_VERSION,
    window_days=CUSTOM_STRUCTURED_WINDOW_DAYS,
)
CUSTOM_SOURCE_REFINEMENT_TEMPLATE_HASH = custom_structured_template_hash(
    template_version=CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION,
    window_days=CUSTOM_STRUCTURED_WINDOW_DAYS,
)

CUSTOM_STRUCTURED_EVENT_NAMES = frozenset(
    {
        "page_view",
        "hotel_search",
        "hotel_click",
        "hotel_detail_view",
        "promotion_impression",
        "promotion_click",
        "campaign_redirect_click",
        "campaign_landing",
        "booking_start",
        "booking_complete",
        "booking_cancel",
    }
)
CUSTOM_STRUCTURED_PROPERTY_KEYS = frozenset(
    {
        "deal",
        "free_cancellation",
        "breakfast_included",
        "age_group",
        "gender",
        "region",
        "preferred_category",
        "user_segment",
        "adult_count",
        "child_count",
        "rooms",
        "hotel_id",
        "hotel_name",
        "hotel_city",
        "hotel_country",
        "hotel_market",
        "hotel_cluster",
        "hotel_star_rating",
        "hotel_guest_rating",
        "price",
        "property_type",
        "room_type",
        "revenue",
    }
)
CUSTOM_STRUCTURED_PROPERTY_OPERATORS = frozenset(
    {"equals", "in", "contains", "exists", "gte", "lte"}
)
SCORE_THRESHOLD_QUANTUM = Decimal("0.000001")
_QUERY_COMPILER_SEMANTICS = {
    "version": SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION,
    "schema_version": SEGMENT_AUDIENCE_SCHEMA_VERSION,
    "behavior_manifest_hash": behavior_manifest_hash(),
    "input": "registered_segment_audience_template_only",
    "destination_encoding": "canonical_id_sha256_signed_bucket",
    "normalization": "manifest_block_weight_then_global_l2",
    "predicate_policy": "registered_template_exact_binding",
}
SEGMENT_AUDIENCE_QUERY_COMPILER_HASH = hashlib.sha256(
    json.dumps(
        _QUERY_COMPILER_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def contract_score_threshold(value: float | Decimal) -> Decimal:
    """Normalize to the NUMERIC(10, 6) precision in the data contract."""
    return Decimal(str(value)).quantize(
        SCORE_THRESHOLD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


class SegmentAudienceContractError(RuntimeError):
    def __init__(self, *, code: str, segment_id: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.segment_id = segment_id
        self.reason = reason

    def to_detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "segment_id": self.segment_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SegmentAudienceSpec:
    segment_id: str
    schema_version: str
    template_id: str
    template_version: int
    template_semantic_hash: str
    candidate_type: str
    condition_keys: tuple[str, ...]
    query_signal_keys: tuple[str, ...]
    hard_predicate_keys: tuple[str, ...]
    destination_ids: tuple[str, ...]
    season_months: tuple[int, ...]
    benefit_keys: tuple[str, ...]
    parameter_policy_id: str
    semantic_selection_policy_id: str
    semantic_anchor_policy_id: str
    observation_window_days: int
    spec_hash: str
    custom_conditions: tuple[Mapping[str, Any], ...] = ()
    base_user_ids: tuple[str, ...] = ()

    @property
    def is_custom_structured(self) -> bool:
        return self.template_id == CUSTOM_STRUCTURED_TEMPLATE_ID

    @property
    def is_source_refinement(self) -> bool:
        return bool(self.base_user_ids)

    @property
    def predicate_parameters(self) -> Mapping[str, Sequence[str] | Sequence[int]]:
        parameters: dict[str, Sequence[str] | Sequence[int]] = {
            "destinations": self.destination_ids,
            "season_months": self.season_months,
            "benefit_keys": self.benefit_keys,
        }
        if self.is_custom_structured:
            parameters["structured_conditions_json"] = (
                json.dumps(
                    list(self.custom_conditions),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            parameters["observation_window_days"] = (
                self.observation_window_days,
            )
        if self.is_source_refinement:
            parameters["base_user_ids"] = self.base_user_ids
        return parameters


@dataclass(frozen=True, slots=True)
class AudienceResolution:
    contract: str
    spec: SegmentAudienceSpec | None = None

    @property
    def is_v2(self) -> bool:
        return self.contract == SEGMENT_AUDIENCE_CONTRACT


class SegmentDefinitionAudienceAdapter:
    """Validate serialized segment meaning without inferring or repairing it."""

    def resolve(
        self,
        *,
        segment_id: str,
        rule_json: Mapping[str, Any],
    ) -> AudienceResolution:
        raw_contract = rule_json.get("audience_resolution_contract")
        if raw_contract is None or raw_contract == LEGACY_AUDIENCE_CONTRACT:
            return AudienceResolution(contract=LEGACY_AUDIENCE_CONTRACT)
        if raw_contract != SEGMENT_AUDIENCE_CONTRACT:
            raise _error(
                "segment_audience_contract_unsupported",
                segment_id,
                f"unsupported audience resolution contract: {raw_contract}",
            )
        raw_spec = rule_json.get("segment_audience_spec")
        if not isinstance(raw_spec, Mapping):
            raise _error(
                "segment_audience_spec_missing",
                segment_id,
                "segment_audience.v1 requires rule_json.segment_audience_spec",
            )
        return AudienceResolution(
            contract=SEGMENT_AUDIENCE_CONTRACT,
            spec=_parse_spec(segment_id=segment_id, raw_spec=raw_spec),
        )


def audience_contract_from_rule_json(rule_json: Mapping[str, Any]) -> str:
    value = rule_json.get("audience_resolution_contract")
    return LEGACY_AUDIENCE_CONTRACT if value is None else str(value)


def _parse_spec(
    *,
    segment_id: str,
    raw_spec: Mapping[str, Any],
) -> SegmentAudienceSpec:
    schema_version = _required_text(raw_spec, "schema_version", segment_id)
    if schema_version != SEGMENT_AUDIENCE_SCHEMA_VERSION:
        raise _error(
            "segment_audience_schema_unsupported",
            segment_id,
            f"segment audience schema must be {SEGMENT_AUDIENCE_SCHEMA_VERSION}",
        )
    template_id = _required_text(raw_spec, "template_id", segment_id)
    if template_id == CUSTOM_STRUCTURED_TEMPLATE_ID:
        return _parse_custom_structured_spec(
            segment_id=segment_id,
            schema_version=schema_version,
            raw_spec=raw_spec,
        )
    try:
        template = require_registered_template(template_id)
    except ValueError as exc:
        raise _error(
            "segment_audience_template_unregistered",
            segment_id,
            str(exc),
        ) from exc
    template_version = raw_spec.get("template_version")
    if template_version != template.template_version:
        raise _error(
            "segment_audience_template_version_mismatch",
            segment_id,
            f"{template_id} requires template_version={template.template_version}",
        )
    template_semantic_hash = _required_text(
        raw_spec,
        "template_semantic_hash",
        segment_id,
    )
    if template_semantic_hash != template.semantic_hash:
        raise _error(
            "segment_audience_template_hash_mismatch",
            segment_id,
            "segment template semantic hash does not match the registry",
        )
    candidate_type = _required_text(raw_spec, "candidate_type", segment_id)
    if candidate_type != template.candidate_type:
        raise _error(
            "segment_audience_template_binding_mismatch",
            segment_id,
            "candidate_type does not match the registered template",
        )

    condition_keys = _required_text_tuple(raw_spec, "condition_keys", segment_id)
    query_signal_keys = _required_text_tuple(
        raw_spec,
        "query_signal_keys",
        segment_id,
    )
    if query_signal_keys != template.query_signal_keys:
        raise _error(
            "segment_audience_template_binding_mismatch",
            segment_id,
            "query_signal_keys must exactly match the registered template",
        )

    raw_parameters = raw_spec.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise _error(
            "segment_audience_parameters_invalid",
            segment_id,
            "segment audience parameters must be an object",
        )
    raw_destination_ids = _raw_text_tuple(
        raw_parameters.get("destination_ids", ()),
        segment_id=segment_id,
        field_name="parameters.destination_ids",
        allow_empty=True,
    )
    raw_season_months = _raw_month_tuple(
        raw_parameters.get("season_months", ()),
        segment_id=segment_id,
    )
    raw_benefit_keys = _raw_text_tuple(
        raw_parameters.get("benefit_keys", ()),
        segment_id=segment_id,
        field_name="parameters.benefit_keys",
        allow_empty=True,
    )
    try:
        destination_ids = canonical_destination_ids(raw_destination_ids)
        season_months = canonical_season_months(raw_season_months)
        benefit_keys = canonical_benefit_keys(raw_benefit_keys)
        validate_template_parameters(
            template,
            destination_ids=destination_ids,
            season_months=season_months,
            benefit_keys=benefit_keys,
        )
    except ValueError as exc:
        raise _error(
            "segment_audience_parameter_policy_unsupported",
            segment_id,
            str(exc),
        ) from exc
    if (
        raw_destination_ids != destination_ids
        or raw_season_months != season_months
        or raw_benefit_keys != benefit_keys
    ):
        raise _error(
            "segment_audience_parameters_not_canonical",
            segment_id,
            "segment audience parameter arrays must be unique and canonically sorted",
        )

    hard_predicate_keys = _required_text_tuple(
        raw_spec,
        "hard_predicate_keys",
        segment_id,
    )
    expected_predicates = template.hard_predicate_keys(
        destination_ids=destination_ids,
        season_months=season_months,
    )
    if hard_predicate_keys != expected_predicates or condition_keys != expected_predicates:
        raise _error(
            "segment_audience_template_binding_mismatch",
            segment_id,
            "condition_keys and hard_predicate_keys must match the template",
        )

    parameter_policy_id = _required_text(
        raw_spec,
        "parameter_policy_id",
        segment_id,
    )
    semantic_selection_policy_id = _required_text(
        raw_spec,
        "semantic_selection_policy_id",
        segment_id,
    )
    semantic_anchor_policy_id = _required_text(
        raw_spec,
        "semantic_anchor_policy_id",
        segment_id,
    )
    expected_policy_values = (
        template.parameter_policy_id,
        template.semantic_selection_policy_id,
        template.semantic_anchor_policy_id,
    )
    if (
        parameter_policy_id,
        semantic_selection_policy_id,
        semantic_anchor_policy_id,
    ) != expected_policy_values:
        raise _error(
            "segment_audience_template_binding_mismatch",
            segment_id,
            "template policy identifiers do not match the registry",
        )
    observation_window_days = raw_spec.get("observation_window_days")
    if observation_window_days != template.observation_window_days:
        raise _error(
            "segment_audience_window_invalid",
            segment_id,
            f"{template_id} requires a {template.observation_window_days}-day window",
        )

    canonical = {
        "schema_version": schema_version,
        "template_id": template_id,
        "template_version": template_version,
        "template_semantic_hash": template_semantic_hash,
        "candidate_type": candidate_type,
        "condition_keys": list(condition_keys),
        "query_signal_keys": list(query_signal_keys),
        "hard_predicate_keys": list(hard_predicate_keys),
        "parameters": {
            "destination_ids": list(destination_ids),
            "season_months": list(season_months),
            "benefit_keys": list(benefit_keys),
        },
        "parameter_policy_id": parameter_policy_id,
        "semantic_selection_policy_id": semantic_selection_policy_id,
        "semantic_anchor_policy_id": semantic_anchor_policy_id,
        "observation_window_days": observation_window_days,
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return SegmentAudienceSpec(
        segment_id=segment_id,
        schema_version=schema_version,
        template_id=template_id,
        template_version=template_version,
        template_semantic_hash=template_semantic_hash,
        candidate_type=candidate_type,
        condition_keys=condition_keys,
        query_signal_keys=query_signal_keys,
        hard_predicate_keys=hard_predicate_keys,
        destination_ids=destination_ids,
        season_months=season_months,
        benefit_keys=benefit_keys,
        parameter_policy_id=parameter_policy_id,
        semantic_selection_policy_id=semantic_selection_policy_id,
        semantic_anchor_policy_id=semantic_anchor_policy_id,
        observation_window_days=observation_window_days,
        spec_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )


def _parse_custom_structured_spec(
    *,
    segment_id: str,
    schema_version: str,
    raw_spec: Mapping[str, Any],
) -> SegmentAudienceSpec:
    raw_parameters = raw_spec.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise _error(
            "segment_audience_parameters_invalid",
            segment_id,
            "custom structured parameters must be an object",
        )
    lookback_days = raw_parameters.get("lookback_days")
    if (
        not isinstance(lookback_days, int)
        or isinstance(lookback_days, bool)
        or not CUSTOM_STRUCTURED_MIN_WINDOW_DAYS
        <= lookback_days
        <= CUSTOM_STRUCTURED_MAX_WINDOW_DAYS
    ):
        raise _error(
            "segment_audience_window_invalid",
            segment_id,
            "custom structured lookback_days must be between 1 and 365",
        )
    template_version = raw_spec.get("template_version")
    if template_version == CUSTOM_STRUCTURED_TEMPLATE_VERSION:
        expected_static = {
            "template_version": CUSTOM_STRUCTURED_TEMPLATE_VERSION,
            "template_semantic_hash": custom_structured_template_hash(
                template_version=CUSTOM_STRUCTURED_TEMPLATE_VERSION,
                window_days=lookback_days,
            ),
            "candidate_type": CUSTOM_STRUCTURED_CANDIDATE_TYPE,
            "parameter_policy_id": CUSTOM_STRUCTURED_PARAMETER_POLICY_ID,
            "semantic_selection_policy_id": CUSTOM_STRUCTURED_SELECTION_POLICY_ID,
            "semantic_anchor_policy_id": CUSTOM_STRUCTURED_ANCHOR_POLICY_ID,
            "observation_window_days": lookback_days,
        }
        is_source_refinement = False
    elif template_version == CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION:
        expected_static = {
            "template_version": CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION,
            "template_semantic_hash": custom_structured_template_hash(
                template_version=CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION,
                window_days=lookback_days,
            ),
            "candidate_type": CUSTOM_STRUCTURED_CANDIDATE_TYPE,
            "parameter_policy_id": CUSTOM_SOURCE_REFINEMENT_PARAMETER_POLICY_ID,
            "semantic_selection_policy_id": (
                CUSTOM_SOURCE_REFINEMENT_SELECTION_POLICY_ID
            ),
            "semantic_anchor_policy_id": CUSTOM_SOURCE_REFINEMENT_ANCHOR_POLICY_ID,
            "observation_window_days": lookback_days,
        }
        is_source_refinement = True
    else:
        raise _error(
            "segment_audience_template_binding_mismatch",
            segment_id,
            "template_version does not match a supported custom structured template",
        )
    for field_name, expected in expected_static.items():
        if raw_spec.get(field_name) != expected:
            raise _error(
                "segment_audience_template_binding_mismatch",
                segment_id,
                f"{field_name} does not match the custom structured template",
            )

    custom_conditions = _canonical_custom_conditions(
        raw_parameters.get("conditions"),
        segment_id=segment_id,
        allow_empty=is_source_refinement,
    )
    base_user_ids = (
        _canonical_source_user_ids(
            raw_parameters.get("base_user_ids"),
            segment_id=segment_id,
        )
        if template_version == CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION
        else ()
    )
    expected_keys = (
        (CUSTOM_SOURCE_MEMBERSHIP_CONDITION_KEY,)
        + ((CUSTOM_STRUCTURED_CONDITION_KEY,) if custom_conditions else ())
        if is_source_refinement
        else (CUSTOM_STRUCTURED_CONDITION_KEY,)
    )
    condition_keys = _required_text_tuple(raw_spec, "condition_keys", segment_id)
    hard_predicate_keys = _required_text_tuple(
        raw_spec,
        "hard_predicate_keys",
        segment_id,
    )
    if condition_keys != expected_keys or hard_predicate_keys != expected_keys:
        raise _error(
            "segment_audience_template_binding_mismatch",
            segment_id,
            "custom structured predicates do not match the template version",
        )
    query_signal_keys = _custom_query_signal_keys(custom_conditions)
    raw_query_signal_keys = _required_text_tuple(
        raw_spec,
        "query_signal_keys",
        segment_id,
    )
    if raw_query_signal_keys != query_signal_keys:
        raise _error(
            "segment_audience_template_binding_mismatch",
            segment_id,
            "query_signal_keys do not match the structured conditions",
        )

    destination_ids = tuple(
        sorted(
            {
                str(condition["destination"]).strip().lower()
                for condition in custom_conditions
                if condition.get("destination")
            }
        )
    )
    season_months = tuple(
        sorted(
            {
                int(month)
                for condition in custom_conditions
                for month in condition.get("checkin_months", ())
            }
        )
    )
    canonical = {
        "schema_version": schema_version,
        "template_id": CUSTOM_STRUCTURED_TEMPLATE_ID,
        **expected_static,
        "condition_keys": list(condition_keys),
        "query_signal_keys": list(query_signal_keys),
        "hard_predicate_keys": list(hard_predicate_keys),
        "parameters": {
            "lookback_days": lookback_days,
            "conditions": list(custom_conditions),
            **(
                {"base_user_ids": list(base_user_ids)}
                if base_user_ids
                else {}
            ),
        },
    }
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SegmentAudienceSpec(
        segment_id=segment_id,
        schema_version=schema_version,
        template_id=CUSTOM_STRUCTURED_TEMPLATE_ID,
        template_version=int(template_version),
        template_semantic_hash=str(expected_static["template_semantic_hash"]),
        candidate_type=CUSTOM_STRUCTURED_CANDIDATE_TYPE,
        condition_keys=condition_keys,
        query_signal_keys=query_signal_keys,
        hard_predicate_keys=hard_predicate_keys,
        destination_ids=destination_ids,
        season_months=season_months,
        benefit_keys=(),
        parameter_policy_id=str(expected_static["parameter_policy_id"]),
        semantic_selection_policy_id=str(
            expected_static["semantic_selection_policy_id"]
        ),
        semantic_anchor_policy_id=str(expected_static["semantic_anchor_policy_id"]),
        observation_window_days=lookback_days,
        spec_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        custom_conditions=custom_conditions,
        base_user_ids=base_user_ids,
    )


def _canonical_source_user_ids(
    value: Any,
    *,
    segment_id: str,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not 1 <= len(value) <= 5_000
    ):
        raise _error(
            "segment_audience_parameters_invalid",
            segment_id,
            "source refinement requires between 1 and 5000 base_user_ids",
        )
    user_ids = tuple(sorted({str(item).strip() for item in value if str(item).strip()}))
    if not user_ids or len(user_ids) != len(value) or tuple(value) != user_ids:
        raise _error(
            "segment_audience_parameters_not_canonical",
            segment_id,
            "base_user_ids must be non-empty, unique, and canonically sorted",
        )
    return user_ids


def _canonical_custom_conditions(
    value: Any,
    *,
    segment_id: str,
    allow_empty: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    minimum_items = 0 if allow_empty else 1
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not minimum_items <= len(value) <= 8
    ):
        raise _error(
            "segment_audience_parameters_invalid",
            segment_id,
            "custom structured conditions must contain between "
            f"{minimum_items} and 8 items",
        )
    conditions: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise _custom_condition_error(segment_id, index, "must be an object")
        event_name = item.get("event_name")
        if event_name not in CUSTOM_STRUCTURED_EVENT_NAMES:
            raise _custom_condition_error(segment_id, index, "uses an unsupported event")
        label = item.get("label")
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 120:
            raise _custom_condition_error(segment_id, index, "has an invalid label")
        minimum_count = item.get("minimum_count")
        maximum_count = item.get("maximum_count")
        if (
            not isinstance(minimum_count, int)
            or isinstance(minimum_count, bool)
            or not 0 <= minimum_count <= 10_000
        ):
            raise _custom_condition_error(segment_id, index, "has an invalid minimum_count")
        if maximum_count is not None and (
            not isinstance(maximum_count, int)
            or isinstance(maximum_count, bool)
            or not minimum_count <= maximum_count <= 10_000
        ):
            raise _custom_condition_error(segment_id, index, "has an invalid maximum_count")
        destination = item.get("destination")
        if destination is not None and (
            not isinstance(destination, str)
            or not 1 <= len(destination.strip()) <= 120
        ):
            raise _custom_condition_error(segment_id, index, "has an invalid destination")
        months = item.get("checkin_months", ())
        if (
            not isinstance(months, Sequence)
            or isinstance(months, (str, bytes))
            or len(months) > 12
            or any(
                not isinstance(month, int)
                or isinstance(month, bool)
                or not 1 <= month <= 12
                for month in months
            )
        ):
            raise _custom_condition_error(segment_id, index, "has invalid checkin_months")
        filters = _canonical_custom_property_filters(
            item.get("property_filters", ()),
            segment_id=segment_id,
            condition_index=index,
        )
        conditions.append(
            {
                "label": label.strip(),
                "event_name": str(event_name),
                "minimum_count": minimum_count,
                "maximum_count": maximum_count,
                "destination": destination.strip() if destination else None,
                "checkin_months": sorted(set(int(month) for month in months)),
                "property_filters": list(filters),
            }
        )
    return tuple(conditions)


def _canonical_custom_property_filters(
    value: Any,
    *,
    segment_id: str,
    condition_index: int,
) -> tuple[Mapping[str, str], ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) > 8
    ):
        raise _custom_condition_error(
            segment_id,
            condition_index,
            "has invalid property_filters",
        )
    filters: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise _custom_condition_error(
                segment_id,
                condition_index,
                "has an invalid property filter",
            )
        key = item.get("key")
        operator = item.get("operator")
        raw_value = item.get("value")
        if key not in CUSTOM_STRUCTURED_PROPERTY_KEYS:
            raise _custom_condition_error(
                segment_id,
                condition_index,
                "uses an unsupported property",
            )
        if operator not in CUSTOM_STRUCTURED_PROPERTY_OPERATORS:
            raise _custom_condition_error(
                segment_id,
                condition_index,
                "uses an unsupported property operator",
            )
        if not isinstance(raw_value, str) or not 1 <= len(raw_value.strip()) <= 200:
            raise _custom_condition_error(
                segment_id,
                condition_index,
                "has an invalid property value",
            )
        canonical_value = raw_value.strip()
        if operator in {"gte", "lte"}:
            try:
                float(canonical_value)
            except ValueError as exc:
                raise _custom_condition_error(
                    segment_id,
                    condition_index,
                    "requires a numeric property value",
                ) from exc
        if operator == "in":
            alternatives = _canonical_property_filter_values(canonical_value)
            if len(alternatives) < 2:
                raise _custom_condition_error(
                    segment_id,
                    condition_index,
                    "requires at least two property alternatives",
                )
            canonical_value = ",".join(alternatives)
        filters.append(
            {
                "key": str(key),
                "operator": str(operator),
                "value": canonical_value,
            }
        )
    return tuple(filters)


def _canonical_property_filter_values(value: str) -> tuple[str, ...]:
    normalized = value.replace("，", ",").replace("/", ",").replace("·", ",")
    normalized = normalized.replace("또는", ",").replace("혹은", ",")
    values = {
        item.strip().casefold()
        for item in normalized.split(",")
        if item.strip()
    }
    return tuple(sorted(values))


def _custom_query_signal_keys(
    conditions: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    event_signals = {
        "hotel_search": "hotel_search_intensity",
        "hotel_click": "hotel_click_intensity",
        "hotel_detail_view": "hotel_detail_view_intensity",
        "promotion_impression": "promotion_impression_intensity",
        "promotion_click": "promotion_click_intensity",
        "campaign_redirect_click": "campaign_redirect_intensity",
        "campaign_landing": "campaign_landing_intensity",
        "booking_start": "booking_start_intensity",
    }
    signals = {
        event_signals[str(condition["event_name"])]
        for condition in conditions
        if int(condition["minimum_count"]) > 0
        and str(condition["event_name"]) in event_signals
    }
    has_booking_start = any(
        condition["event_name"] == "booking_start"
        and int(condition["minimum_count"]) > 0
        for condition in conditions
    )
    has_no_booking_complete = any(
        condition["event_name"] == "booking_complete"
        and condition.get("maximum_count") == 0
        for condition in conditions
    )
    if has_booking_start and has_no_booking_complete:
        signals.add("booking_start_without_complete")
    if not signals:
        signals.add("hotel_consideration_intensity")
    return tuple(sorted(signals))


def _custom_condition_error(
    segment_id: str,
    index: int,
    reason: str,
) -> SegmentAudienceContractError:
    return _error(
        "segment_audience_parameters_invalid",
        segment_id,
        f"custom condition {index + 1} {reason}",
    )


def _required_text(
    value: Mapping[str, Any],
    field_name: str,
    segment_id: str,
) -> str:
    result = value.get(field_name)
    if not isinstance(result, str) or not result.strip():
        raise _error(
            "segment_audience_spec_invalid",
            segment_id,
            f"{field_name} must be a non-empty string",
        )
    return result.strip()


def _required_text_tuple(
    value: Mapping[str, Any],
    field_name: str,
    segment_id: str,
) -> tuple[str, ...]:
    return _raw_text_tuple(
        value.get(field_name),
        segment_id=segment_id,
        field_name=field_name,
        allow_empty=False,
    )


def _raw_text_tuple(
    value: Any,
    *,
    segment_id: str,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(
            "segment_audience_spec_invalid",
            segment_id,
            f"{field_name} must be an array of strings",
        )
    result = tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )
    if len(result) != len(value) or (not allow_empty and not result):
        raise _error(
            "segment_audience_spec_invalid",
            segment_id,
            f"{field_name} contains an invalid value",
        )
    return result


def _raw_month_tuple(value: Any, *, segment_id: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(
            "segment_audience_parameters_invalid",
            segment_id,
            "parameters.season_months must be an array",
        )
    months: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or not 1 <= item <= 12:
            raise _error(
                "segment_audience_parameters_invalid",
                segment_id,
                "season_months must contain integers between 1 and 12",
            )
        months.append(item)
    return tuple(months)


def _error(
    code: str,
    segment_id: str,
    reason: str,
) -> SegmentAudienceContractError:
    return SegmentAudienceContractError(
        code=code,
        segment_id=segment_id,
        reason=reason,
    )
