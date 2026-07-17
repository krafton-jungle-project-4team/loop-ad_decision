from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

    @property
    def predicate_parameters(self) -> Mapping[str, Sequence[str] | Sequence[int]]:
        return {
            "destinations": self.destination_ids,
            "season_months": self.season_months,
            "benefit_keys": self.benefit_keys,
        }


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
