from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CONTENT_BRIEF_SCHEMA_VERSION = "content_brief.v2"
FALLBACK_GUIDANCE_SOURCE = "legacy_segment_content_hints"
LEGACY_BRIEF_SOURCE = "legacy_content_brief_v1"
DEFAULT_MESSAGE_DIRECTION = "Use the segment profile to produce clear hotel booking content."
MISSING_EVIDENCE_SECTIONS = (
    "primary_signals",
    "score_components",
)
SUPPORTED_AUDIENCE_EVIDENCE_SECTIONS = (
    "primary_signals",
    "score_components",
    "promotion_vector_basis",
    "promotion_matched_features",
)
SUPPORTED_AVAILABLE_SECTIONS = (
    "segment_snapshot",
    "promotion_context",
    "fallback_guidance",
    "source_refs",
    "audience_evidence",
    "hotel_profile",
    "operator_instruction",
    "generation_constraints",
)
DEFAULT_DO_NOT_CLAIM = (
    "확인되지 않은 할인율",
    "확인되지 않은 객실 재고",
)


@dataclass(frozen=True)
class NormalizedContentBrief:
    schema_version: str
    message_direction: str
    keywords: list[str]
    readiness: dict[str, Any]
    fallback_guidance: dict[str, Any]
    audience_evidence: dict[str, Any]
    generation_constraints: dict[str, Any]
    hotel_profile: dict[str, Any] | None
    fallback_guidance_used: bool
    raw: dict[str, Any]


def build_content_brief_v2(
    *,
    analysis_id: str,
    segment_snapshot: Mapping[str, Any],
    promotion_context: Mapping[str, Any],
    fallback_message_direction: str,
    fallback_keywords: Sequence[str],
    audience_evidence: Mapping[str, Any] | None = None,
    hotel_profile: Mapping[str, Any] | None = None,
    operator_instruction: str | None = None,
) -> dict[str, Any]:
    compact_segment_snapshot = _compact_nulls(segment_snapshot)
    compact_promotion_context = _compact_nulls(promotion_context)
    compact_audience_evidence = _supported_audience_evidence(
        _compact_empty(audience_evidence or {})
    )
    compact_hotel_profile = _json_object(hotel_profile)
    readiness = _readiness_for(compact_audience_evidence)
    available_sections = [
        "segment_snapshot",
        "promotion_context",
        "fallback_guidance",
        "source_refs",
    ]
    if compact_audience_evidence:
        available_sections.append("audience_evidence")
    if compact_hotel_profile:
        available_sections.append("hotel_profile")
    if operator_instruction:
        available_sections.append("operator_instruction")

    brief: dict[str, Any] = {
        "schema_version": CONTENT_BRIEF_SCHEMA_VERSION,
        "readiness": {
            **readiness,
            "available_sections": available_sections,
        },
        "segment_snapshot": compact_segment_snapshot,
        "promotion_context": compact_promotion_context,
        "generation_constraints": {
            "do_not_claim": list(DEFAULT_DO_NOT_CLAIM),
        },
        "fallback_guidance": {
            "message_direction": fallback_message_direction,
            "keywords": _string_list(fallback_keywords),
            "source": FALLBACK_GUIDANCE_SOURCE,
        },
        "source_refs": {
            "analysis_id": analysis_id,
            "segment_definition_id": str(
                compact_segment_snapshot.get("segment_id", "")
            ),
        },
    }
    if compact_audience_evidence:
        brief["audience_evidence"] = compact_audience_evidence
    if compact_hotel_profile:
        brief["hotel_profile"] = compact_hotel_profile
    if operator_instruction:
        brief["operator_instruction"] = operator_instruction
    return brief


def normalize_content_brief(value: Mapping[str, Any] | object) -> NormalizedContentBrief:
    raw = _json_object(value)
    if raw.get("schema_version") == CONTENT_BRIEF_SCHEMA_VERSION:
        return _normalize_v2(raw)
    return _normalize_legacy(raw)


def _normalize_v2(raw: dict[str, Any]) -> NormalizedContentBrief:
    fallback_guidance = _json_object(raw.get("fallback_guidance"))
    message_direction = _optional_text(fallback_guidance.get("message_direction"))
    keywords = _string_list(fallback_guidance.get("keywords"))
    audience_evidence = _supported_audience_evidence(
        _json_object(raw.get("audience_evidence"))
    )
    readiness = {
        **_readiness_for(audience_evidence),
        "available_sections": _available_sections_from(
            raw_readiness=_json_object(raw.get("readiness")),
            audience_evidence=audience_evidence,
        ),
    }
    return NormalizedContentBrief(
        schema_version=CONTENT_BRIEF_SCHEMA_VERSION,
        message_direction=message_direction or DEFAULT_MESSAGE_DIRECTION,
        keywords=keywords,
        readiness=readiness,
        fallback_guidance=fallback_guidance,
        audience_evidence=audience_evidence,
        generation_constraints=_json_object(raw.get("generation_constraints")),
        hotel_profile=_json_object(raw.get("hotel_profile")) or None,
        fallback_guidance_used=bool(message_direction or keywords),
        raw=dict(raw),
    )


def _normalize_legacy(raw: dict[str, Any]) -> NormalizedContentBrief:
    message_direction = _optional_text(raw.get("message_direction"))
    keywords = _string_list(raw.get("keywords"))
    fallback_guidance = {
        "message_direction": message_direction or DEFAULT_MESSAGE_DIRECTION,
        "keywords": keywords,
        "source": LEGACY_BRIEF_SOURCE,
    }
    return NormalizedContentBrief(
        schema_version="content_brief.v1",
        message_direction=fallback_guidance["message_direction"],
        keywords=keywords,
        readiness={
            "level": "fallback_only",
            "available_sections": ["fallback_guidance"],
            "missing_sections": list(MISSING_EVIDENCE_SECTIONS),
        },
        fallback_guidance=fallback_guidance,
        audience_evidence={},
        generation_constraints={},
        hotel_profile=None,
        fallback_guidance_used=True,
        raw=dict(raw),
    )


def _compact_nulls(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if item is not None}


def _readiness_for(audience_evidence: Mapping[str, Any]) -> dict[str, Any]:
    missing_sections = [
        section
        for section in MISSING_EVIDENCE_SECTIONS
        if section not in audience_evidence
    ]
    return {
        "level": "partial"
        if len(missing_sections) < len(MISSING_EVIDENCE_SECTIONS)
        else "fallback_only",
        "missing_sections": missing_sections,
    }


def _available_sections_from(
    *,
    raw_readiness: Mapping[str, Any],
    audience_evidence: Mapping[str, Any],
) -> list[str]:
    raw_sections = raw_readiness.get("available_sections")
    if isinstance(raw_sections, Sequence) and not isinstance(raw_sections, str):
        available_sections = [
            str(section)
            for section in raw_sections
            if str(section) in SUPPORTED_AVAILABLE_SECTIONS
        ]
    else:
        available_sections = ["fallback_guidance"]
    if audience_evidence and "audience_evidence" not in available_sections:
        available_sections.append("audience_evidence")
    return available_sections


def _supported_audience_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        section: value[section]
        for section in SUPPORTED_AUDIENCE_EVIDENCE_SECTIONS
        if section in value
    }


def _compact_empty(value: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, Mapping):
            nested = _compact_empty(item)
            if nested:
                compact[str(key)] = nested
            continue
        if isinstance(item, Sequence) and not isinstance(item, str):
            items = [nested_item for nested_item in item if nested_item is not None]
            if items:
                compact[str(key)] = items
            continue
        compact[str(key)] = item
    return compact


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
