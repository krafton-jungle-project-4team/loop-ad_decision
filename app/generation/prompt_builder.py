from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.content_brief import NormalizedContentBrief, normalize_content_brief
from app.generation.evidence import EvidenceResolver, verified_hotel_benefits
from app.generation.schemas import ContentChannel, GenerationRequest


PROMPT_BUILDER_VERSION = "dec-c2.v4"
CANDIDATE_STRATEGY_BLOCK_HEADER = (
    "Candidate strategy (apply only to this content option):"
)
SAFE_VISUAL_DIRECTIONS = (
    "generic hotel booking travel scene",
    "traveler reviewing an accommodation booking on a mobile device",
    "neutral accommodation planning composition",
)


@dataclass(frozen=True)
class PromotionPromptInput:
    project_id: str
    campaign_id: str
    promotion_id: str
    channel: ContentChannel
    goal_metric: str
    goal_target_value: str
    goal_basis: str
    message_brief: str | None
    landing_url: str | None


@dataclass(frozen=True)
class TargetSegmentPromptInput:
    analysis_id: str
    promotion_id: str
    segment_id: str
    segment_name: str
    content_brief_json: Mapping[str, Any]
    segment_vector_id: str | None
    estimated_size: int
    priority: str | None
    content_slug: str | None = None
    natural_language_query: str | None = None
    generated_sql: str | None = None
    sample_ratio: str | None = None
    source: str | None = None
    query_preview_id: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class GenerationPromptInput:
    request: GenerationRequest
    promotion: PromotionPromptInput
    target_segment: TargetSegmentPromptInput


@dataclass(frozen=True)
class GenerationContext:
    promotion_objective: Mapping[str, Any]
    verified_audience_evidence: Mapping[str, Any]
    hotel_profile: Mapping[str, Any] | None
    readiness: Mapping[str, Any]
    missing_sections: tuple[str, ...]
    generation_constraints: Mapping[str, Any]
    operator_instruction: str | None
    fallback_guidance: Mapping[str, Any]
    fallback_guidance_present: bool
    content_brief_schema_version: str
    brief_fingerprint: str
    normalized_content_brief: NormalizedContentBrief = field(
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class GenerationStrategyPlan:
    strategy_key: str
    audience_focus: tuple[str, ...]
    message_angle: str
    benefit_focus: tuple[str, ...]
    visual_direction: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "strategy_key": self.strategy_key,
            "audience_focus": list(self.audience_focus),
            "message_angle": self.message_angle,
            "benefit_focus": list(self.benefit_focus),
            "visual_direction": list(self.visual_direction),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class PromptBuildResult:
    generation_prompt: str
    message_strategy: str
    reason_summary: str
    data_evidence_json: dict[str, Any] = field(default_factory=dict)
    metadata_json: dict[str, Any] = field(default_factory=dict)
    fallback_guidance_present: bool = False
    fallback_guidance_used: bool = False
    generation_context: GenerationContext | None = None
    strategy_plan: GenerationStrategyPlan | None = None


class GenerationContextBuilder:
    def build(self, prompt_input: GenerationPromptInput) -> GenerationContext:
        content_brief = normalize_content_brief(
            prompt_input.target_segment.content_brief_json
        )
        missing_sections = tuple(
            _string_list(content_brief.readiness.get("missing_sections"))
        )
        operator_instruction = _resolved_operator_instruction(
            request_value=prompt_input.request.operator_instruction,
            brief_value=content_brief.operator_instruction,
        )
        promotion = prompt_input.promotion
        promotion_objective = {
            "channel": promotion.channel.value,
            "goal_metric": promotion.goal_metric,
            "goal_basis": promotion.goal_basis,
            "goal_target_value": promotion.goal_target_value,
            "message_brief": promotion.message_brief,
        }
        fingerprint_payload = {
            "promotion_objective": promotion_objective,
            "verified_audience_evidence": content_brief.audience_evidence,
            "hotel_profile": content_brief.hotel_profile,
            "readiness": content_brief.readiness,
            "missing_sections": list(missing_sections),
            "generation_constraints": content_brief.generation_constraints,
            "operator_instruction": operator_instruction,
            "fallback_guidance": content_brief.fallback_guidance,
            "fallback_guidance_present": content_brief.fallback_guidance_present,
            "content_brief_schema_version": content_brief.schema_version,
        }
        return GenerationContext(
            promotion_objective=promotion_objective,
            verified_audience_evidence=dict(content_brief.audience_evidence),
            hotel_profile=(
                dict(content_brief.hotel_profile)
                if content_brief.hotel_profile is not None
                else None
            ),
            readiness=dict(content_brief.readiness),
            missing_sections=missing_sections,
            generation_constraints=dict(content_brief.generation_constraints),
            operator_instruction=operator_instruction,
            fallback_guidance=dict(content_brief.fallback_guidance),
            fallback_guidance_present=content_brief.fallback_guidance_present,
            content_brief_schema_version=content_brief.schema_version,
            brief_fingerprint=_brief_fingerprint(fingerprint_payload),
            normalized_content_brief=content_brief,
        )


class GenerationStrategyPlanner:
    def build(
        self,
        generation_context: GenerationContext,
        *,
        option_index: int,
    ) -> GenerationStrategyPlan:
        if option_index < 1:
            raise ValueError("option_index must be at least 1")

        evidence = generation_context.verified_audience_evidence
        resolver = EvidenceResolver(
            audience_evidence=evidence,
            hotel_profile=generation_context.hotel_profile,
        )
        audience_items = _indexed_evidence_items(
            evidence.get("primary_signals"),
            section="primary_signals",
        )
        matched_feature_items = _indexed_evidence_items(
            evidence.get("promotion_matched_features"),
            section="promotion_matched_features",
        )
        selected_audience = _select_indexed_evidence(
            audience_items,
            option_index=option_index,
        )
        selected_matched_feature = _select_indexed_evidence(
            matched_feature_items,
            option_index=option_index,
        )
        selected_benefit = _select_indexed_evidence(
            verified_hotel_benefits(resolver),
            option_index=option_index,
        )

        audience_focus = [
            selected[0]
            for selected in (selected_audience, selected_matched_feature)
            if selected is not None
        ]
        audience_focus = list(dict.fromkeys(audience_focus))
        benefit_focus = [selected_benefit[0]] if selected_benefit else []
        evidence_refs = [
            selected[1]
            for selected in (
                selected_audience,
                selected_matched_feature,
                selected_benefit,
            )
            if selected is not None
        ]
        resolver.validate_all(evidence_refs)

        message_angle = _message_angle_for(
            str(generation_context.promotion_objective.get("goal_metric", ""))
        )
        strategy_focus = (
            benefit_focus[0]
            if benefit_focus
            else audience_focus[0]
            if audience_focus
            else "promotion"
        )
        visual_direction = SAFE_VISUAL_DIRECTIONS[
            (option_index - 1) % len(SAFE_VISUAL_DIRECTIONS)
        ]
        return GenerationStrategyPlan(
            strategy_key=(
                f"{message_angle}__{_strategy_slug(strategy_focus)}"
            ),
            audience_focus=tuple(audience_focus),
            message_angle=message_angle,
            benefit_focus=tuple(benefit_focus),
            visual_direction=(visual_direction,),
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        )


class GenerationInputBuilder:
    def build(
        self,
        *,
        request: GenerationRequest,
        promotion: PromotionPromptInput,
        target_segments: Sequence[TargetSegmentPromptInput],
    ) -> list[GenerationPromptInput]:
        _validate_request_promotion_match(request=request, promotion=promotion)
        if not target_segments:
            raise ValueError("target_segments must contain at least one segment")

        prompt_inputs: list[GenerationPromptInput] = []
        for target_segment in target_segments:
            _validate_request_target_segment_match(
                request=request,
                target_segment=target_segment,
            )
            prompt_inputs.append(
                GenerationPromptInput(
                    request=request,
                    promotion=promotion,
                    target_segment=target_segment,
                )
            )
        return prompt_inputs


class PromptBuilder:
    def build(
        self,
        prompt_input: GenerationPromptInput,
        *,
        generation_context: GenerationContext | None = None,
        strategy_plan: GenerationStrategyPlan | None = None,
    ) -> PromptBuildResult:
        promotion = prompt_input.promotion
        target_segment = prompt_input.target_segment
        channel_contract = _channel_contract(promotion.channel)
        generation_context = generation_context or GenerationContextBuilder().build(
            prompt_input
        )
        strategy_plan = strategy_plan or GenerationStrategyPlanner().build(
            generation_context,
            option_index=1,
        )
        content_brief = generation_context.normalized_content_brief
        fallback_guidance_used = _should_use_fallback_guidance(
            content_brief=content_brief,
            strategy_plan=strategy_plan,
        )
        message_strategy = _message_strategy(
            content_brief,
            fallback_guidance_used=fallback_guidance_used,
        )
        reason_summary = _reason_summary(prompt_input)
        evidence = _data_evidence(
            prompt_input,
            content_brief,
            fallback_guidance_used=fallback_guidance_used,
        )

        prompt_lines = [
            "Generate one Loop-Ad hotel marketing content candidate.",
            "Output language: Korean (ko-KR).",
            (
                "Write customer-facing copy fields in natural Korean; "
                "keep JSON field names in English."
            ),
            (
                "Use source segment details as context, but do not copy "
                "English source text verbatim into customer-facing copy."
            ),
            f"Channel: {promotion.channel.value}",
            f"Required output fields: {', '.join(channel_contract)}.",
            f"Project: {promotion.project_id}",
            f"Campaign: {promotion.campaign_id}",
            f"Promotion: {promotion.promotion_id}",
            f"Analysis: {prompt_input.request.analysis_id}",
            f"Target segment: {target_segment.segment_name} ({target_segment.segment_id})",
            f"Estimated segment size: {target_segment.estimated_size}",
            f"Goal: {promotion.goal_metric} {promotion.goal_basis} {promotion.goal_target_value}",
            f"Fixed landing URL: {promotion.landing_url or 'not provided'}",
            f"Promotion brief: {promotion.message_brief or 'not provided'}",
            f"Content brief readiness: {_readiness_level(content_brief)}",
            (
                "Missing evidence sections: "
                f"{', '.join(generation_context.missing_sections) or 'none'}"
            ),
            f"Message strategy: {message_strategy}",
            _optional_line(
                "Natural language segment query",
                target_segment.natural_language_query,
            ),
            _optional_line("Generated SQL summary", target_segment.generated_sql),
            _optional_line(
                "Operator instruction",
                generation_context.operator_instruction,
            ),
            "Return only fields that belong to the requested channel contract.",
            "Do not generate or override landing_url; Loop-Ad assigns the fixed landing URL.",
            "Keep the content in the hotel booking domain.",
        ]
        if fallback_guidance_used:
            prompt_lines.extend(
                _fallback_guardrail_lines(
                    readiness_level=_readiness_level(content_brief),
                )
            )
            prompt_lines.extend(
                [
                    (
                        "Fallback message direction: "
                        f"{content_brief.message_direction}"
                    ),
                    (
                        "Fallback keywords: "
                        f"{', '.join(content_brief.keywords) or 'not provided'}"
                    ),
                ]
            )
        prompt_lines.extend(_content_brief_context_lines(content_brief))
        prompt_lines.extend(_strategy_block_lines(strategy_plan))
        generation_prompt = "\n".join(prompt_lines)
        strategy_metadata = strategy_plan.to_metadata()

        metadata = {
            "prompt_builder_version": PROMPT_BUILDER_VERSION,
            "reason_summary": reason_summary,
            "data_evidence": evidence,
            "message_strategy": message_strategy,
            "content_brief_schema_version": content_brief.schema_version,
            "content_brief_readiness": content_brief.readiness,
            "missing_sections": list(generation_context.missing_sections),
            "fallback_guidance_present": content_brief.fallback_guidance_present,
            "fallback_guidance_used": fallback_guidance_used,
            "operator_instruction": generation_context.operator_instruction,
            "strategy_key": strategy_plan.strategy_key,
            "strategy_plan": strategy_metadata,
            "brief_fingerprint": generation_context.brief_fingerprint,
            "evidence_refs": list(strategy_plan.evidence_refs),
            "source_segment_definition_id": target_segment.segment_id,
            "source_query_preview_id": target_segment.query_preview_id,
            "generated_sql_summary": target_segment.generated_sql,
        }

        return PromptBuildResult(
            generation_prompt=generation_prompt,
            message_strategy=message_strategy,
            reason_summary=reason_summary,
            data_evidence_json=evidence,
            metadata_json=metadata,
            fallback_guidance_present=content_brief.fallback_guidance_present,
            fallback_guidance_used=fallback_guidance_used,
            generation_context=generation_context,
            strategy_plan=strategy_plan,
        )


def _validate_request_promotion_match(
    *,
    request: GenerationRequest,
    promotion: PromotionPromptInput,
) -> None:
    if request.project_id != promotion.project_id:
        raise ValueError("project_id must match promotion input")
    if request.campaign_id != promotion.campaign_id:
        raise ValueError("campaign_id must match promotion input")
    if request.promotion_id != promotion.promotion_id:
        raise ValueError("promotion_id must match promotion input")


def _validate_request_target_segment_match(
    *,
    request: GenerationRequest,
    target_segment: TargetSegmentPromptInput,
) -> None:
    if request.analysis_id != target_segment.analysis_id:
        raise ValueError("analysis_id must match target segment input")
    if request.promotion_id != target_segment.promotion_id:
        raise ValueError("promotion_id must match target segment input")


def _channel_contract(channel: ContentChannel) -> tuple[str, ...]:
    if channel == ContentChannel.EMAIL:
        return ("subject", "preheader", "body", "cta")
    if channel == ContentChannel.SMS:
        return ("message",)
    return ("title", "body", "cta", "image_prompt")


def _message_strategy(
    content_brief: NormalizedContentBrief,
    *,
    fallback_guidance_used: bool,
) -> str:
    if fallback_guidance_used and content_brief.keywords:
        return (
            f"{content_brief.message_direction} "
            f"Use these fallback hotel audience cues: "
            f"{', '.join(content_brief.keywords)}."
        )
    if fallback_guidance_used:
        return content_brief.message_direction
    if _readiness_level(content_brief) == "evidence_ready":
        return "Use verified audience evidence and promotion context."
    return (
        "Use the available verified evidence and promotion context; "
        "do not infer missing audience details."
    )


def _should_use_fallback_guidance(
    *,
    content_brief: NormalizedContentBrief,
    strategy_plan: GenerationStrategyPlan,
) -> bool:
    if not content_brief.fallback_guidance_present:
        return False
    readiness_level = _readiness_level(content_brief)
    if readiness_level == "evidence_ready":
        return False
    if readiness_level == "fallback_only":
        return True
    if readiness_level != "partial":
        return False
    return not any(
        reference.startswith(("primary_signals[", "promotion_matched_features["))
        for reference in strategy_plan.evidence_refs
    )


def _fallback_guardrail_lines(*, readiness_level: str) -> list[str]:
    if readiness_level == "fallback_only":
        return [
            (
                "Fallback basis: fallback_only. Treat fallback guidance as "
                "non-evidentiary message direction only."
            ),
            (
                "Do not infer audience traits, hotel policies, hotel benefits, "
                "discounts, room inventory, amenities, or facilities."
            ),
        ]
    return [
        (
            "Partial fallback scope: use fallback only for the missing audience "
            "or message direction; do not treat it as evidence."
        )
    ]


def _reason_summary(prompt_input: GenerationPromptInput) -> str:
    promotion = prompt_input.promotion
    target_segment = prompt_input.target_segment
    return (
        f"Uses {target_segment.segment_name} from analysis "
        f"{prompt_input.request.analysis_id} for {promotion.goal_metric}."
    )


def _data_evidence(
    prompt_input: GenerationPromptInput,
    content_brief: NormalizedContentBrief,
    *,
    fallback_guidance_used: bool,
) -> dict[str, Any]:
    promotion = prompt_input.promotion
    target_segment = prompt_input.target_segment
    goal_target_value = _optional_float(promotion.goal_target_value)
    evidence: dict[str, Any] = {
        "analysis_id": prompt_input.request.analysis_id,
        "promotion_id": promotion.promotion_id,
        "segment_id": target_segment.segment_id,
        "segment_name": target_segment.segment_name,
        "segment_vector_id": target_segment.segment_vector_id,
        "sample_size": target_segment.estimated_size,
        "priority": target_segment.priority,
        "target_segment_status": target_segment.status,
        "sample_ratio": _optional_float(target_segment.sample_ratio),
        "goal_metric": promotion.goal_metric,
        "goal_basis": promotion.goal_basis,
        "goal_target_value": (
            goal_target_value
            if goal_target_value is not None
            else promotion.goal_target_value
        ),
        "content_brief_schema_version": content_brief.schema_version,
        "content_brief_readiness": content_brief.readiness,
        "fallback_guidance_present": content_brief.fallback_guidance_present,
        "fallback_guidance_used": fallback_guidance_used,
    }
    if fallback_guidance_used:
        evidence["content_brief_keywords"] = content_brief.keywords
    if content_brief.schema_version != "content_brief.v2":
        raw_brief = content_brief.raw
        top_common_features = _string_list(raw_brief.get("top_common_features"))
        evidence.update(
            {
                "booking_conversion_rate": _optional_float(
                    raw_brief.get("booking_conversion_rate")
                ),
                "comparison_group_conversion_rate": _optional_float(
                    raw_brief.get("comparison_group_conversion_rate")
                ),
                "top_common_features": top_common_features
                or content_brief.keywords,
            }
        )
    if content_brief.audience_evidence:
        evidence["audience_evidence"] = content_brief.audience_evidence
    return evidence


def _readiness_level(content_brief: NormalizedContentBrief) -> str:
    value = content_brief.readiness.get("level")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "fallback_only"


def _content_brief_context_lines(
    content_brief: NormalizedContentBrief,
) -> list[str]:
    lines: list[str] = []
    if content_brief.audience_evidence:
        lines.append(f"Audience evidence: {_compact_jsonish(content_brief.audience_evidence)}")
    if content_brief.hotel_profile:
        lines.append(
            f"Hotel profile context: {_compact_jsonish(content_brief.hotel_profile)}"
        )
    do_not_claim = content_brief.generation_constraints.get("do_not_claim")
    if isinstance(do_not_claim, Sequence) and not isinstance(do_not_claim, str):
        claims = [str(item).strip() for item in do_not_claim if str(item).strip()]
        if claims:
            lines.append(f"Do not claim: {', '.join(claims)}")
    return lines


def _strategy_block_lines(
    strategy_plan: GenerationStrategyPlan,
) -> list[str]:
    return [
        CANDIDATE_STRATEGY_BLOCK_HEADER,
        (
            "Strategy plan JSON: "
            + json.dumps(
                strategy_plan.to_metadata(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        (
            "Use this strategy to shape this option's copy and image_prompt. "
            "Evidence references are provenance only; do not expose them in "
            "customer-facing copy."
        ),
    ]


def _brief_fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _resolved_operator_instruction(
    *,
    request_value: str | None,
    brief_value: str | None,
) -> str | None:
    for value in (request_value, brief_value):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _message_angle_for(goal_metric: str) -> str:
    normalized = goal_metric.strip().lower()
    if normalized == "booking_conversion_rate":
        return "booking_confidence"
    if normalized == "inflow_rate":
        return "landing_motivation"
    return "promotion_relevance"


def _indexed_evidence_items(
    value: object,
    *,
    section: str,
) -> list[tuple[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    items: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        text = _evidence_text(item)
        if text:
            items.append((text, f"{section}[{index}]"))
    return items


def _select_indexed_evidence(
    items: Sequence[tuple[str, str]],
    *,
    option_index: int,
) -> tuple[str, str] | None:
    if not items:
        return None
    return items[(option_index - 1) % len(items)]


def _evidence_text(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("key", "feature", "name", "label", "chip"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return None
    if isinstance(value, Sequence) and not isinstance(value, str):
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strategy_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if slug:
        return slug
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"evidence_{digest}"


def _compact_jsonish(value: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, item in value.items():
        if isinstance(item, Mapping):
            if item:
                parts.append(f"{key}={dict(item)}")
            continue
        if isinstance(item, Sequence) and not isinstance(item, str):
            items = [str(nested_item) for nested_item in item if str(nested_item)]
            if items:
                parts.append(f"{key}={items}")
            continue
        if item is not None:
            parts.append(f"{key}={item}")
    return "; ".join(parts)


def _optional_line(label: str, value: str | None) -> str:
    if not value:
        return f"{label}: not provided"
    return f"{label}: {value}"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
