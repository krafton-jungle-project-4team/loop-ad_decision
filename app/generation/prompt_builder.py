from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.generation.schemas import ContentChannel, GenerationRequest


PROMPT_BUILDER_VERSION = "dec-c2.v1"


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


@dataclass(frozen=True)
class GenerationPromptInput:
    request: GenerationRequest
    promotion: PromotionPromptInput
    target_segment: TargetSegmentPromptInput


@dataclass(frozen=True)
class PromptBuildResult:
    generation_prompt: str
    message_strategy: str
    reason_summary: str
    data_evidence_json: dict[str, Any] = field(default_factory=dict)
    metadata_json: dict[str, Any] = field(default_factory=dict)


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
    def build(self, prompt_input: GenerationPromptInput) -> PromptBuildResult:
        promotion = prompt_input.promotion
        target_segment = prompt_input.target_segment
        channel_contract = _channel_contract(promotion.channel)
        message_strategy = _message_strategy(prompt_input)
        reason_summary = _reason_summary(prompt_input)
        evidence = _data_evidence(prompt_input)

        generation_prompt = "\n".join(
            [
                "Generate one Loop-Ad hotel marketing content candidate.",
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
                f"Segment message direction: {_message_direction(target_segment)}",
                f"Segment keywords: {', '.join(_keywords(target_segment)) or 'not provided'}",
                f"Message strategy: {message_strategy}",
                _optional_line(
                    "Natural language segment query",
                    target_segment.natural_language_query,
                ),
                _optional_line("Generated SQL summary", target_segment.generated_sql),
                _optional_line(
                    "Operator instruction",
                    prompt_input.request.operator_instruction,
                ),
                "Return only fields that belong to the requested channel contract.",
                "Do not generate or override landing_url; Loop-Ad assigns the fixed landing URL.",
                "Keep the content in the hotel booking domain.",
            ]
        )

        metadata = {
            "prompt_builder_version": PROMPT_BUILDER_VERSION,
            "reason_summary": reason_summary,
            "data_evidence": evidence,
            "message_strategy": message_strategy,
            "operator_instruction": prompt_input.request.operator_instruction,
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


def _message_strategy(prompt_input: GenerationPromptInput) -> str:
    target_segment = prompt_input.target_segment
    direction = _message_direction(target_segment)
    keywords = _keywords(target_segment)
    if keywords:
        return f"{direction} Use these hotel audience cues: {', '.join(keywords)}."
    return direction


def _reason_summary(prompt_input: GenerationPromptInput) -> str:
    promotion = prompt_input.promotion
    target_segment = prompt_input.target_segment
    return (
        f"Uses {target_segment.segment_name} from analysis "
        f"{prompt_input.request.analysis_id} for {promotion.goal_metric}."
    )


def _data_evidence(prompt_input: GenerationPromptInput) -> dict[str, Any]:
    promotion = prompt_input.promotion
    target_segment = prompt_input.target_segment
    return {
        "analysis_id": prompt_input.request.analysis_id,
        "promotion_id": promotion.promotion_id,
        "segment_id": target_segment.segment_id,
        "segment_name": target_segment.segment_name,
        "segment_vector_id": target_segment.segment_vector_id,
        "estimated_size": target_segment.estimated_size,
        "priority": target_segment.priority,
        "sample_ratio": target_segment.sample_ratio,
        "goal_metric": promotion.goal_metric,
        "goal_basis": promotion.goal_basis,
        "goal_target_value": promotion.goal_target_value,
        "content_brief_keywords": _keywords(target_segment),
    }


def _message_direction(target_segment: TargetSegmentPromptInput) -> str:
    value = target_segment.content_brief_json.get("message_direction")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "Use the segment profile to produce clear hotel booking content."


def _keywords(target_segment: TargetSegmentPromptInput) -> list[str]:
    value = target_segment.content_brief_json.get("keywords")
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_line(label: str, value: str | None) -> str:
    if not value:
        return f"{label}: not provided"
    return f"{label}: {value}"
