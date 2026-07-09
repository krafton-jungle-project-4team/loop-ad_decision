from __future__ import annotations

from typing import Any

import pytest

from app.generation.prompt_builder import (
    GenerationInputBuilder,
    GenerationPromptInput,
    PromotionPromptInput,
    PromptBuilder,
    TargetSegmentPromptInput,
)
from app.generation.schemas import ContentChannel, GenerationRequest


FORBIDDEN_PUBLIC_TERMS = (
    "creative_id",
    "variant_id",
    "experiment_id",
    "recommendation",
)


def generation_request(
    *,
    operator_instruction: str | None = "Make the message direct and concise.",
) -> GenerationRequest:
    return GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=3,
        operator_instruction=operator_instruction,
    )


def promotion_input(
    *,
    channel: ContentChannel = ContentChannel.ONSITE_BANNER,
) -> PromotionPromptInput:
    return PromotionPromptInput(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel=channel,
        goal_metric="booking_conversion_rate",
        goal_target_value="0.030000",
        goal_basis="all_segments",
        message_brief="Drive hotel booking conversion for summer stays.",
        landing_url="https://demo-stay.example.com/summer",
    )


def target_segment_input() -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id="analysis_banner_001",
        promotion_id="promo_banner_001",
        segment_id="seg_repeat_hotel_no_booking",
        segment_name="Repeat hotel viewers without booking",
        content_slug="repeat_hotel",
        content_brief_json={
            "message_direction": (
                "Emphasize refundable rooms, same-weekend availability, "
                "and a clear hotel deals CTA."
            ),
            "keywords": ["refundable rooms", "same-weekend availability"],
        },
        segment_vector_id="segvec_repeat_hotel_v1",
        estimated_size=1342,
        priority="high",
        natural_language_query="repeat hotel viewers who did not book",
        generated_sql="SELECT user_id FROM hotel_detail_events",
        sample_ratio="0.018000",
        query_preview_id="seg_query_preview_001",
        status="approved",
    )


def test_generation_input_builder_builds_prompt_inputs_from_dependencies() -> None:
    builder = GenerationInputBuilder()

    prompt_inputs = builder.build(
        request=generation_request(),
        promotion=promotion_input(),
        target_segments=[target_segment_input()],
    )

    assert prompt_inputs == [
        GenerationPromptInput(
            request=generation_request(),
            promotion=promotion_input(),
            target_segment=target_segment_input(),
        )
    ]
    assert prompt_inputs[0].target_segment.content_brief_json["keywords"] == [
        "refundable rooms",
        "same-weekend availability",
    ]


def test_generation_input_builder_rejects_mismatched_dependencies() -> None:
    builder = GenerationInputBuilder()
    wrong_promotion = PromotionPromptInput(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_other_001",
        channel=ContentChannel.ONSITE_BANNER,
        goal_metric="booking_conversion_rate",
        goal_target_value="0.030000",
        goal_basis="all_segments",
        message_brief=None,
        landing_url="https://demo-stay.example.com/summer",
    )

    with pytest.raises(ValueError, match="promotion_id"):
        builder.build(
            request=generation_request(),
            promotion=wrong_promotion,
            target_segments=[target_segment_input()],
        )


@pytest.mark.parametrize(
    ("channel", "required_terms"),
    (
        (
            ContentChannel.EMAIL,
            ("subject", "preheader", "body", "cta"),
        ),
        (ContentChannel.SMS, ("message",)),
        (
            ContentChannel.ONSITE_BANNER,
            ("title", "body", "cta", "image_prompt"),
        ),
    ),
)
def test_prompt_builder_includes_channel_contract_and_operator_instruction(
    channel: ContentChannel,
    required_terms: tuple[str, ...],
) -> None:
    builder = PromptBuilder()
    prompt_input = GenerationPromptInput(
        request=generation_request(
            operator_instruction="Use a calm premium hotel tone.",
        ),
        promotion=promotion_input(channel=channel),
        target_segment=target_segment_input(),
    )

    result = builder.build(prompt_input)

    for required_term in required_terms:
        assert required_term in result.generation_prompt
    required_fields_line = next(
        line
        for line in result.generation_prompt.splitlines()
        if line.startswith("Required output fields:")
    )
    assert "landing_url" not in required_fields_line
    assert "Fixed landing URL: https://demo-stay.example.com/summer" in (
        result.generation_prompt
    )
    assert "Do not generate or override landing_url" in result.generation_prompt
    assert "Output language: Korean (ko-KR)." in result.generation_prompt
    assert "customer-facing copy fields in natural Korean" in (
        result.generation_prompt
    )
    assert "do not copy English source text verbatim" in result.generation_prompt
    assert "Use a calm premium hotel tone." in result.generation_prompt
    assert "booking_conversion_rate" in result.generation_prompt
    assert "Repeat hotel viewers without booking" in result.generation_prompt
    assert result.message_strategy
    assert result.reason_summary
    assert result.data_evidence_json["segment_id"] == "seg_repeat_hotel_no_booking"
    assert result.data_evidence_json["target_segment_status"] == "approved"
    assert result.metadata_json["source_query_preview_id"] == "seg_query_preview_001"


def test_prompt_builder_output_does_not_use_legacy_public_terms() -> None:
    result = PromptBuilder().build(
        GenerationPromptInput(
            request=generation_request(),
            promotion=promotion_input(),
            target_segment=target_segment_input(),
        )
    )

    output_text = " ".join(_collect_strings(result))
    for term in FORBIDDEN_PUBLIC_TERMS:
        assert term not in output_text


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        strings: list[str] = []
        for key, child in value.items():
            strings.append(str(key))
            strings.extend(_collect_strings(child))
        return strings
    if isinstance(value, list):
        strings = []
        for child in value:
            strings.extend(_collect_strings(child))
        return strings
    if isinstance(value, tuple):
        strings = []
        for child in value:
            strings.extend(_collect_strings(child))
        return strings
    if isinstance(value, str):
        return [value]
    return []
