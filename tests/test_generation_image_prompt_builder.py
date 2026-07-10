from __future__ import annotations

from app.generation.image_prompt_builder import RichImagePromptBuilder
from app.generation.prompt_builder import (
    GenerationPromptInput,
    PromotionPromptInput,
    PromptBuilder,
    TargetSegmentPromptInput,
)
from app.generation.schemas import ContentChannel, GenerationRequest


def test_rich_image_prompt_uses_verified_context_and_guardrails() -> None:
    prompt_result = PromptBuilder().build(
        prompt_input(
            hotel_profile={
                "visual_context": {
                    "scene": "coastal hotel exterior",
                    "style": ["bright", "calm"],
                }
            }
        )
    )

    image_prompt = RichImagePromptBuilder().build(
        prompt_result,
        provider_visual_concept="editorial travel composition",
    )

    assert "goal_metric=booking_conversion_rate" in image_prompt
    assert "Audience focus: near_checkin" in image_prompt
    assert "coastal hotel exterior" in image_prompt
    assert "editorial travel composition" in image_prompt
    assert "generic hotel booking travel scene" in image_prompt
    assert "확인되지 않은 할인율" in image_prompt
    assert "no visible text" in image_prompt
    assert "do not depict discounts, prices, room inventory" in image_prompt


def test_rich_image_prompt_ignores_provider_facilities_without_visual_context() -> None:
    prompt_result = PromptBuilder().build(
        prompt_input(
            hotel_profile={"event_count": 5000, "booking_count": 120}
        )
    )

    image_prompt = RichImagePromptBuilder().build(
        prompt_result,
        provider_visual_concept="luxury rooftop pool and spa",
    )

    assert "Verified hotel visual context: none" in image_prompt
    assert "property-agnostic" in image_prompt
    assert "rooftop pool" not in image_prompt
    assert "spa" not in image_prompt
    assert "no visible text" in image_prompt


def prompt_input(*, hotel_profile: dict[str, object]) -> GenerationPromptInput:
    return GenerationPromptInput(
        request=GenerationRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            analysis_id="analysis_banner_001",
            content_option_count=1,
            operator_instruction=None,
        ),
        promotion=PromotionPromptInput(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            channel=ContentChannel.ONSITE_BANNER,
            goal_metric="booking_conversion_rate",
            goal_target_value="0.03",
            goal_basis="all_segments",
            message_brief="Drive hotel booking conversion.",
            landing_url="https://demo-stay.example.com/summer",
        ),
        target_segment=TargetSegmentPromptInput(
            analysis_id="analysis_banner_001",
            promotion_id="promo_banner_001",
            segment_id="seg_near_checkin",
            segment_name="Near check-in hotel audience",
            content_brief_json={
                "schema_version": "content_brief.v2",
                "hotel_profile": hotel_profile,
                "generation_constraints": {
                    "do_not_claim": [
                        "확인되지 않은 할인율",
                        "확인되지 않은 객실 재고",
                    ]
                },
                "audience_evidence": {
                    "primary_signals": ["near_checkin"],
                    "score_components": {"final_score": 0.91},
                },
            },
            segment_vector_id="segvec_near_checkin_v1",
            estimated_size=1200,
            priority="high",
        ),
    )
