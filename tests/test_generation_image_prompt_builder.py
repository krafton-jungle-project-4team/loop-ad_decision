from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from app.generation.image_prompt_builder import (
    MAX_RICH_IMAGE_PROMPT_LENGTH,
    RichImagePromptBuilder,
)
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
    assert "color palette" in image_prompt
    assert "color swatches" in image_prompt
    assert "contact sheet" in image_prompt
    assert "mood board" in image_prompt
    assert "collage" in image_prompt
    assert "clearly adult travelers aged 20 to 39" in image_prompt
    assert "never children, teenagers, middle-aged people, or elderly people" in (
        image_prompt
    )


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


def test_rich_image_prompt_hides_raw_brand_colors_and_selected_asset_id() -> None:
    prompt_result = PromptBuilder().build(
        prompt_input(
            hotel_profile={
                "visual_context": {
                    "scene": "sunlit coastal hotel exterior",
                    "style": ["bright", "calm"],
                }
            }
        )
    )
    assert prompt_result.generation_context is not None
    assert prompt_result.strategy_plan is not None
    selected_asset_id = "brand_asset_private_jeju_hero"
    brand_context = SimpleNamespace(
        guardrails=SimpleNamespace(
            approved_colours=("#0F55C8", "#FFFFFF"),
            approved_image_styles=("editorial photography",),
        ),
        documents=(
            SimpleNamespace(
                source_kind="brand_asset",
                source_id=selected_asset_id,
                document_text="Calm coastal sunlight.",
            ),
        ),
        selected_asset_id=selected_asset_id,
    )
    prompt_result = replace(
        prompt_result,
        generation_context=replace(
            prompt_result.generation_context,
            brand_context=brand_context,
        ),
        strategy_plan=replace(
            prompt_result.strategy_plan,
            audience_focus=(),
            visual_direction=(),
        ),
    )

    image_prompt = RichImagePromptBuilder().build(
        prompt_result,
        provider_visual_concept="Editorial composition using #0F55C8 accents.",
    )

    assert "#0F55C8" not in image_prompt
    assert "#FFFFFF" not in image_prompt
    assert selected_asset_id not in image_prompt
    assert "selected_asset_id" not in image_prompt
    assert "Approved brand visual direction" in image_prompt
    assert "editorial photography" in image_prompt


def test_required_image_guardrails_survive_max_length_compaction() -> None:
    prompt_result = PromptBuilder().build(
        prompt_input(
            hotel_profile={
                "visual_context": {
                    "scene": "sunlit coastal hotel travel scene " * 200,
                    "style": ["premium editorial photography"] * 100,
                }
            }
        )
    )

    image_prompt = RichImagePromptBuilder().build(
        prompt_result,
        provider_visual_concept="cinematic summer travel composition " * 200,
    )

    assert len(image_prompt) == MAX_RICH_IMAGE_PROMPT_LENGTH == 1200
    assert "color palette" in image_prompt
    assert "color swatches" in image_prompt
    assert "contact sheet" in image_prompt
    assert "mood board" in image_prompt
    assert "collage" in image_prompt
    assert "clearly adult travelers aged 20 to 39" in image_prompt
    assert "never children, teenagers, middle-aged people, or elderly people" in (
        image_prompt
    )
    assert image_prompt.endswith(
        "generation constraints=확인되지 않은 할인율, 확인되지 않은 객실 재고."
    )


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
