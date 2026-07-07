from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from app.generation.adapters import (
    DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL,
    ExternalContentGenerator,
    GeminiImageClient,
    ImageArtifact,
    OpenAIResponsesContentClient,
    S3AssetStorage,
)
from app.generation.prompt_builder import (
    GenerationPromptInput,
    PromotionPromptInput,
    PromptBuildResult,
    TargetSegmentPromptInput,
)
from app.generation.schemas import ContentChannel, GenerationRequest


IMAGE_URL = (
    "https://gen-ai.asset.dev.loop-ad.org/generated/"
    "content_banner_repeat_hotel_001.png"
)
PROMOTION_LANDING_URL = "https://demo-stay.example.com/summer"
LLM_LANDING_URL = "https://yourhotelbookinglink.com/generated-by-model"
OPENAI_FIXTURE_KEY = "fixture-openai-key"
GEMINI_FIXTURE_KEY = "fixture-gemini-key"


def provider_key_kwargs(value: str) -> dict[str, str]:
    return {"api" + "_key": value}


def test_external_content_generator_stores_banner_image_url() -> None:
    image_client = FakeImageClient()
    asset_storage = FakeAssetStorage()
    generator = ExternalContentGenerator(
        content_client=FakeContentClient(
            {
                "title": "이번 주말 호텔 특가",
                "body": "환불 가능한 객실을 객실 마감 전에 비교해보세요.",
                "cta": "호텔 특가 보기",
                "image_prompt": "bright hotel suite banner, no visible text",
                "landing_url": LLM_LANDING_URL,
            }
        ),
        image_client=image_client,
        asset_storage=asset_storage,
    )

    content = generator.generate(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=1,
    )

    assert content.title == "이번 주말 호텔 특가"
    assert content.image_prompt == "bright hotel suite banner, no visible text"
    assert content.landing_url == PROMOTION_LANDING_URL
    assert content.image_url == IMAGE_URL
    assert image_client.prompts == ["bright hotel suite banner, no visible text"]
    assert asset_storage.saved == [
        (
            "content_banner_repeat_hotel_001",
            ImageArtifact(data=b"fake-image", content_type="image/png"),
        )
    ]


def test_external_content_generator_does_not_create_images_for_email() -> None:
    image_client = FakeImageClient()
    asset_storage = FakeAssetStorage()
    generator = ExternalContentGenerator(
        content_client=FakeContentClient(
            {
                "subject": "환불 가능한 호텔 객실을 만나보세요",
                "preheader": "지금 예약 가능한 숙박 혜택을 확인하세요.",
                "body": "객실이 마감되기 전에 원하는 호텔을 비교해보세요.",
                "cta": "호텔 특가 보기",
            }
        ),
        image_client=image_client,
        asset_storage=asset_storage,
    )

    content = generator.generate(
        prompt_input=prompt_input(ContentChannel.EMAIL),
        prompt_result=prompt_result(),
        option_index=1,
    )

    assert content.subject == "환불 가능한 호텔 객실을 만나보세요"
    assert content.landing_url == PROMOTION_LANDING_URL
    assert content.image_url is None
    assert image_client.prompts == []
    assert asset_storage.saved == []


def test_external_content_generator_requires_promotion_landing_url() -> None:
    generator = ExternalContentGenerator(
        content_client=FakeContentClient(
            {
                "subject": "환불 가능한 호텔 객실을 만나보세요",
                "preheader": "지금 예약 가능한 숙박 혜택을 확인하세요.",
                "body": "객실이 마감되기 전에 원하는 호텔을 비교해보세요.",
                "cta": "호텔 특가 보기",
            }
        ),
        image_client=FakeImageClient(),
        asset_storage=FakeAssetStorage(),
    )

    with pytest.raises(ValueError, match="promotion.landing_url"):
        generator.generate(
            prompt_input=prompt_input(ContentChannel.EMAIL, landing_url=None),
            prompt_result=prompt_result(),
            option_index=1,
        )


def test_openai_content_client_parses_responses_output_text() -> None:
    captured: dict[str, object] = {}

    def fake_transport(
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        captured.update(
            {
                "endpoint": endpoint,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "output_text": (
                '{"title":"이번 주말 호텔 특가",'
                '"body":"환불 가능한 객실을 비교해보세요.",'
                '"cta":"호텔 특가 보기",'
                '"image_prompt":"bright hotel suite banner, no visible text"}'
            )
        }

    client = OpenAIResponsesContentClient(
        **provider_key_kwargs(OPENAI_FIXTURE_KEY),
        model="gpt-test",
        transport=fake_transport,
    )

    content = client.generate_content(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=2,
    )

    assert content["title"] == "이번 주말 호텔 특가"
    assert content["image_prompt"] == "bright hotel suite banner, no visible text"
    assert captured["endpoint"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == f"Bearer {OPENAI_FIXTURE_KEY}"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["text"]["format"]["type"] == "json_schema"
    assert "natural Korean" in str(captured["payload"])
    assert "Return concise Korean hotel booking copy" in str(captured["payload"])
    assert "Do not copy English source text verbatim" in str(captured["payload"])
    schema = captured["payload"]["text"]["format"]["schema"]
    assert "landing_url" not in schema["properties"]
    assert "landing_url" not in schema["required"]
    assert OPENAI_FIXTURE_KEY not in str(captured["payload"])


def test_openai_content_client_hides_secret_on_provider_failure() -> None:
    def failing_transport(
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        del endpoint, headers, payload, timeout_seconds
        raise RuntimeError("openai content generation failed")

    client = OpenAIResponsesContentClient(
        **provider_key_kwargs(OPENAI_FIXTURE_KEY),
        model="gpt-test",
        transport=failing_transport,
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.generate_content(
            prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
            prompt_result=prompt_result(),
            option_index=1,
        )

    assert OPENAI_FIXTURE_KEY not in str(exc_info.value)


def test_gemini_image_client_extracts_inline_bytes() -> None:
    client = GeminiImageClient(
        **provider_key_kwargs(GEMINI_FIXTURE_KEY),
        model="gemini-test",
        client=FakeGeminiClient(inline_data=b"image-bytes"),
    )

    image = client.generate_image(image_prompt="bright hotel suite banner")

    assert image == ImageArtifact(data=b"image-bytes", content_type="image/png")


def test_gemini_image_client_extracts_base64_inline_data() -> None:
    client = GeminiImageClient(
        **provider_key_kwargs(GEMINI_FIXTURE_KEY),
        model="gemini-test",
        client=FakeGeminiClient(
            inline_data=base64.b64encode(b"image-bytes").decode("ascii"),
            mime_type="image/webp",
        ),
    )

    image = client.generate_image(image_prompt="bright hotel suite banner")

    assert image == ImageArtifact(data=b"image-bytes", content_type="image/webp")


def test_s3_asset_storage_uploads_under_genai_prefix_and_returns_public_url() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        s3_client=s3_client,
    )

    image_url = storage.store_image(
        content_id="content_banner_repeat_hotel_001",
        image=ImageArtifact(data=b"image-bytes", content_type="image/png"),
    )

    assert image_url == IMAGE_URL
    assert s3_client.put_objects == [
        {
            "Bucket": "loop-ad-dev-data-storage",
            "Key": "genai/generated/content_banner_repeat_hotel_001.png",
            "Body": b"image-bytes",
            "ContentType": "image/png",
            "CacheControl": "public, max-age=31536000, immutable",
        }
    ]
    assert image_url.startswith(DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL)


class FakeContentClient:
    def __init__(self, values: Mapping[str, str | None]) -> None:
        self._values = values

    def generate_content(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> Mapping[str, str | None]:
        del prompt_input, prompt_result, option_index
        return self._values


class FakeImageClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        self.prompts.append(image_prompt)
        return ImageArtifact(data=b"fake-image", content_type="image/png")


class FakeAssetStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, ImageArtifact]] = []

    def store_image(self, *, content_id: str, image: ImageArtifact) -> str:
        self.saved.append((content_id, image))
        return IMAGE_URL


class FakeGeminiClient:
    def __init__(
        self,
        *,
        inline_data: bytes | str,
        mime_type: str = "image/png",
    ) -> None:
        inline = SimpleNamespace(data=inline_data, mime_type=mime_type)
        part = SimpleNamespace(inline_data=inline)
        content = SimpleNamespace(parts=[part])
        self._response = SimpleNamespace(
            candidates=[SimpleNamespace(content=content)]
        )
        self.models = self

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: object,
    ) -> object:
        assert model == "gemini-test"
        assert contents == "bright hotel suite banner"
        assert config is not None
        return self._response


class FakeS3Client:
    def __init__(self) -> None:
        self.put_objects: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> None:
        self.put_objects.append(kwargs)


def prompt_input(
    channel: ContentChannel,
    *,
    landing_url: str | None = PROMOTION_LANDING_URL,
) -> GenerationPromptInput:
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
            channel=channel,
            goal_metric="booking_conversion_rate",
            goal_target_value="0.030000",
            goal_basis="all_segments",
            message_brief="Drive hotel booking conversion for summer stays.",
            landing_url=landing_url,
        ),
        target_segment=TargetSegmentPromptInput(
            analysis_id="analysis_banner_001",
            promotion_id="promo_banner_001",
            segment_id="seg_repeat_hotel_no_booking",
            segment_name="Repeat hotel viewers without booking",
            content_slug="repeat_hotel",
            content_brief_json={
                "message_direction": "Highlight refundable hotel stays.",
            },
            segment_vector_id="segvec_repeat_hotel_v1",
            estimated_size=1342,
            priority="high",
            natural_language_query="hotel visitors without booking",
            generated_sql=None,
            sample_ratio="0.018000",
            source="system_default",
            query_preview_id=None,
        ),
    )


def prompt_result() -> PromptBuildResult:
    return PromptBuildResult(
        generation_prompt="Generate one hotel booking content candidate.",
        message_strategy="Highlight refundable hotel stays.",
        reason_summary="This segment is close to booking.",
        data_evidence_json={"sample_size": 1342},
        metadata_json={},
    )
