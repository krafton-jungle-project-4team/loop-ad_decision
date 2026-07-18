from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from app.generation.adapters import (
    ExternalContentGenerator,
    GeminiImageClient,
    ImageArtifact,
    MAX_CONCURRENT_GEMINI_IMAGE_REQUESTS,
    OpenAIResponsesContentClient,
    S3AssetStorage,
)
from app.generation.artifacts import (
    ArtifactIdentity,
    CREATIVE_SOURCE_META_NAME,
    S3CreativeArtifactPublisher,
    StoredAsset,
    image_prompt_sha256,
    recovered_image_prompt,
    render_banner_html,
)
from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
)
from app.generation.generator import GeneratedContent
from app.generation.prompt_builder import (
    GenerationPromptInput,
    PromotionPromptInput,
    PromptBuildResult,
    PromptBuilder,
    TargetSegmentPromptInput,
)
from app.generation.schemas import ContentChannel, CreativeFormat, GenerationRequest


IMAGE_SHA256 = hashlib.sha256(b"image-bytes").hexdigest()
PUBLIC_BASE_URL = "https://gen-ai.asset.dev.loop-ad.org"
STORAGE_IMAGE_PROMPT = "hotel image prompt, no visible text"
IMAGE_PROMPT_SHA256 = image_prompt_sha256(STORAGE_IMAGE_PROMPT)
IMAGE_URL = (
    "https://gen-ai.asset.dev.loop-ad.org/hotel-client-a/promo_banner_001/"
    "generation_banner_001/content_banner_repeat_hotel_001/"
    f"image.{IMAGE_PROMPT_SHA256}.png"
)
PROMOTION_LANDING_URL = "https://demo-stay.example.com/summer"
LLM_LANDING_URL = "https://yourhotelbookinglink.com/generated-by-model"
OPENAI_FIXTURE_KEY = "fixture-openai-key"
GEMINI_FIXTURE_KEY = "fixture-gemini-key"


def _with_v2_creative_source(html_body: str) -> str:
    match = re.search(
        rf'<meta name="{re.escape(CREATIVE_SOURCE_META_NAME)}" content="([^"]+)">',
        html_body,
    )
    assert match is not None
    encoded = match.group(1)
    payload = json.loads(
        base64.urlsafe_b64decode(
            f"{encoded}{'=' * (-len(encoded) % 4)}"
        ).decode("utf-8")
    )
    payload["schema_version"] = "creative.source.v2"
    payload.pop("creative_contract_sha256")
    previous_encoded = base64.urlsafe_b64encode(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return (
        html_body[: match.start(1)]
        + previous_encoded
        + html_body[match.end(1) :]
    )


def provider_key_kwargs(value: str) -> dict[str, str]:
    return {"api" + "_key": value}


def artifact_identity(
    content_id: str = "content_banner_repeat_hotel_001",
) -> ArtifactIdentity:
    return ArtifactIdentity(
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        generation_id="generation_banner_001",
        content_id=content_id,
    )


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
        artifact_identity=artifact_identity(),
    )

    assert content.title == "이번 주말 호텔 특가"
    assert content.image_prompt is not None
    assert content.image_prompt.startswith(
        "Property-agnostic hotel booking advertisement image."
    )
    assert "no visible text" in content.image_prompt
    assert "do not render a color palette" in content.image_prompt
    assert "adult travelers aged 20 to 39" in content.image_prompt
    assert content.landing_url == PROMOTION_LANDING_URL
    assert content.image_url == IMAGE_URL
    assert image_client.prompts == [content.image_prompt]
    assert asset_storage.saved == [
        (
            "content_banner_repeat_hotel_001",
            ImageArtifact(data=b"fake-image", content_type="image/png"),
        )
    ]


def test_external_content_generator_does_not_reuse_image_from_different_prompt() -> None:
    existing = StoredAsset(
        storage_key=(
            "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
            f"content_banner_repeat_hotel_001/image.{IMAGE_PROMPT_SHA256}.png"
        ),
        public_url=IMAGE_URL,
        sha256=hashlib.sha256(b"existing-image").hexdigest(),
        bytes=len(b"existing-image"),
        content_type="image/png",
    )
    image_client = FakeImageClient()
    asset_storage = FakeAssetStorage(existing=existing)
    generator = ExternalContentGenerator(
        content_client=FakeContentClient(
            {
                "title": "이번 주말 호텔 특가",
                "body": "환불 가능한 객실을 객실 마감 전에 비교해보세요.",
                "cta": "호텔 특가 보기",
                "image_prompt": "bright hotel suite banner, no visible text",
            }
        ),
        image_client=image_client,
        asset_storage=asset_storage,
    )

    content = generator.generate(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=1,
        artifact_identity=artifact_identity(),
    )

    assert content.image_url == IMAGE_URL
    assert content.image_artifact != existing
    assert image_client.prompts == [content.image_prompt]
    assert len(asset_storage.saved) == 1


def test_external_content_generator_recovers_stored_creative_before_providers() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )
    identity = artifact_identity()
    original_image_prompt = "first hotel image prompt, no visible text"
    stored_image = storage.store_image(
        identity=identity,
        image_prompt_sha256=image_prompt_sha256(original_image_prompt),
        image=ImageArtifact(data=b"image-bytes", content_type="image/png"),
    )
    original_values = {
        "subject": None,
        "preheader": None,
        "title": "처음 생성된 호텔 특가",
        "body": "처음 생성된 객실 문구입니다.",
        "cta": "처음 호텔 보기",
        "message": None,
        "image_prompt": original_image_prompt,
        "image_url": stored_image.public_url,
        "landing_url": PROMOTION_LANDING_URL,
    }
    storage.store_html(
        identity=identity,
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=render_banner_html(original_values),
    )
    content_client = FakeContentClient(
        {
            "title": "재시도에서 달라진 문구",
            "body": "이 문구는 사용되면 안 됩니다.",
            "cta": "다른 CTA",
            "image_prompt": "different prompt",
        }
    )
    image_client = FakeImageClient()

    content = ExternalContentGenerator(
        content_client=content_client,
        image_client=image_client,
        asset_storage=storage,
    ).generate(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=1,
        artifact_identity=identity,
    )

    assert content.title == original_values["title"]
    assert content.body == original_values["body"]
    assert content.cta == original_values["cta"]
    assert content.image_prompt == recovered_image_prompt(
        image_prompt_sha256(original_image_prompt)
    )
    assert content.artifact_renderer_version == "generation.renderer.v1"
    assert content.artifact_template_version == "banner.overlay.v1"
    assert content.image_artifact == stored_image
    assert content_client.calls == 0
    assert image_client.prompts == []
    reused_artifact = S3CreativeArtifactPublisher(storage).publish(
        identity=identity,
        channel=ContentChannel.ONSITE_BANNER,
        content_values=content.to_record_values(ContentChannel.ONSITE_BANNER),
    )
    assert reused_artifact["sha256"] == hashlib.sha256(
        render_banner_html(original_values).encode("utf-8")
    ).hexdigest()
    assert len(s3_client.put_objects) == 2


def test_s3_asset_storage_rejects_tampered_creative_source() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )
    identity = artifact_identity()
    html_body = render_banner_html(
        {
            "title": "호텔 특가",
            "body": "객실을 확인하세요.",
            "cta": "호텔 보기",
            "image_prompt": "hotel image, no visible text",
            "image_url": IMAGE_URL,
            "landing_url": PROMOTION_LANDING_URL,
        }
    )
    stored = storage.store_html(
        identity=identity,
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=html_body,
    )
    object_key = ("loop-ad-dev-data-storage", stored.storage_key)
    s3_client.objects[object_key]["Body"] = html_body.replace(
        "객실을 확인하세요.",
        "변조된 문구입니다.",
        1,
    ).encode("utf-8")

    with pytest.raises(PermanentGenerationError) as exc_info:
        storage.find_creative_content(
            identity=identity,
            channel=ContentChannel.ONSITE_BANNER,
        )

    assert exc_info.value.code == "artifact_source_invalid"


def test_external_content_generator_can_defer_banner_image_generation() -> None:
    image_client = FakeImageClient()
    asset_storage = FakeAssetStorage()
    generator = ExternalContentGenerator(
        content_client=FakeContentClient(
            {
                "title": "이번 주말 호텔 특가",
                "body": "환불 가능한 객실을 객실 마감 전에 비교해보세요.",
                "cta": "호텔 특가 보기",
                "image_prompt": "bright hotel suite banner, no visible text",
            }
        ),
        image_client=image_client,
        asset_storage=asset_storage,
        generate_images=False,
    )

    content = generator.generate(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=1,
        artifact_identity=artifact_identity(),
    )

    assert content.title == "이번 주말 호텔 특가"
    assert content.image_prompt is not None
    assert content.image_prompt.startswith(
        "Property-agnostic hotel booking advertisement image."
    )
    assert content.image_url is None
    assert image_client.prompts == []
    assert asset_storage.saved == []


def test_external_content_generator_fills_missing_banner_image_prompt() -> None:
    image_client = FakeImageClient()
    asset_storage = FakeAssetStorage()
    generator = ExternalContentGenerator(
        content_client=FakeContentClient(
            {
                "title": "Hotel rooms ready this weekend",
                "body": "Compare refundable hotel stays before rooms run out.",
                "cta": "View hotel deals",
                "image_prompt": None,
            }
        ),
        image_client=image_client,
        asset_storage=asset_storage,
    )

    content = generator.generate(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=1,
        artifact_identity=artifact_identity(),
    )

    assert content.image_prompt is not None
    assert content.image_prompt.startswith(
        "Property-agnostic hotel booking advertisement image."
    )
    assert content.image_url == IMAGE_URL
    assert image_client.prompts == [content.image_prompt]
    assert len(asset_storage.saved) == 1


def test_external_content_generator_creates_images_for_email_contract() -> None:
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
        artifact_identity=artifact_identity("content_email_repeat_hotel_001"),
    )

    assert content.subject == "환불 가능한 호텔 객실을 만나보세요"
    assert content.landing_url == PROMOTION_LANDING_URL
    assert content.image_prompt is not None
    assert content.image_url == IMAGE_URL
    assert image_client.prompts == [content.image_prompt]
    assert "do not render a color palette" in image_client.prompts[0]
    assert "adult travelers aged 20 to 39" in image_client.prompts[0]
    assert asset_storage.saved == [
        (
            "content_email_repeat_hotel_001",
            ImageArtifact(data=b"fake-image", content_type="image/png"),
        )
    ]


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
            artifact_identity=artifact_identity("content_email_repeat_hotel_001"),
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

    rich_prompt_input = prompt_input(ContentChannel.ONSITE_BANNER)
    rich_prompt_result = PromptBuilder().build(rich_prompt_input)
    content = client.generate_content(
        prompt_input=rich_prompt_input,
        prompt_result=rich_prompt_result,
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
    assert "must not request visible text in the image" in str(captured["payload"])
    assert "Candidate strategy (apply only to this content option)" in str(
        captured["payload"]
    )
    assert rich_prompt_result.metadata_json["strategy_key"] in str(
        captured["payload"]
    )
    schema = captured["payload"]["text"]["format"]["schema"]
    assert "landing_url" not in schema["properties"]
    assert "landing_url" not in schema["required"]
    assert schema["properties"]["title"] == {"type": "string"}
    assert schema["properties"]["body"] == {"type": "string"}
    assert schema["properties"]["cta"] == {"type": "string"}
    assert schema["properties"]["image_prompt"] == {"type": "string"}
    assert schema["properties"]["subject"] == {"type": ["string", "null"]}
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


def test_gemini_image_client_limits_process_wide_parallel_requests() -> None:
    probe = BlockingGeminiClient()
    client = GeminiImageClient(
        **provider_key_kwargs(GEMINI_FIXTURE_KEY),
        model="gemini-test",
        client=probe,
    )
    request_count = MAX_CONCURRENT_GEMINI_IMAGE_REQUESTS * 2

    with ThreadPoolExecutor(max_workers=request_count) as executor:
        futures = [
            executor.submit(
                client.generate_image,
                image_prompt="bright hotel suite banner",
            )
            for _ in range(request_count)
        ]
        try:
            assert probe.limit_reached.wait(timeout=5)
            with probe.lock:
                assert (
                    probe.active_requests == MAX_CONCURRENT_GEMINI_IMAGE_REQUESTS
                )
                assert (
                    probe.max_active_requests
                    == MAX_CONCURRENT_GEMINI_IMAGE_REQUESTS
                )
        finally:
            probe.release.set()
        images = [future.result(timeout=5) for future in futures]

    assert len(images) == request_count
    assert all(
        image == ImageArtifact(data=b"image-bytes", content_type="image/png")
        for image in images
    )
    assert probe.max_active_requests == MAX_CONCURRENT_GEMINI_IMAGE_REQUESTS


def test_gemini_image_client_uses_developer_api_supported_config() -> None:
    from google import genai
    from google.genai import types

    captured: dict[str, object] = {}
    sdk_client = genai.Client(api_key=GEMINI_FIXTURE_KEY)

    def request(
        http_method: str,
        path: str,
        request_dict: dict[str, object],
        http_options: object = None,
    ) -> types.HttpResponse:
        captured.update(
            http_method=http_method,
            path=path,
            request_dict=request_dict,
            http_options=http_options,
        )
        return types.HttpResponse(
            headers={},
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": base64.b64encode(
                                                b"image-bytes"
                                            ).decode("ascii"),
                                        }
                                    }
                                ],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                            "index": 0,
                        }
                    ]
                }
            ),
        )

    sdk_client._api_client.request = request
    try:
        image = GeminiImageClient(
            **provider_key_kwargs(GEMINI_FIXTURE_KEY),
            model="gemini-test",
            client=sdk_client,
        ).generate_image(image_prompt="bright hotel suite banner")
    finally:
        sdk_client.close()

    request_dict = captured["request_dict"]
    assert isinstance(request_dict, dict)
    assert not sdk_client._api_client.vertexai
    assert request_dict["generationConfig"] == {
        "responseModalities": ["IMAGE"],
    }
    assert "outputMimeType" not in json.dumps(request_dict)
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


@pytest.mark.parametrize("inline_data", [b"", "", "not-valid-base64!"])
def test_gemini_image_client_rejects_empty_or_invalid_image_data(
    inline_data: bytes | str,
) -> None:
    client = GeminiImageClient(
        **provider_key_kwargs(GEMINI_FIXTURE_KEY),
        model="gemini-test",
        client=FakeGeminiClient(inline_data=inline_data),
    )

    with pytest.raises(ValueError):
        client.generate_image(image_prompt="bright hotel suite banner")


def test_s3_asset_storage_uploads_under_genai_prefix_and_returns_public_url() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )

    stored_image = storage.store_image(
        identity=artifact_identity(),
        image_prompt_sha256=IMAGE_PROMPT_SHA256,
        image=ImageArtifact(data=b"image-bytes", content_type="image/png"),
    )

    assert stored_image.public_url == IMAGE_URL
    assert stored_image.sha256 == IMAGE_SHA256
    assert s3_client.put_objects == [
        {
            "Bucket": "loop-ad-dev-data-storage",
            "Key": (
                "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
                f"content_banner_repeat_hotel_001/image.{IMAGE_PROMPT_SHA256}.png"
            ),
            "Body": b"image-bytes",
            "ContentType": "image/png",
            "CacheControl": "public, max-age=31536000, immutable",
            "Metadata": {"sha256": IMAGE_SHA256},
            "IfNoneMatch": "*",
        }
    ]
    assert stored_image.public_url.startswith(PUBLIC_BASE_URL)


def test_s3_asset_storage_reuses_images_and_rejects_changed_html() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )

    first_image = storage.store_image(
        identity=artifact_identity(),
        image_prompt_sha256=image_prompt_sha256("first prompt"),
        image=ImageArtifact(data=b"first-image", content_type="image/png"),
    )
    repeated_image = storage.store_image(
        identity=artifact_identity(),
        image_prompt_sha256=image_prompt_sha256("first prompt"),
        image=ImageArtifact(data=b"first-image", content_type="image/png"),
    )
    assert repeated_image == first_image
    assert len(s3_client.put_objects) == 1

    independently_stored_image = storage.store_image(
        identity=artifact_identity(),
        image_prompt_sha256=image_prompt_sha256("second prompt"),
        image=ImageArtifact(data=b"second-image", content_type="image/png"),
    )
    assert independently_stored_image != first_image
    assert storage.find_image(
        identity=artifact_identity(),
        image_prompt_sha256=image_prompt_sha256("first prompt"),
    ) == first_image
    assert storage.find_image(
        identity=artifact_identity(),
        image_prompt_sha256=image_prompt_sha256("second prompt"),
    ) == independently_stored_image
    assert len(s3_client.put_objects) == 2

    html_body = "<html><body>first</body></html>"
    html_metadata = storage.store_html(
        identity=artifact_identity(),
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=html_body,
    )
    repeated_html_metadata = storage.store_html(
        identity=artifact_identity(),
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=html_body,
    )
    html_sha256 = hashlib.sha256(html_body.encode("utf-8")).hexdigest()
    assert html_metadata.storage_key == (
        "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
        "content_banner_repeat_hotel_001/creative.banner.html"
    )
    assert html_metadata.sha256 == html_sha256
    assert repeated_html_metadata == html_metadata
    assert len(s3_client.put_objects) == 3

    with pytest.raises(RetryableGenerationError, match="immutable") as exc_info:
        storage.store_html(
            identity=artifact_identity(),
            creative_format=CreativeFormat.BANNER_HTML,
            html_body="<html><body>second</body></html>",
        )
    assert exc_info.value.code == "artifact_hash_conflict"
    assert len(s3_client.put_objects) == 3


def test_s3_asset_storage_reuses_same_source_across_renderer_changes() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )
    values = {
        "title": "호텔 특가",
        "body": "객실을 확인하세요.",
        "cta": "호텔 보기",
        "image_prompt": "hotel image, no visible text",
        "image_url": IMAGE_URL,
        "renderer_version": "generation.renderer.old",
        "template_version": "banner.overlay.old",
    }
    first_html = render_banner_html(values)
    changed_template_html = render_banner_html(
        {
            **values,
            "renderer_version": "generation.renderer.new",
            "template_version": "banner.overlay.new",
        }
    ).replace(
        "background:#072b63", "background:#082c64", 1
    )

    first = storage.store_html(
        identity=artifact_identity(),
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=first_html,
    )
    reused = storage.store_html(
        identity=artifact_identity(),
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=changed_template_html,
    )

    assert reused == first
    assert reused.renderer_version == "generation.renderer.old"
    assert reused.template_version == "banner.overlay.old"
    assert len(s3_client.put_objects) == 1

    publication = S3CreativeArtifactPublisher(storage).publish(
        identity=artifact_identity(),
        channel=ContentChannel.ONSITE_BANNER,
        content_values={
            **values,
            "renderer_version": "generation.renderer.new",
            "template_version": "banner.overlay.new",
        },
    )

    assert publication.renderer == {
        "version": "generation.renderer.old",
        "template_version": "banner.overlay.old",
    }
    assert len(s3_client.put_objects) == 1


def test_s3_asset_storage_rejects_changed_creative_contract() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )
    values = {
        "title": "호텔 특가",
        "body": "객실을 확인하세요.",
        "cta": "호텔 보기",
        "image_prompt": "hotel image, no visible text",
        "image_url": IMAGE_URL,
        "variant_type": "editorial",
        "link_targets": [
            {"placeholder": "{{redirect_url}}", "target_type": "promotion"}
        ],
    }
    storage.store_html(
        identity=artifact_identity(),
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=render_banner_html(values),
    )

    with pytest.raises(RetryableGenerationError) as exc_info:
        storage.store_html(
            identity=artifact_identity(),
            creative_format=CreativeFormat.BANNER_HTML,
            html_body=render_banner_html(
                {
                    **values,
                    "variant_type": "comparison",
                }
            ),
        )

    assert exc_info.value.code == "artifact_hash_conflict"
    assert len(s3_client.put_objects) == 1


def test_s3_asset_storage_reuses_v2_source_for_empty_creative_contract() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )
    values = {
        "title": "호텔 특가",
        "body": "객실을 확인하세요.",
        "cta": "호텔 보기",
        "image_prompt": "hotel image, no visible text",
        "image_url": IMAGE_URL,
    }
    v2_html = _with_v2_creative_source(render_banner_html(values))
    first = storage.store_html(
        identity=artifact_identity(),
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=v2_html,
    )

    reused = storage.store_html(
        identity=artifact_identity(),
        creative_format=CreativeFormat.BANNER_HTML,
        html_body=render_banner_html(values),
    )

    assert reused == first
    assert len(s3_client.put_objects) == 1


def test_s3_conditional_write_conflict_is_retryable() -> None:
    s3_client = FakeS3Client(conditional_conflicts=1)
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        s3_client=s3_client,
    )

    with pytest.raises(RetryableGenerationError) as exc_info:
        storage.store_image(
            identity=artifact_identity(),
            image_prompt_sha256=image_prompt_sha256("first prompt"),
            image=ImageArtifact(data=b"first-image", content_type="image/png"),
        )

    assert exc_info.value.code == "artifact_write_conflict"
    assert s3_client.put_objects == []


def test_external_generator_reuses_private_source_and_image_after_crash() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        source_manifest_prefix="genai-source/",
        s3_client=s3_client,
    )
    first_content_client = FakeContentClient(
        {
            "title": "최초 생성 제목",
            "body": "최초 생성 본문",
            "cta": "호텔 보기",
            "image_prompt": "first canonical hotel image, no visible text",
        }
    )
    first_image_client = FakeImageClient()
    first_generator = ExternalContentGenerator(
        content_client=first_content_client,
        image_client=first_image_client,
        asset_storage=storage,
        source_manifest_storage=storage,
    )

    first = first_generator.generate(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=1,
        artifact_identity=artifact_identity(),
    )

    retry_content_client = FakeContentClient(
        {
            "title": "재시도에서 생성되면 안 되는 제목",
            "body": "재시도에서 생성되면 안 되는 본문",
            "cta": "다시 생성",
            "image_prompt": "different retry image prompt",
        }
    )
    retry_image_client = FakeImageClient()
    retry_generator = ExternalContentGenerator(
        content_client=retry_content_client,
        image_client=retry_image_client,
        asset_storage=storage,
        source_manifest_storage=storage,
    )

    restored = retry_generator.generate(
        prompt_input=prompt_input(ContentChannel.ONSITE_BANNER),
        prompt_result=prompt_result(),
        option_index=1,
        artifact_identity=artifact_identity(),
    )

    assert first_content_client.calls == 1
    assert len(first_image_client.prompts) == 1
    assert retry_content_client.calls == 0
    assert retry_image_client.prompts == []
    assert restored.title == first.title
    assert restored.body == first.body
    assert restored.image_prompt == first.image_prompt
    assert restored.image_url == first.image_url
    assert len(s3_client.put_objects) == 2
    assert str(s3_client.put_objects[0]["Key"]).startswith("genai-source/")
    assert s3_client.put_objects[0]["CacheControl"] == "no-store"
    assert str(s3_client.put_objects[1]["Key"]).startswith("genai/")


def test_source_manifest_precondition_uses_first_canonical_source() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        source_manifest_prefix="genai-source/",
        s3_client=s3_client,
    )
    first = GeneratedContent(
        title="최초 제목",
        body="최초 본문",
        cta="호텔 보기",
        image_prompt="first prompt, no visible text",
        landing_url=PROMOTION_LANDING_URL,
    )
    second = replace(first, title="경합에서 폐기할 제목")

    stored = storage.store_source_manifest(
        identity=artifact_identity(),
        channel=ContentChannel.ONSITE_BANNER,
        request_fingerprint="a" * 64,
        content=first,
    )
    reused = storage.store_source_manifest(
        identity=artifact_identity(),
        channel=ContentChannel.ONSITE_BANNER,
        request_fingerprint="a" * 64,
        content=second,
    )

    assert stored.title == "최초 제목"
    assert reused == stored
    assert len(s3_client.put_objects) == 1


def test_source_manifest_rejects_different_request_fingerprint() -> None:
    s3_client = FakeS3Client()
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        source_manifest_prefix="genai-source/",
        s3_client=s3_client,
    )
    storage.store_source_manifest(
        identity=artifact_identity(),
        channel=ContentChannel.ONSITE_BANNER,
        request_fingerprint="a" * 64,
        content=GeneratedContent(
            title="최초 제목",
            body="최초 본문",
            cta="호텔 보기",
            image_prompt="first prompt, no visible text",
            landing_url=PROMOTION_LANDING_URL,
        ),
    )

    with pytest.raises(PermanentGenerationError) as exc_info:
        storage.find_source_manifest(
            identity=artifact_identity(),
            channel=ContentChannel.ONSITE_BANNER,
            request_fingerprint="b" * 64,
        )

    assert exc_info.value.code == "source_manifest_identity_mismatch"


def test_source_manifest_conditional_conflict_is_retryable() -> None:
    storage = S3AssetStorage(
        bucket_name="loop-ad-dev-data-storage",
        base_prefix="genai/",
        public_base_url=PUBLIC_BASE_URL,
        source_manifest_prefix="genai-source/",
        s3_client=FakeS3Client(conditional_conflicts=1),
    )

    with pytest.raises(RetryableGenerationError) as exc_info:
        storage.store_source_manifest(
            identity=artifact_identity(),
            channel=ContentChannel.ONSITE_BANNER,
            request_fingerprint="a" * 64,
            content=GeneratedContent(
                title="최초 제목",
                body="최초 본문",
                cta="호텔 보기",
                image_prompt="first prompt, no visible text",
                landing_url=PROMOTION_LANDING_URL,
            ),
        )

    assert exc_info.value.code == "source_manifest_write_conflict"


def test_s3_storage_rejects_source_manifest_inside_public_prefix() -> None:
    with pytest.raises(ValueError, match="outside the public"):
        S3AssetStorage(
            bucket_name="loop-ad-dev-data-storage",
            base_prefix="genai/",
            public_base_url=PUBLIC_BASE_URL,
            source_manifest_prefix="genai/source/",
            s3_client=FakeS3Client(),
        )


class FakeContentClient:
    def __init__(self, values: Mapping[str, str | None]) -> None:
        self._values = values
        self.calls = 0

    def generate_content(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> Mapping[str, str | None]:
        del prompt_input, prompt_result, option_index
        self.calls += 1
        return self._values


class FakeImageClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        self.prompts.append(image_prompt)
        return ImageArtifact(data=b"fake-image", content_type="image/png")


class FakeAssetStorage:
    def __init__(self, *, existing: StoredAsset | None = None) -> None:
        self.existing = existing
        self.saved: list[tuple[str, ImageArtifact]] = []

    def find_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        public_url: str | None = None,
    ) -> StoredAsset | None:
        del identity, public_url
        if image_prompt_sha256 != IMAGE_PROMPT_SHA256:
            return None
        return self.existing

    def store_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        image: ImageArtifact,
    ) -> StoredAsset:
        del image_prompt_sha256
        self.saved.append((identity.content_id, image))
        return StoredAsset(
            storage_key=(
                "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
                f"{identity.content_id}/image.png"
            ),
            public_url=IMAGE_URL,
            sha256=hashlib.sha256(image.data).hexdigest(),
            bytes=len(image.data),
            content_type=image.content_type,
        )


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
        assert config.response_modalities == ["IMAGE"]
        assert config.image_config is None
        return self._response


class BlockingGeminiClient:
    def __init__(self) -> None:
        inline = SimpleNamespace(data=b"image-bytes", mime_type="image/png")
        part = SimpleNamespace(inline_data=inline)
        content = SimpleNamespace(parts=[part])
        self._response = SimpleNamespace(
            candidates=[SimpleNamespace(content=content)]
        )
        self.models = self
        self.lock = threading.Lock()
        self.active_requests = 0
        self.max_active_requests = 0
        self.limit_reached = threading.Event()
        self.release = threading.Event()

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
        with self.lock:
            self.active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests,
                self.active_requests,
            )
            if self.active_requests == MAX_CONCURRENT_GEMINI_IMAGE_REQUESTS:
                self.limit_reached.set()
        try:
            self.release.wait(timeout=5)
            return self._response
        finally:
            with self.lock:
                self.active_requests -= 1


class FakeS3Client:
    def __init__(self, *, conditional_conflicts: int = 0) -> None:
        self.conditional_conflicts = conditional_conflicts
        self.put_objects: list[dict[str, object]] = []
        self.objects: dict[tuple[object, object], dict[str, object]] = {}

    def put_object(self, **kwargs: object) -> None:
        if self.conditional_conflicts > 0:
            self.conditional_conflicts -= 1
            raise FakeS3ConditionalConflict()
        key = (kwargs["Bucket"], kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise FakeS3PreconditionFailed()
        self.put_objects.append(kwargs)
        body = bytes(kwargs["Body"])
        self.objects[key] = {
            "Body": body,
            "ContentLength": len(body),
            "ContentType": kwargs["ContentType"],
            "Metadata": dict(kwargs.get("Metadata") or {}),
        }

    def head_object(self, **kwargs: object) -> dict[str, object]:
        key = (kwargs["Bucket"], kwargs["Key"])
        if key not in self.objects:
            raise FakeS3NotFound()
        return self.objects[key]

    def get_object(self, **kwargs: object) -> dict[str, object]:
        key = (kwargs["Bucket"], kwargs["Key"])
        if key not in self.objects:
            raise FakeS3NotFound()
        return self.objects[key]


class FakeS3NotFound(RuntimeError):
    response = {
        "ResponseMetadata": {"HTTPStatusCode": 404},
        "Error": {"Code": "NoSuchKey"},
    }


class FakeS3PreconditionFailed(RuntimeError):
    response = {
        "ResponseMetadata": {"HTTPStatusCode": 412},
        "Error": {"Code": "PreconditionFailed"},
    }


class FakeS3ConditionalConflict(RuntimeError):
    response = {
        "ResponseMetadata": {"HTTPStatusCode": 409},
        "Error": {"Code": "ConditionalRequestConflict"},
    }


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
