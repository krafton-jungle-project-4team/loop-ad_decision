from __future__ import annotations

import base64
from dataclasses import replace

import pytest

from app.contents.assets import ContentAssetService, InMemoryAssetStorage
from app.contents.config import (
    ContentGenerationConfig,
    DEFAULT_GEMINI_IMAGE_MODEL,
    build_banner_visual_provider,
    build_content_asset_service,
)
from app.contents.visuals import (
    GeminiBannerVisualProvider,
    MockBannerVisualProvider,
    build_gemini_visual_prompt,
)
from tests.test_content_assets import make_draft


class FakeGeminiInteractions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output_image": {
                "data": base64.b64encode(b"fake-png-bytes").decode("ascii"),
                "mime_type": "image/png",
            }
        }


class FakeGeminiClient:
    def __init__(self) -> None:
        self.interactions = FakeGeminiInteractions()


def test_gemini_visual_prompt_keeps_copy_out_of_generated_image() -> None:
    draft = replace(
        make_draft(),
        title="Summer Sale Title",
        body="Use this body only as overlay copy",
        cta_label="Shop Now",
        image_prompt="summer ocean ecommerce visual with generic skincare bottles",
    )

    prompt = build_gemini_visual_prompt(draft)

    assert "summer ocean ecommerce visual" in prompt
    assert "Do not render any words" in prompt
    assert "logos" in prompt
    assert draft.title not in prompt
    assert draft.body not in prompt
    assert draft.cta_label not in prompt


def test_gemini_visual_provider_decodes_output_image_and_records_request() -> None:
    client = FakeGeminiClient()
    provider = GeminiBannerVisualProvider(
        api_key="gemini-key",
        model="gemini-image-test",
        client=client,
    )

    visual = provider.generate(make_draft())

    assert visual.body == b"fake-png-bytes"
    assert visual.content_type == "image/png"
    assert visual.provider == "gemini"
    assert visual.model == "gemini-image-test"
    assert client.interactions.calls[0]["model"] == "gemini-image-test"
    assert client.interactions.calls[0]["response_format"]["type"] == "image"


def test_gemini_visual_provider_rejects_missing_output_image() -> None:
    class EmptyInteractions:
        def create(self, **kwargs):
            del kwargs
            return {}

    class EmptyClient:
        interactions = EmptyInteractions()

    provider = GeminiBannerVisualProvider(
        api_key="gemini-key",
        model="gemini-image-test",
        client=EmptyClient(),
    )

    with pytest.raises(ValueError, match="output_image"):
        provider.generate(make_draft())


def test_content_asset_service_layers_visual_under_renderer_copy() -> None:
    storage = InMemoryAssetStorage(public_base_url="https://cdn.example.com")
    service = ContentAssetService(
        storage=storage,
        visual_provider=MockBannerVisualProvider(),
        asset_prefix="generated",
    )

    draft = service.store_banner(make_draft())
    svg = storage.objects[draft.media_s3_key].body.decode("utf-8")

    assert "data:image/svg+xml;base64," in svg
    assert draft.title in svg
    assert draft.body in svg
    assert draft.cta_label in svg
    assert draft.metadata["visual_provider"] == "mock"
    assert draft.metadata["visual_model"] == "mock-visual"
    assert draft.metadata["visual_content_type"] == "image/svg+xml"


def test_build_content_asset_service_accepts_injected_visual_provider() -> None:
    provider = MockBannerVisualProvider(model="mock-visual-test")
    service = build_content_asset_service(
        ContentGenerationConfig(
            content_asset_storage="memory",
            content_asset_public_base_url="https://cdn.example.com",
        ),
        visual_provider=provider,
    )

    assert service.visual_provider is provider


def test_build_banner_visual_provider_returns_none_without_gemini_config() -> None:
    assert build_banner_visual_provider(ContentGenerationConfig()) is None


def test_build_banner_visual_provider_uses_default_model_with_gemini_key(monkeypatch) -> None:
    class FakeGeminiBannerVisualProvider:
        def __init__(self, *, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def generate(self, draft):
            del draft
            raise AssertionError("builder test should not generate images")

    monkeypatch.setattr(
        "app.contents.config.GeminiBannerVisualProvider",
        FakeGeminiBannerVisualProvider,
    )

    provider = build_banner_visual_provider(
        ContentGenerationConfig(gemini_api_key="gemini-key"),
    )

    assert provider.api_key == "gemini-key"
    assert provider.model == DEFAULT_GEMINI_IMAGE_MODEL


def test_content_generation_config_reads_optional_gemini_visual_env(monkeypatch) -> None:
    from tests.test_content_assets import set_loopad_content_env

    set_loopad_content_env(
        monkeypatch,
        LOOPAD_GEMINI_API_KEY="gemini-key",
    )

    config = ContentGenerationConfig.from_env()

    assert config.gemini_api_key == "gemini-key"
    assert config.gemini_image_model == DEFAULT_GEMINI_IMAGE_MODEL
