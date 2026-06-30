from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.contents.assets import (
    AssetObject,
    ContentAssetService,
    InMemoryAssetStorage,
    LocalAssetStorage,
    S3AssetStorage,
    SvgBannerRenderer,
    build_asset_key,
)
from app.contents.config import ContentGenerationConfig, build_content_asset_service
from app.contents.generators import MockContentGenerator
from tests.test_content_generation_service import make_target


class FakeS3Client:
    def __init__(self) -> None:
        self.put_object_calls: list[dict[str, object]] = []
        self.presigned_url_calls: list[dict[str, object]] = []

    def put_object(self, **kwargs):
        self.put_object_calls.append(kwargs)
        return {"ETag": "fake-etag"}

    def generate_presigned_url(self, **kwargs):
        self.presigned_url_calls.append(kwargs)
        raise AssertionError("presigned URLs must not be used for generated content assets")


def make_draft():
    return MockContentGenerator().generate(
        target=make_target(),
        variant_key="control",
    )


def test_build_asset_key_is_stable_and_sanitized() -> None:
    assert build_asset_key(
        project_id="demo shop",
        recommendation_action_id=10,
        variant_key="treatment/a",
        prefix="/generated-contents/",
    ) == "generated-contents/projects/demo-shop/actions/10/variants/treatment-a/banner.svg"


def test_in_memory_asset_storage_stores_object_and_public_url() -> None:
    storage = InMemoryAssetStorage(public_base_url="https://cdn.example.com/assets/")
    stored = storage.put_object(
        AssetObject(
            key="generated-contents/banner.svg",
            body=b"<svg />",
            content_type="image/svg+xml",
        )
    )

    assert stored.key == "generated-contents/banner.svg"
    assert stored.public_url == "https://cdn.example.com/assets/generated-contents/banner.svg"
    assert storage.objects["generated-contents/banner.svg"].body == b"<svg />"


def test_local_asset_storage_writes_file_without_requiring_s3(tmp_path: Path) -> None:
    storage = LocalAssetStorage(
        root_dir=tmp_path,
        public_base_url="https://cdn.example.com",
    )
    stored = storage.put_object(
        AssetObject(
            key="generated-contents/projects/demo/actions/10/banner.svg",
            body=b"<svg />",
            content_type="image/svg+xml",
        )
    )

    assert (tmp_path / stored.key).read_bytes() == b"<svg />"
    assert stored.public_url == (
        "https://cdn.example.com/generated-contents/projects/demo/actions/10/banner.svg"
    )


def test_local_asset_storage_rejects_path_traversal(tmp_path: Path) -> None:
    storage = LocalAssetStorage(root_dir=tmp_path)

    with pytest.raises(ValueError):
        storage.put_object(
            AssetObject(
                key="../escape.svg",
                body=b"<svg />",
                content_type="image/svg+xml",
            )
        )


def test_s3_asset_storage_puts_object_without_presigned_url() -> None:
    client = FakeS3Client()
    storage = S3AssetStorage(
        bucket="loop-assets",
        public_base_url="https://cdn.example.com/assets",
        client=client,
        cache_control="public, max-age=60",
    )

    stored = storage.put_object(
        AssetObject(
            key="generated-contents/projects/demo/actions/10/banner.svg",
            body=b"<svg />",
            content_type="image/svg+xml",
        )
    )

    assert client.put_object_calls == [
        {
            "Bucket": "loop-assets",
            "Key": "generated-contents/projects/demo/actions/10/banner.svg",
            "Body": b"<svg />",
            "ContentType": "image/svg+xml",
            "CacheControl": "public, max-age=60",
        }
    ]
    assert client.presigned_url_calls == []
    assert stored.key == "generated-contents/projects/demo/actions/10/banner.svg"
    assert stored.public_url == (
        "https://cdn.example.com/assets/generated-contents/projects/demo/actions/10/banner.svg"
    )


def test_s3_asset_storage_rejects_path_traversal() -> None:
    storage = S3AssetStorage(
        bucket="loop-assets",
        public_base_url="https://cdn.example.com",
        client=FakeS3Client(),
    )

    with pytest.raises(ValueError):
        storage.put_object(
            AssetObject(
                key="../escape.svg",
                body=b"<svg />",
                content_type="image/svg+xml",
            )
        )


def test_s3_asset_storage_requires_bucket_and_public_base_url() -> None:
    with pytest.raises(ValueError):
        S3AssetStorage(
            bucket=" ",
            public_base_url="https://cdn.example.com",
            client=FakeS3Client(),
        )

    with pytest.raises(ValueError):
        S3AssetStorage(
            bucket="loop-assets",
            public_base_url=" ",
            client=FakeS3Client(),
        )


def test_content_asset_service_fills_draft_asset_fields() -> None:
    storage = InMemoryAssetStorage(public_base_url="https://cdn.example.com")
    service = ContentAssetService(storage=storage, asset_prefix="generated")

    draft = service.store_banner(make_draft())

    assert draft.media_s3_key == "generated/projects/demo-shop/actions/10/variants/control/banner.svg"
    assert draft.image_url == (
        "https://cdn.example.com/generated/projects/demo-shop/actions/10/variants/control/banner.svg"
    )
    assert draft.metadata["asset_key"] == draft.media_s3_key
    assert draft.metadata["asset_content_type"] == "image/svg+xml"
    assert storage.objects[draft.media_s3_key].body.startswith(b"<svg")


def test_content_asset_service_preserves_created_run_id() -> None:
    storage = InMemoryAssetStorage(public_base_url="https://cdn.example.com")
    service = ContentAssetService(storage=storage)

    draft = service.store_banner(replace(make_draft(), created_run_id=123))

    assert draft.created_run_id == 123


def test_svg_banner_renderer_escapes_copy() -> None:
    draft = make_draft()
    unsafe_draft = type(draft)(
        project_id=draft.project_id,
        recommendation_action_id=draft.recommendation_action_id,
        segment_id=draft.segment_id,
        variant_key=draft.variant_key,
        content_type=draft.content_type,
        title="<sale>",
        body="A & B",
        cta_label='"click"',
        landing_url=draft.landing_url,
        image_prompt=draft.image_prompt,
        generation_model=draft.generation_model,
        generation_status=draft.generation_status,
        metadata=draft.metadata,
    )

    svg = SvgBannerRenderer().render(unsafe_draft).decode("utf-8")

    assert "&lt;sale&gt;" in svg
    assert "A &amp; B" in svg
    assert "&quot;click&quot;" in svg


def test_build_content_asset_service_supports_memory_and_local_without_s3(tmp_path: Path) -> None:
    memory_service = build_content_asset_service(
        ContentGenerationConfig(
            content_asset_storage="memory",
            content_asset_prefix="memory-prefix",
            content_asset_public_base_url="https://cdn.example.com",
        )
    )
    local_service = build_content_asset_service(
        ContentGenerationConfig(
            content_asset_storage="local",
            content_asset_local_dir=str(tmp_path),
            content_asset_prefix="local-prefix",
        )
    )

    memory_draft = memory_service.store_banner(make_draft())
    local_draft = local_service.store_banner(make_draft())

    assert memory_draft.image_url is not None
    assert local_draft.image_url is None
    assert (tmp_path / local_draft.media_s3_key).exists()


def test_build_content_asset_service_defaults_to_memory_outside_production() -> None:
    service = build_content_asset_service(ContentGenerationConfig())

    draft = service.store_banner(make_draft())

    assert draft.media_s3_key is not None
    assert draft.metadata["asset_storage"] == "InMemoryAssetStorage"


def test_build_content_asset_service_requires_storage_in_production() -> None:
    with pytest.raises(ValueError, match="CONTENT_ASSET_STORAGE is required"):
        build_content_asset_service(ContentGenerationConfig(app_env="production"))


def test_build_content_asset_service_rejects_memory_in_production() -> None:
    with pytest.raises(ValueError, match="memory is not allowed"):
        build_content_asset_service(
            ContentGenerationConfig(
                app_env="production",
                content_asset_storage="memory",
            )
        )


def test_build_content_asset_service_treats_either_env_as_production(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("CONTENT_ASSET_STORAGE", raising=False)

    with pytest.raises(ValueError, match="CONTENT_ASSET_STORAGE is required"):
        build_content_asset_service()


def test_build_content_asset_service_supports_s3_with_injected_client() -> None:
    client = FakeS3Client()
    service = build_content_asset_service(
        ContentGenerationConfig(
            content_asset_storage="s3",
            content_asset_prefix="s3-prefix",
            content_asset_public_base_url="https://cdn.example.com",
            content_asset_s3_bucket="loop-assets",
            content_asset_s3_region="ap-northeast-2",
            content_asset_s3_endpoint_url="https://s3.ap-northeast-2.amazonaws.com",
            content_asset_s3_cache_control="public, max-age=60",
        ),
        s3_client=client,
    )

    draft = service.store_banner(make_draft())

    assert draft.media_s3_key == "s3-prefix/projects/demo-shop/actions/10/variants/control/banner.svg"
    assert draft.image_url == (
        "https://cdn.example.com/s3-prefix/projects/demo-shop/actions/10/variants/control/banner.svg"
    )
    assert client.put_object_calls[0]["Bucket"] == "loop-assets"
    assert client.put_object_calls[0]["Key"] == draft.media_s3_key
    assert client.put_object_calls[0]["ContentType"] == "image/svg+xml"
    assert client.put_object_calls[0]["CacheControl"] == "public, max-age=60"
    assert client.presigned_url_calls == []


def test_build_content_asset_service_requires_s3_bucket() -> None:
    with pytest.raises(ValueError, match="CONTENT_ASSET_S3_BUCKET is required"):
        build_content_asset_service(
            ContentGenerationConfig(
                content_asset_storage="s3",
                content_asset_public_base_url="https://cdn.example.com",
            ),
            s3_client=FakeS3Client(),
        )


def test_build_content_asset_service_requires_s3_public_base_url() -> None:
    with pytest.raises(ValueError, match="CONTENT_ASSET_PUBLIC_BASE_URL is required"):
        build_content_asset_service(
            ContentGenerationConfig(
                content_asset_storage="s3",
                content_asset_s3_bucket="loop-assets",
            ),
            s3_client=FakeS3Client(),
        )


def test_content_generation_config_reads_s3_env(monkeypatch) -> None:
    monkeypatch.setenv("CONTENT_ASSET_STORAGE", "s3")
    monkeypatch.setenv("CONTENT_ASSET_PUBLIC_BASE_URL", "https://cdn.example.com/assets")
    monkeypatch.setenv("CONTENT_ASSET_S3_BUCKET", "loop-assets")
    monkeypatch.setenv("CONTENT_ASSET_S3_REGION", "ap-northeast-2")
    monkeypatch.setenv("CONTENT_ASSET_S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("CONTENT_ASSET_S3_CACHE_CONTROL", "public, max-age=60")

    config = ContentGenerationConfig.from_env()

    assert config.content_asset_storage == "s3"
    assert config.content_asset_public_base_url == "https://cdn.example.com/assets"
    assert config.content_asset_s3_bucket == "loop-assets"
    assert config.content_asset_s3_region == "ap-northeast-2"
    assert config.content_asset_s3_endpoint_url == "https://s3.example.com"
    assert config.content_asset_s3_cache_control == "public, max-age=60"
