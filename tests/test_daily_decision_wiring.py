from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.contents.assets import AssetObject, ContentAssetService, S3AssetStorage
from app.contents.config import ContentGenerationConfig
from app.contents.generators import MockContentGenerator
from app.contents.postgres_repository import PostgresContentRepository
from app.contents.service import ContentGenerationService
from app.contents.visuals import MockBannerVisualProvider
from app.jobs.wiring import build_content_generation_service


@dataclass
class FakeConnection:
    cursors: list[object]

    def cursor(self, *args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise AssertionError("wiring tests should not execute database queries")


class FakeS3Client:
    def __init__(self) -> None:
        self.put_object_calls: list[dict[str, object]] = []

    def put_object(self, **kwargs):
        self.put_object_calls.append(kwargs)
        return {"ETag": "fake-etag"}


def test_build_content_generation_service_wires_content_dependencies() -> None:
    connection = FakeConnection(cursors=[])

    service = build_content_generation_service(
        connection=connection,
        config=ContentGenerationConfig(
            content_asset_storage="memory",
            content_asset_public_base_url="https://cdn.example.com",
        ),
    )

    assert isinstance(service, ContentGenerationService)
    assert isinstance(service.repository, PostgresContentRepository)
    assert service.repository.connection.connection is connection
    assert isinstance(service.generator, MockContentGenerator)
    assert isinstance(service.asset_service, ContentAssetService)


def test_build_content_generation_service_accepts_explicit_overrides() -> None:
    connection = FakeConnection(cursors=[])
    generator = MockContentGenerator(generation_model="mock-override")
    asset_service = ContentAssetService(storage=type("Storage", (), {"put_object": None})())

    service = build_content_generation_service(
        connection=connection,
        config=ContentGenerationConfig(content_asset_storage="memory"),
        generator=generator,
        asset_service=asset_service,
    )

    assert service.generator is generator
    assert service.asset_service is asset_service


def test_build_content_generation_service_can_wire_s3_asset_storage() -> None:
    s3_client = FakeS3Client()

    service = build_content_generation_service(
        connection=FakeConnection(cursors=[]),
        config=ContentGenerationConfig(
            content_asset_storage="s3",
            content_asset_public_base_url="https://cdn.example.com",
            content_asset_s3_bucket="loop-assets",
        ),
        s3_client=s3_client,
    )

    assert isinstance(service.asset_service, ContentAssetService)
    assert isinstance(service.asset_service.storage, S3AssetStorage)
    assert service.asset_service.storage.bucket == "loop-assets"
    service.asset_service.storage.put_object(
        AssetObject(
            key="generated-contents/banner.svg",
            body=b"<svg />",
            content_type="image/svg+xml",
        )
    )
    assert s3_client.put_object_calls[0]["Bucket"] == "loop-assets"


def test_build_content_generation_service_passes_visual_provider_to_asset_service() -> None:
    visual_provider = MockBannerVisualProvider(model="mock-visual-test")

    service = build_content_generation_service(
        connection=FakeConnection(cursors=[]),
        config=ContentGenerationConfig(
            content_asset_storage="memory",
            content_asset_public_base_url="https://cdn.example.com",
        ),
        visual_provider=visual_provider,
    )

    assert service.asset_service.visual_provider is visual_provider


def test_build_content_generation_service_requires_asset_storage_config() -> None:
    with pytest.raises(ValueError, match="content_asset_storage is required"):
        build_content_generation_service(
            connection=FakeConnection(cursors=[]),
            config=ContentGenerationConfig(app_env="production"),
        )
