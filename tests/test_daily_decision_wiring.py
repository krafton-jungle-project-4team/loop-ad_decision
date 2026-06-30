from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.contents.assets import ContentAssetService
from app.contents.config import ContentGenerationConfig
from app.contents.generators import MockContentGenerator
from app.contents.postgres_repository import PostgresContentRepository
from app.contents.service import ContentGenerationService
from app.jobs.wiring import build_content_generation_service


@dataclass
class FakeConnection:
    cursors: list[object]

    def cursor(self, *args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise AssertionError("wiring tests should not execute database queries")


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


def test_build_content_generation_service_keeps_s3_deferred() -> None:
    with pytest.raises(NotImplementedError):
        build_content_generation_service(
            connection=FakeConnection(cursors=[]),
            config=ContentGenerationConfig(content_asset_storage="s3"),
        )


def test_build_content_generation_service_enforces_production_storage_config() -> None:
    with pytest.raises(ValueError, match="CONTENT_ASSET_STORAGE is required"):
        build_content_generation_service(
            connection=FakeConnection(cursors=[]),
            config=ContentGenerationConfig(app_env="production"),
        )
