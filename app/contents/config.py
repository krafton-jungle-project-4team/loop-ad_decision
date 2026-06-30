from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.contents.assets import (
    DEFAULT_ASSET_PREFIX,
    DEFAULT_S3_CACHE_CONTROL,
    ContentAssetService,
    InMemoryAssetStorage,
    LocalAssetStorage,
    S3AssetStorage,
    S3ClientLike,
)
from app.contents.generators import ContentGenerator, MockContentGenerator, OpenAIContentGenerator


@dataclass(frozen=True)
class ContentGenerationConfig:
    app_env: str | None = None
    openai_api_key: str | None = None
    openai_content_model: str | None = None
    content_asset_storage: str | None = None
    content_asset_local_dir: str | None = None
    content_asset_prefix: str = DEFAULT_ASSET_PREFIX
    content_asset_public_base_url: str | None = None
    content_asset_s3_bucket: str | None = None
    content_asset_s3_region: str | None = None
    content_asset_s3_endpoint_url: str | None = None
    content_asset_s3_cache_control: str | None = DEFAULT_S3_CACHE_CONTROL

    @classmethod
    def from_env(cls) -> "ContentGenerationConfig":
        return cls(
            app_env=_resolve_app_env(),
            openai_api_key=_clean(os.getenv("OPENAI_API_KEY")),
            openai_content_model=_clean(os.getenv("OPENAI_CONTENT_MODEL")),
            content_asset_storage=_clean(os.getenv("CONTENT_ASSET_STORAGE")),
            content_asset_local_dir=_clean(os.getenv("CONTENT_ASSET_LOCAL_DIR")),
            content_asset_prefix=_clean(os.getenv("CONTENT_ASSET_PREFIX")) or DEFAULT_ASSET_PREFIX,
            content_asset_public_base_url=_clean(os.getenv("CONTENT_ASSET_PUBLIC_BASE_URL")),
            content_asset_s3_bucket=_clean(os.getenv("CONTENT_ASSET_S3_BUCKET")),
            content_asset_s3_region=_clean(os.getenv("CONTENT_ASSET_S3_REGION")),
            content_asset_s3_endpoint_url=_clean(os.getenv("CONTENT_ASSET_S3_ENDPOINT_URL")),
            content_asset_s3_cache_control=(
                _clean(os.getenv("CONTENT_ASSET_S3_CACHE_CONTROL")) or DEFAULT_S3_CACHE_CONTROL
            ),
        )


def build_content_generator(config: ContentGenerationConfig | None = None) -> ContentGenerator:
    config = config or ContentGenerationConfig.from_env()
    if config.openai_api_key and config.openai_content_model:
        return OpenAIContentGenerator(
            api_key=config.openai_api_key,
            model=config.openai_content_model,
        )
    return MockContentGenerator()


def build_content_asset_service(
    config: ContentGenerationConfig | None = None,
    *,
    s3_client: S3ClientLike | None = None,
) -> ContentAssetService:
    config = config or ContentGenerationConfig.from_env()
    storage_name = _resolve_content_asset_storage(config)
    if storage_name == "memory":
        storage = InMemoryAssetStorage(public_base_url=config.content_asset_public_base_url)
    elif storage_name == "local":
        storage = LocalAssetStorage(
            root_dir=Path(config.content_asset_local_dir or ".generated-assets"),
            public_base_url=config.content_asset_public_base_url,
        )
    elif storage_name == "s3":
        if config.content_asset_s3_bucket is None:
            raise ValueError("CONTENT_ASSET_S3_BUCKET is required when CONTENT_ASSET_STORAGE=s3")
        if config.content_asset_public_base_url is None:
            raise ValueError("CONTENT_ASSET_PUBLIC_BASE_URL is required when CONTENT_ASSET_STORAGE=s3")
        storage = S3AssetStorage(
            bucket=config.content_asset_s3_bucket,
            public_base_url=config.content_asset_public_base_url,
            client=s3_client,
            region_name=config.content_asset_s3_region,
            endpoint_url=config.content_asset_s3_endpoint_url,
            cache_control=config.content_asset_s3_cache_control,
        )
    else:
        raise ValueError("CONTENT_ASSET_STORAGE must be memory, local, or s3")

    return ContentAssetService(
        storage=storage,
        asset_prefix=config.content_asset_prefix,
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_content_asset_storage(config: ContentGenerationConfig) -> str:
    storage_name = _clean(config.content_asset_storage)
    if storage_name is None:
        if _is_production(config.app_env):
            raise ValueError("CONTENT_ASSET_STORAGE is required in production")
        return "memory"

    normalized = storage_name.lower()
    if normalized == "memory" and _is_production(config.app_env):
        raise ValueError("CONTENT_ASSET_STORAGE=memory is not allowed in production")
    return normalized


def _is_production(app_env: str | None) -> bool:
    return (app_env or "").strip().lower() == "production"


def _resolve_app_env() -> str | None:
    app_env = _clean(os.getenv("APP_ENV"))
    env = _clean(os.getenv("ENV"))
    if _is_production(app_env) or _is_production(env):
        return "production"
    return app_env or env
