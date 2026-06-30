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
from app.contents.visuals import BannerVisualProvider, GeminiBannerVisualProvider

AWS_SEOUL_REGION = "ap-northeast-2"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"


@dataclass(frozen=True)
class ContentGenerationConfig:
    app_env: str | None = None
    openai_api_key: str | None = None
    openai_content_model: str | None = None
    gemini_api_key: str | None = None
    gemini_image_model: str = DEFAULT_GEMINI_IMAGE_MODEL
    content_asset_storage: str | None = None
    content_asset_local_dir: str | None = None
    content_asset_prefix: str = DEFAULT_ASSET_PREFIX
    content_asset_public_base_url: str | None = None
    content_asset_s3_bucket: str | None = None
    content_asset_s3_region: str | None = None
    content_asset_s3_endpoint_url: str | None = None
    content_asset_s3_cache_control: str | None = DEFAULT_S3_CACHE_CONTROL
    content_asset_public_url_strip_prefix: str | None = None

    @classmethod
    def from_env(cls) -> "ContentGenerationConfig":
        values = {
            "LOOPAD_ENV": _clean(os.getenv("LOOPAD_ENV")),
            "LOOPAD_DATA_STORAGE_BUCKET": _clean(os.getenv("LOOPAD_DATA_STORAGE_BUCKET")),
            "LOOPAD_GENAI_ASSETS_BASE_PREFIX": _clean(
                os.getenv("LOOPAD_GENAI_ASSETS_BASE_PREFIX")
            ),
            "LOOPAD_OPENAI_API_KEY": _clean(os.getenv("LOOPAD_OPENAI_API_KEY")),
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError("missing required env: " + ", ".join(missing))

        app_env = values["LOOPAD_ENV"]
        data_storage_bucket = values["LOOPAD_DATA_STORAGE_BUCKET"]
        assets_base_prefix = values["LOOPAD_GENAI_ASSETS_BASE_PREFIX"]
        openai_api_key = values["LOOPAD_OPENAI_API_KEY"]
        if (
            app_env is None
            or data_storage_bucket is None
            or assets_base_prefix is None
            or openai_api_key is None
        ):
            raise ValueError("missing required LoopAd runtime env")
        asset_prefix = _normalize_assets_base_prefix(assets_base_prefix)

        return cls(
            app_env=app_env,
            openai_api_key=openai_api_key,
            openai_content_model=_clean(os.getenv("LOOPAD_OPENAI_CONTENT_MODEL")),
            gemini_api_key=_clean(os.getenv("LOOPAD_GEMINI_API_KEY")),
            content_asset_storage="s3",
            content_asset_prefix=asset_prefix,
            content_asset_public_base_url=_build_s3_public_base_url(data_storage_bucket),
            content_asset_s3_bucket=data_storage_bucket,
            content_asset_s3_region=AWS_SEOUL_REGION,
        )


def build_content_generator(config: ContentGenerationConfig | None = None) -> ContentGenerator:
    config = config or ContentGenerationConfig.from_env()
    if config.openai_api_key and config.openai_content_model:
        return OpenAIContentGenerator(
            api_key=config.openai_api_key,
            model=config.openai_content_model,
        )
    return MockContentGenerator()


def build_banner_visual_provider(
    config: ContentGenerationConfig | None = None,
) -> BannerVisualProvider | None:
    config = config or ContentGenerationConfig.from_env()
    if config.gemini_api_key is None:
        return None
    return GeminiBannerVisualProvider(
        api_key=config.gemini_api_key,
        model=config.gemini_image_model,
    )


def build_content_asset_service(
    config: ContentGenerationConfig | None = None,
    *,
    s3_client: S3ClientLike | None = None,
    visual_provider: BannerVisualProvider | None = None,
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
            raise ValueError("LOOPAD_DATA_STORAGE_BUCKET is required")
        if config.content_asset_public_base_url is None:
            raise ValueError("content_asset_public_base_url is required")
        storage = S3AssetStorage(
            bucket=config.content_asset_s3_bucket,
            public_base_url=_resolve_s3_public_base_url(config),
            client=s3_client,
            region_name=config.content_asset_s3_region,
            endpoint_url=config.content_asset_s3_endpoint_url,
            cache_control=config.content_asset_s3_cache_control,
            public_url_strip_prefix=config.content_asset_public_url_strip_prefix,
        )
    else:
        raise ValueError("content_asset_storage must be memory, local, or s3")

    resolved_visual_provider = (
        visual_provider
        if visual_provider is not None
        else build_banner_visual_provider(config)
    )
    return ContentAssetService(
        storage=storage,
        visual_provider=resolved_visual_provider,
        asset_prefix=config.content_asset_prefix,
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_assets_base_prefix(value: str) -> str:
    stripped = value.strip().strip("/")
    if not stripped:
        raise ValueError("LOOPAD_GENAI_ASSETS_BASE_PREFIX must include an asset path prefix")
    path = Path(stripped)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("LOOPAD_GENAI_ASSETS_BASE_PREFIX must be a relative S3 key prefix")
    return stripped


def _build_s3_public_base_url(bucket: str) -> str:
    return f"https://{bucket}.s3.{AWS_SEOUL_REGION}.amazonaws.com"


def _resolve_content_asset_storage(config: ContentGenerationConfig) -> str:
    storage_name = _clean(config.content_asset_storage)
    if storage_name is None:
        raise ValueError("content_asset_storage is required")

    normalized = storage_name.lower()
    return normalized


def _resolve_s3_public_base_url(config: ContentGenerationConfig) -> str:
    if config.content_asset_public_base_url is not None:
        return config.content_asset_public_base_url
    raise ValueError("content_asset_public_base_url is required")
