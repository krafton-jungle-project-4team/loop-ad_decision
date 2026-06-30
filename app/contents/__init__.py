"""Content generation internals for the daily decision job."""

from app.contents.assets import (
    AssetObject,
    ContentAssetService,
    DEFAULT_S3_CACHE_CONTROL,
    InMemoryAssetStorage,
    LocalAssetStorage,
    S3AssetStorage,
    S3ClientLike,
    StoredAsset,
    SvgBannerRenderer,
    build_asset_key,
)
from app.contents.config import (
    ContentGenerationConfig,
    build_banner_visual_provider,
    build_content_asset_service,
    build_content_generator,
)
from app.contents.generators import MockContentGenerator, OpenAIContentGenerator
from app.contents.postgres_repository import PostgresContentRepository
from app.contents.repository import GenerationLockUnavailable
from app.contents.service import ContentGenerationService
from app.contents.types import (
    ContentGenerationActionResult,
    ContentGenerationSummary,
    GENERATION_MODEL_MANUAL,
    GENERATION_MODEL_MOCK,
    GENERATION_MODEL_SEED,
    GeneratedContentDraft,
    GeneratedContentRecord,
    RecommendationActionTarget,
    SegmentContext,
)
from app.contents.visuals import (
    BannerVisual,
    GeminiBannerVisualProvider,
    MockBannerVisualProvider,
)

__all__ = [
    "ContentGenerationActionResult",
    "ContentGenerationConfig",
    "ContentGenerationService",
    "ContentGenerationSummary",
    "AssetObject",
    "BannerVisual",
    "ContentAssetService",
    "DEFAULT_S3_CACHE_CONTROL",
    "GENERATION_MODEL_MANUAL",
    "GENERATION_MODEL_MOCK",
    "GENERATION_MODEL_SEED",
    "GeminiBannerVisualProvider",
    "GenerationLockUnavailable",
    "GeneratedContentDraft",
    "GeneratedContentRecord",
    "InMemoryAssetStorage",
    "LocalAssetStorage",
    "MockBannerVisualProvider",
    "MockContentGenerator",
    "OpenAIContentGenerator",
    "PostgresContentRepository",
    "RecommendationActionTarget",
    "S3AssetStorage",
    "S3ClientLike",
    "SegmentContext",
    "StoredAsset",
    "SvgBannerRenderer",
    "build_asset_key",
    "build_banner_visual_provider",
    "build_content_asset_service",
    "build_content_generator",
]
