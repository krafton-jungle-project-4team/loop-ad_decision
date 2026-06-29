"""Content generation internals for the daily decision job."""

from app.contents.assets import (
    AssetObject,
    ContentAssetService,
    InMemoryAssetStorage,
    LocalAssetStorage,
    StoredAsset,
    SvgBannerRenderer,
    build_asset_key,
)
from app.contents.config import (
    ContentGenerationConfig,
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

__all__ = [
    "ContentGenerationActionResult",
    "ContentGenerationConfig",
    "ContentGenerationService",
    "ContentGenerationSummary",
    "AssetObject",
    "ContentAssetService",
    "GENERATION_MODEL_MANUAL",
    "GENERATION_MODEL_MOCK",
    "GENERATION_MODEL_SEED",
    "GenerationLockUnavailable",
    "GeneratedContentDraft",
    "GeneratedContentRecord",
    "InMemoryAssetStorage",
    "LocalAssetStorage",
    "MockContentGenerator",
    "OpenAIContentGenerator",
    "PostgresContentRepository",
    "RecommendationActionTarget",
    "SegmentContext",
    "StoredAsset",
    "SvgBannerRenderer",
    "build_asset_key",
    "build_content_asset_service",
    "build_content_generator",
]
