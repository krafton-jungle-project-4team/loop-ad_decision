"""Content generation internals for the daily decision job."""

from app.contents.config import ContentGenerationConfig, build_content_generator
from app.contents.generators import MockContentGenerator, OpenAIContentGenerator
from app.contents.service import ContentGenerationService
from app.contents.types import (
    ContentGenerationActionResult,
    ContentGenerationSummary,
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
    "GeneratedContentDraft",
    "GeneratedContentRecord",
    "MockContentGenerator",
    "OpenAIContentGenerator",
    "RecommendationActionTarget",
    "SegmentContext",
    "build_content_generator",
]
