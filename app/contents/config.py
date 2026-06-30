from __future__ import annotations

import os
from dataclasses import dataclass

from app.contents.generators import ContentGenerator, MockContentGenerator, OpenAIContentGenerator


@dataclass(frozen=True)
class ContentGenerationConfig:
    openai_api_key: str | None = None
    openai_content_model: str | None = None

    @classmethod
    def from_env(cls) -> "ContentGenerationConfig":
        return cls(
            openai_api_key=_clean(os.getenv("OPENAI_API_KEY")),
            openai_content_model=_clean(os.getenv("OPENAI_CONTENT_MODEL")),
        )


def build_content_generator(config: ContentGenerationConfig | None = None) -> ContentGenerator:
    config = config or ContentGenerationConfig.from_env()
    if config.openai_api_key and config.openai_content_model:
        return OpenAIContentGenerator(
            api_key=config.openai_api_key,
            model=config.openai_content_model,
        )
    return MockContentGenerator()


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
