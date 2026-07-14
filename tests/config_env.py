from __future__ import annotations

from app.config import REQUIRED_ENV_NAMES


def required_env_values() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
    values.update(
        {
            "LOOPAD_GENAI_ASSETS_PUBLIC_BASE_URL": "https://assets.example.test",
            "LOOPAD_BRAND_CONTEXT_BASE_PREFIX": "brand-context/",
            "LOOPAD_OPENAI_CONTENT_MODEL": "gpt-test",
            "LOOPAD_GEMINI_IMAGE_MODEL": "gemini-test",
        }
    )
    return values
