from __future__ import annotations

from app.config import REQUIRED_ENV_NAMES


def required_env_values() -> dict[str, str]:
    return {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
