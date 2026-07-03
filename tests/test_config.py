from __future__ import annotations

import pytest

from app.config import REQUIRED_ENV_NAMES, SettingsError, load_settings


def valid_env() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "15432",
            "LOOPAD_OPENAI_CONTENT_MODEL": "gpt-test",
        }
    )
    return values


def test_load_settings_requires_all_contract_env_values() -> None:
    env = valid_env()
    env.pop("LOOPAD_OPENAI_API_KEY")

    with pytest.raises(SettingsError, match="LOOPAD_OPENAI_API_KEY"):
        load_settings(env)


def test_load_settings_rejects_wrong_service_id() -> None:
    env = valid_env()
    env["LOOPAD_SERVICE_ID"] = "dashboard-api"

    with pytest.raises(SettingsError, match="decision-api"):
        load_settings(env)


def test_load_settings_rejects_invalid_port() -> None:
    env = valid_env()
    env["PORT"] = "0"

    with pytest.raises(SettingsError, match="PORT"):
        load_settings(env)


def test_load_settings_collects_validated_values() -> None:
    settings = load_settings(valid_env())

    assert settings.service_id == "decision-api"
    assert settings.port == 8080
    assert settings.aurora_port == 15432
    assert settings.openai_content_model == "gpt-test"
