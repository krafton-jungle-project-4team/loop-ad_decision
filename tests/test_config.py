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


def test_load_settings_requires_gemini_api_key() -> None:
    env = valid_env()
    env.pop("LOOPAD_GEMINI_API_KEY")

    with pytest.raises(SettingsError, match="LOOPAD_GEMINI_API_KEY"):
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
    assert settings.gemini_api_key == "value-for-loopad_gemini_api_key"
    assert settings.postgres_pool_min_size == 1
    assert settings.postgres_pool_max_size == 5
    assert settings.postgres_pool_timeout_seconds == 30.0


def test_load_settings_accepts_optional_postgres_pool_overrides() -> None:
    env = valid_env()
    env.update(
        {
            "LOOPAD_POSTGRES_POOL_MIN_SIZE": "2",
            "LOOPAD_POSTGRES_POOL_MAX_SIZE": "8",
            "LOOPAD_POSTGRES_POOL_TIMEOUT_SECONDS": "12.5",
        }
    )

    settings = load_settings(env)

    assert settings.postgres_pool_min_size == 2
    assert settings.postgres_pool_max_size == 8
    assert settings.postgres_pool_timeout_seconds == 12.5


def test_postgres_pool_env_values_are_optional_not_required() -> None:
    assert "LOOPAD_POSTGRES_POOL_MIN_SIZE" not in REQUIRED_ENV_NAMES
    assert "LOOPAD_POSTGRES_POOL_MAX_SIZE" not in REQUIRED_ENV_NAMES
    assert "LOOPAD_POSTGRES_POOL_TIMEOUT_SECONDS" not in REQUIRED_ENV_NAMES


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LOOPAD_POSTGRES_POOL_MIN_SIZE", "-1", "non-negative integer"),
        ("LOOPAD_POSTGRES_POOL_MIN_SIZE", "bad", "non-negative integer"),
        ("LOOPAD_POSTGRES_POOL_MAX_SIZE", "0", "positive integer"),
        ("LOOPAD_POSTGRES_POOL_MAX_SIZE", "bad", "positive integer"),
        ("LOOPAD_POSTGRES_POOL_TIMEOUT_SECONDS", "0", "positive number"),
        ("LOOPAD_POSTGRES_POOL_TIMEOUT_SECONDS", "bad", "positive number"),
    ],
)
def test_load_settings_rejects_invalid_postgres_pool_values(
    name: str,
    value: str,
    message: str,
) -> None:
    env = valid_env()
    env[name] = value

    with pytest.raises(SettingsError, match=message):
        load_settings(env)


def test_load_settings_rejects_postgres_pool_min_greater_than_max() -> None:
    env = valid_env()
    env["LOOPAD_POSTGRES_POOL_MIN_SIZE"] = "6"
    env["LOOPAD_POSTGRES_POOL_MAX_SIZE"] = "5"

    with pytest.raises(SettingsError, match="less than or equal"):
        load_settings(env)
