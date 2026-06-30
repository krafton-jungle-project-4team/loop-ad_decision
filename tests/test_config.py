from __future__ import annotations

import pytest

from app.config import SettingsError, load_settings


def valid_env() -> dict[str, str]:
    return {
        "LOOPAD_ENV": "dev",
        "LOOPAD_SERVICE_ID": "decision-api",
        "PORT": "8080",
        "LOOPAD_INTERNAL_API_KEY": "internal-secret",
        "LOOPAD_AURORA_HOST": "aurora.local",
        "LOOPAD_AURORA_PORT": "5432",
        "LOOPAD_AURORA_DATABASE": "loopad",
        "LOOPAD_AURORA_USERNAME": "app",
        "LOOPAD_AURORA_PASSWORD": "password",
        "LOOPAD_CLICKHOUSE_URL": "http://clickhouse.local:8123",
        "LOOPAD_CLICKHOUSE_DATABASE": "loopad",
        "LOOPAD_CLICKHOUSE_USERNAME": "app",
        "LOOPAD_CLICKHOUSE_PASSWORD": "password",
    }


def test_load_settings_requires_loopad_contract_env_names() -> None:
    env = valid_env()
    del env["LOOPAD_INTERNAL_API_KEY"]

    with pytest.raises(SettingsError, match="LOOPAD_INTERNAL_API_KEY"):
        load_settings(env)


def test_load_settings_rejects_wrong_service_id() -> None:
    env = valid_env()
    env["LOOPAD_SERVICE_ID"] = "dashboard-api"

    with pytest.raises(SettingsError, match="decision-api"):
        load_settings(env)


def test_load_settings_parses_port_and_keeps_legacy_token_optional() -> None:
    env = valid_env()
    env["AI_DECISION_ADMIN_TOKEN"] = "legacy-secret"

    settings = load_settings(env)

    assert settings.service_id == "decision-api"
    assert settings.port == 8080
    assert settings.aurora_port == 5432
    assert settings.legacy_admin_token == "legacy-secret"
