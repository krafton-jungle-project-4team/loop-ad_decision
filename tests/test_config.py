from __future__ import annotations

import os

import pytest

from app.config import SettingsError, load_local_dotenv, load_settings


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


def test_load_settings_fails_fast_when_required_env_missing() -> None:
    env = valid_env()
    del env["LOOPAD_INTERNAL_API_KEY"]

    with pytest.raises(SettingsError, match="LOOPAD_INTERNAL_API_KEY"):
        load_settings(env)


def test_load_settings_rejects_wrong_service_id() -> None:
    env = valid_env()
    env["LOOPAD_SERVICE_ID"] = "dashboard-api"

    with pytest.raises(SettingsError, match="decision-api"):
        load_settings(env)


def test_load_settings_rejects_invalid_port() -> None:
    env = valid_env()
    env["PORT"] = "not-a-port"

    with pytest.raises(SettingsError, match="PORT"):
        load_settings(env)


def test_load_settings_reads_loopad_contract_env() -> None:
    settings = load_settings(valid_env())

    assert settings.env == "dev"
    assert settings.service_id == "decision-api"
    assert settings.port == 8080
    assert settings.internal_api_key == "internal-secret"
    assert settings.aurora_host == "aurora.local"
    assert settings.aurora_port == 5432
    assert settings.clickhouse_url == "http://clickhouse.local:8123"
    assert settings.legacy_admin_token is None


def test_load_settings_reads_optional_legacy_admin_token() -> None:
    env = valid_env()
    env["AI_DECISION_ADMIN_TOKEN"] = "legacy-secret"

    settings = load_settings(env)

    assert settings.legacy_admin_token == "legacy-secret"


def test_load_local_dotenv_does_not_override_existing_env(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PORT", "9000")
    tmp_path.joinpath(".env").write_text(
        "\n".join(
            [
                "LOOPAD_ENV=dev",
                "LOOPAD_SERVICE_ID=decision-api",
                "PORT=8090",
                "LOOPAD_INTERNAL_API_KEY=local-internal-secret",
                "LOOPAD_AURORA_HOST=localhost",
                "LOOPAD_AURORA_PORT=15432",
                "LOOPAD_AURORA_DATABASE=loopad",
                "LOOPAD_AURORA_USERNAME=loopad",
                "LOOPAD_AURORA_PASSWORD=loopad",
                "LOOPAD_CLICKHOUSE_URL=http://localhost:18123",
                "LOOPAD_CLICKHOUSE_DATABASE=loopad",
                "LOOPAD_CLICKHOUSE_USERNAME=loopad_app",
                "LOOPAD_CLICKHOUSE_PASSWORD=loopad_local_password",
            ]
        ),
        encoding="utf-8",
    )

    load_local_dotenv()

    assert os.environ["PORT"] == "9000"
    assert os.environ["LOOPAD_SERVICE_ID"] == "decision-api"
