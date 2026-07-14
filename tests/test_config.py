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
            "LOOPAD_SEGMENT_PERFORMANCE_MODEL_PATH": "/models/segment.json",
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
    assert settings.segment_performance_model_path == "/models/segment.json"


def test_load_settings_uses_safe_generation_defaults() -> None:
    settings = load_settings(valid_env())

    assert (
        settings.genai_assets_public_base_url
        == "https://gen-ai.asset.dev.loop-ad.org"
    )
    assert settings.generation_worker_max_concurrency == 2
    assert settings.generation_poll_interval_seconds == 1
    assert settings.generation_idle_poll_interval_seconds == 30
    assert settings.generation_lease_seconds == 180
    assert settings.generation_heartbeat_seconds == 30
    assert settings.generation_max_retries == 3
    assert settings.generation_retry_backoff_seconds == (60, 300, 900)
    assert settings.generation_provider_timeout_seconds == 30
    assert settings.generation_db_operation_timeout_seconds == 5
    assert settings.generation_shutdown_grace_seconds == 20


def test_load_settings_collects_generation_overrides() -> None:
    env = valid_env()
    env.update(
        {
            "GENERATION_WORKER_MAX_CONCURRENCY": "4",
            "GENERATION_POLL_INTERVAL_SECONDS": "2",
            "GENERATION_IDLE_POLL_INTERVAL_SECONDS": "45",
            "GENERATION_LEASE_SECONDS": "240",
            "GENERATION_HEARTBEAT_SECONDS": "20",
            "GENERATION_MAX_RETRIES": "2",
            "GENERATION_RETRY_BACKOFF_SECONDS": "10, 20, 30",
            "GENERATION_PROVIDER_TIMEOUT_SECONDS": "40",
            "GENERATION_DB_OPERATION_TIMEOUT_SECONDS": "4",
            "GENERATION_SHUTDOWN_GRACE_SECONDS": "15",
            "LOOPAD_GENAI_ASSETS_PUBLIC_BASE_URL": (
                "https://assets.example.test/genai"
            ),
        }
    )

    settings = load_settings(env)

    assert settings.generation_worker_max_concurrency == 4
    assert settings.generation_poll_interval_seconds == 2
    assert settings.generation_idle_poll_interval_seconds == 45
    assert settings.generation_lease_seconds == 240
    assert settings.generation_heartbeat_seconds == 20
    assert settings.generation_max_retries == 2
    assert settings.generation_retry_backoff_seconds == (10, 20, 30)
    assert settings.generation_provider_timeout_seconds == 40
    assert settings.generation_db_operation_timeout_seconds == 4
    assert settings.generation_shutdown_grace_seconds == 15
    assert (
        settings.genai_assets_public_base_url
        == "https://assets.example.test/genai"
    )


def test_load_settings_requires_heartbeat_shorter_than_lease() -> None:
    env = valid_env()
    env["GENERATION_HEARTBEAT_SECONDS"] = "180"

    with pytest.raises(SettingsError, match="HEARTBEAT.*less than.*LEASE"):
        load_settings(env)


def test_load_settings_requires_backoff_for_each_retry() -> None:
    env = valid_env()
    env["GENERATION_MAX_RETRIES"] = "3"
    env["GENERATION_RETRY_BACKOFF_SECONDS"] = "60,300"

    with pytest.raises(SettingsError, match="BACKOFF.*MAX_RETRIES"):
        load_settings(env)


def test_load_settings_requires_coordinator_db_budget_shorter_than_lease() -> None:
    env = valid_env()
    env.update(
        {
            "GENERATION_WORKER_MAX_CONCURRENCY": "2",
            "GENERATION_HEARTBEAT_SECONDS": "3",
            "GENERATION_DB_OPERATION_TIMEOUT_SECONDS": "2",
            "GENERATION_LEASE_SECONDS": "10",
        }
    )

    with pytest.raises(SettingsError, match="DB timeout budget.*less than.*LEASE"):
        load_settings(env)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GENERATION_WORKER_MAX_CONCURRENCY", "0"),
        ("GENERATION_MAX_RETRIES", "-1"),
        ("GENERATION_DB_OPERATION_TIMEOUT_SECONDS", "0"),
        ("GENERATION_RETRY_BACKOFF_SECONDS", "60,,900"),
    ],
)
def test_load_settings_rejects_invalid_generation_values(
    name: str,
    value: str,
) -> None:
    env = valid_env()
    env[name] = value

    with pytest.raises(SettingsError, match=name):
        load_settings(env)
