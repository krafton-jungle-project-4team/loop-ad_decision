from __future__ import annotations

import pytest

from app.config import (
    GENAI_SOURCE_MANIFEST_PREFIX,
    REQUIRED_ENV_NAMES,
    SettingsError,
    load_settings,
)


def valid_env() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "15432",
            "LOOPAD_GENAI_ASSETS_PUBLIC_BASE_URL": "https://assets.example.test",
            "LOOPAD_BRAND_CONTEXT_BASE_PREFIX": "brand-context/",
            "LOOPAD_OPENAI_CONTENT_MODEL": "gpt-test",
            "LOOPAD_GEMINI_IMAGE_MODEL": "gemini-test",
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


@pytest.mark.parametrize(
    "name",
    [
        "LOOPAD_GENAI_ASSETS_PUBLIC_BASE_URL",
        "LOOPAD_BRAND_CONTEXT_BASE_PREFIX",
        "LOOPAD_OPENAI_CONTENT_MODEL",
        "LOOPAD_GEMINI_IMAGE_MODEL",
    ],
)
def test_load_settings_requires_provider_runtime_env(name: str) -> None:
    env = valid_env()
    env.pop(name)

    with pytest.raises(SettingsError, match=name):
        load_settings(env)


def test_source_manifest_prefix_is_application_invariant() -> None:
    assert GENAI_SOURCE_MANIFEST_PREFIX == "genai-source/"


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
    assert settings.gemini_image_model == "gemini-test"
    assert settings.brand_context_base_prefix == "brand-context/"
    assert settings.gemini_api_key == "value-for-loopad_gemini_api_key"
    assert settings.segment_performance_model_path == "/models/segment.json"


def test_load_settings_rejects_public_prefix_matching_source_manifest() -> None:
    env = valid_env()
    env["LOOPAD_GENAI_ASSETS_BASE_PREFIX"] = "genai-source/"

    with pytest.raises(SettingsError, match="public"):
        load_settings(env)


def test_load_settings_rejects_brand_context_inside_public_prefix() -> None:
    env = valid_env()
    env["LOOPAD_GENAI_ASSETS_BASE_PREFIX"] = "genai/"
    env["LOOPAD_BRAND_CONTEXT_BASE_PREFIX"] = "genai/brand-context/"

    with pytest.raises(SettingsError, match="public"):
        load_settings(env)


def test_load_settings_rejects_brand_context_overlapping_source_manifest() -> None:
    env = valid_env()
    env["LOOPAD_BRAND_CONTEXT_BASE_PREFIX"] = "genai-source/"

    with pytest.raises(SettingsError, match="GENAI_SOURCE_MANIFEST_PREFIX"):
        load_settings(env)


def test_load_settings_uses_generation_worker_code_policy() -> None:
    settings = load_settings(valid_env())

    assert (
        settings.genai_assets_public_base_url
        == "https://assets.example.test"
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


def test_generation_worker_env_names_do_not_override_code_policy() -> None:
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
        }
    )

    settings = load_settings(env)

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
