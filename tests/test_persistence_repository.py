import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.db.postgres import build_postgres_url, create_postgres_engine
from app.persistence.repository import build_segment_hash, canonical_segment_json


def settings_with_required_env(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "loopad_env": "local",
        "loopad_service_id": "decision-api",
        "loopad_runtime": "go",
        "port": 8000,
        "loopad_aurora_host": "postgres",
        "loopad_aurora_port": 5432,
        "loopad_aurora_database": "loopad",
        "loopad_aurora_username": "postgres",
        "loopad_aurora_password": "secret",
        "loopad_clickhouse_url": "http://clickhouse:8123",
        "loopad_clickhouse_username": "default",
        "loopad_data_storage_bucket": "bucket",
        "loopad_genai_generated_assets_prefix": "genai/generated/",
        "loopad_openai_api_key": "sk-test",
        "loopad_postgres_auto_create_tables": True,
        "loopad_analysis_worker_poll_interval_seconds": 2.0,
    }
    values.update(overrides)
    return Settings(**values)


def test_canonical_segment_json_removes_null_values_and_sorts_keys() -> None:
    canonical = canonical_segment_json(
        {
            "product_id": "sku-1",
            "channel": None,
            "age_group": "30s",
        }
    )

    assert canonical == '{"age_group":"30s","product_id":"sku-1"}'


def test_build_segment_hash_uses_canonical_segment_json() -> None:
    first_hash = build_segment_hash(
        {
            "product_id": "sku-1",
            "channel": None,
            "age_group": "30s",
        }
    )
    second_hash = build_segment_hash(
        {
            "age_group": "30s",
            "product_id": "sku-1",
        }
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_settings_requires_loopad_env_contract() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_validates_decision_service_id() -> None:
    with pytest.raises(ValidationError):
        settings_with_required_env(loopad_service_id="dashboard-api")


def test_build_postgres_url_uses_aurora_settings() -> None:
    settings = settings_with_required_env(
        loopad_aurora_host="postgres",
        loopad_aurora_port=5433,
        loopad_aurora_username="loopad",
        loopad_aurora_password="secret",
        loopad_aurora_database="decision",
    )

    assert build_postgres_url(settings) == "postgresql+psycopg://loopad:***@postgres:5433/decision"


def test_postgres_engine_is_created_lazily() -> None:
    create_postgres_engine.cache_clear()

    assert create_postgres_engine.cache_info().currsize == 0
