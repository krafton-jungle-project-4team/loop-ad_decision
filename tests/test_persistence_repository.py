from app.core.config import Settings
from app.db.postgres import build_postgres_url, create_postgres_engine
from app.persistence.repository import build_segment_hash, canonical_segment_json


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


def test_build_postgres_url_prefers_explicit_url() -> None:
    settings = Settings(postgres_url="postgresql+psycopg://user:pass@db:5432/custom")

    assert build_postgres_url(settings) == "postgresql+psycopg://user:pass@db:5432/custom"


def test_build_postgres_url_uses_individual_settings() -> None:
    settings = Settings(
        postgres_host="postgres",
        postgres_port=5433,
        postgres_user="loopad",
        postgres_password="secret",
        postgres_database="decision",
    )

    assert build_postgres_url(settings) == "postgresql+psycopg://loopad:***@postgres:5433/decision"


def test_postgres_engine_is_created_lazily() -> None:
    create_postgres_engine.cache_clear()

    assert create_postgres_engine.cache_info().currsize == 0
