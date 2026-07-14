from __future__ import annotations

from typing import Any

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.db import (
    create_clickhouse_client,
    create_generation_coordinator_connection,
    create_postgres_connection,
)


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


def test_create_postgres_connection_uses_validated_aurora_settings(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_connect(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("app.db.psycopg.connect", fake_connect)
    settings = load_settings(valid_env())

    create_postgres_connection(settings)

    assert calls == [
        {
            "host": settings.aurora_host,
            "port": settings.aurora_port,
            "dbname": settings.aurora_database,
            "user": settings.aurora_username,
            "password": settings.aurora_password,
            "autocommit": False,
        }
    ]


def test_create_clickhouse_client_uses_validated_clickhouse_settings(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_get_client(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("app.db.clickhouse_connect.get_client", fake_get_client)
    settings = load_settings(valid_env())

    create_clickhouse_client(settings)

    assert calls == [
        {
            "interface": "http",
            "dsn": settings.clickhouse_url,
            "database": settings.clickhouse_database,
            "username": settings.clickhouse_username,
            "password": settings.clickhouse_password,
        }
    ]


def test_generation_coordinator_connection_has_bounded_db_timeouts(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_connect(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("app.db.psycopg.connect", fake_connect)
    settings = load_settings(valid_env())

    create_generation_coordinator_connection(settings)

    assert calls == [
        {
            "host": settings.aurora_host,
            "port": settings.aurora_port,
            "dbname": settings.aurora_database,
            "user": settings.aurora_username,
            "password": settings.aurora_password,
            "autocommit": False,
            "connect_timeout": 5,
            "options": "-c statement_timeout=5000 -c lock_timeout=5000",
        }
    ]
