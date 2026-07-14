from __future__ import annotations

from typing import Any

import clickhouse_connect
import psycopg

from app.config import Settings


def create_postgres_connection(settings: Settings) -> Any:
    return psycopg.connect(
        host=settings.aurora_host,
        port=settings.aurora_port,
        dbname=settings.aurora_database,
        user=settings.aurora_username,
        password=settings.aurora_password,
        autocommit=False,
    )


def create_generation_coordinator_connection(settings: Settings) -> Any:
    """Create a bounded connection for lease claim/heartbeat/recovery work."""

    timeout_seconds = settings.generation_db_operation_timeout_seconds
    timeout_ms = timeout_seconds * 1000
    return psycopg.connect(
        host=settings.aurora_host,
        port=settings.aurora_port,
        dbname=settings.aurora_database,
        user=settings.aurora_username,
        password=settings.aurora_password,
        autocommit=False,
        connect_timeout=timeout_seconds,
        options=(
            f"-c statement_timeout={timeout_ms} "
            f"-c lock_timeout={timeout_ms}"
        ),
    )


def create_clickhouse_client(settings: Settings) -> Any:
    return clickhouse_connect.get_client(
        interface="http",
        dsn=settings.clickhouse_url,
        database=settings.clickhouse_database,
        username=settings.clickhouse_username,
        password=settings.clickhouse_password,
    )
