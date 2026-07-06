from __future__ import annotations

from typing import Any

import clickhouse_connect
import psycopg
from psycopg_pool import ConnectionPool

from app.config import Settings


def create_postgres_connection(settings: Settings) -> Any:
    return psycopg.connect(**_postgres_connection_kwargs(settings))


def create_postgres_pool(settings: Settings) -> ConnectionPool[Any]:
    pool = ConnectionPool(
        kwargs=_postgres_connection_kwargs(settings),
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        timeout=settings.postgres_pool_timeout_seconds,
        open=False,
    )
    pool.open(wait=False)
    return pool


def _postgres_connection_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "host": settings.aurora_host,
        "port": settings.aurora_port,
        "dbname": settings.aurora_database,
        "user": settings.aurora_username,
        "password": settings.aurora_password,
        "autocommit": False,
    }


def create_clickhouse_client(settings: Settings) -> Any:
    return clickhouse_connect.get_client(
        interface="http",
        dsn=settings.clickhouse_url,
        database=settings.clickhouse_database,
        username=settings.clickhouse_username,
        password=settings.clickhouse_password,
    )
