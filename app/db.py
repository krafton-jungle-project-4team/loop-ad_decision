from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import clickhouse_connect
import psycopg
from psycopg.rows import dict_row

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


def create_clickhouse_client(settings: Settings) -> Any:
    return clickhouse_connect.get_client(
        interface="http",
        dsn=settings.clickhouse_url,
        database=settings.clickhouse_database,
        username=settings.clickhouse_username,
        password=settings.clickhouse_password,
    )


class PsycopgPostgresExecutor:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
