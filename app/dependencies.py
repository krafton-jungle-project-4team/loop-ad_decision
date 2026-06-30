from __future__ import annotations

from urllib.parse import urlparse

import clickhouse_connect
import psycopg

from app.config import Settings


def connect_postgres(settings: Settings):
    return psycopg.connect(
        host=settings.aurora_host,
        port=settings.aurora_port,
        dbname=settings.aurora_database,
        user=settings.aurora_username,
        password=settings.aurora_password,
    )


def connect_clickhouse(settings: Settings):
    parsed_url = urlparse(settings.clickhouse_url)
    if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname is None:
        raise ValueError("LOOPAD_CLICKHOUSE_URL must be an http(s) URL with a host")

    return clickhouse_connect.get_client(
        host=parsed_url.hostname,
        port=parsed_url.port,
        username=settings.clickhouse_username,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
        secure=parsed_url.scheme == "https",
    )
