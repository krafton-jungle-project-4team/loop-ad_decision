from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ContextManager, Protocol
from urllib.parse import urlparse

from app.core.config import Settings, get_settings


class ClickHouseQueryResult(Protocol):
    result_rows: list[tuple[Any, ...]]


class ClickHouseClient(Protocol):
    def query(self, query: str, parameters: dict[str, Any] | None = None) -> ClickHouseQueryResult:
        ...


ClickHouseClientFactory = Callable[[], ContextManager[ClickHouseClient]]


def create_clickhouse_client(settings: Settings | None = None) -> ClickHouseClient:
    import clickhouse_connect

    resolved_settings = settings or get_settings()
    endpoint = urlparse(resolved_settings.loopad_clickhouse_url)
    return clickhouse_connect.get_client(
        host=endpoint.hostname,
        port=endpoint.port,
        username=resolved_settings.loopad_clickhouse_username,
        secure=endpoint.scheme == "https",
    )


@contextmanager
def get_clickhouse_client() -> Iterator[ClickHouseClient]:
    client = create_clickhouse_client()
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def get_clickhouse_client_factory() -> ClickHouseClientFactory:
    return get_clickhouse_client
