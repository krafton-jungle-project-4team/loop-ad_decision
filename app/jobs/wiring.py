from __future__ import annotations

from typing import Any, Protocol

from psycopg.rows import dict_row

from app.contents.assets import ContentAssetService
from app.contents.config import (
    ContentGenerationConfig,
    build_content_asset_service,
    build_content_generator,
)
from app.contents.generators import ContentGenerator
from app.contents.postgres_repository import PostgresContentRepository
from app.contents.service import ContentGenerationService


class ConnectionLike(Protocol):
    def cursor(self, *args: Any, **kwargs: Any) -> object:
        ...


class _BufferedDictCursor:
    def __init__(self, rows: list[dict[str, Any]], rowcount: int) -> None:
        self.rows = rows
        self.rowcount = rowcount

    def fetchone(self) -> dict[str, Any] | None:
        if not self.rows:
            return None
        return self.rows.pop(0)

    def fetchall(self) -> list[dict[str, Any]]:
        rows = self.rows
        self.rows = []
        return rows


class _DictRowConnectionAdapter:
    def __init__(self, connection: ConnectionLike) -> None:
        self.connection = connection

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> _BufferedDictCursor:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall() if cursor.description is not None else []
            return _BufferedDictCursor(rows=list(rows), rowcount=cursor.rowcount)


def build_content_generation_service(
    *,
    connection: ConnectionLike,
    config: ContentGenerationConfig | None = None,
    generator: ContentGenerator | None = None,
    asset_service: ContentAssetService | None = None,
) -> ContentGenerationService:
    """Build the content generation service used by DailyDecisionJobService.

    The caller owns the database connection lifecycle and transaction boundary.
    This creates no external API route, scheduler, S3 client, or Gemini client.
    """

    content_config = config or ContentGenerationConfig.from_env()
    return ContentGenerationService(
        repository=PostgresContentRepository(_DictRowConnectionAdapter(connection)),
        generator=generator or build_content_generator(content_config),
        asset_service=asset_service or build_content_asset_service(content_config),
    )
