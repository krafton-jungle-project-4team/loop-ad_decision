from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


GENERATION_RUN_COLUMNS: tuple[str, ...] = (
    "generation_id",
    "project_id",
    "campaign_id",
    "promotion_id",
    "analysis_id",
    "content_option_count",
    "operator_instruction",
    "prompt_context_json",
    "report_json",
    "status",
    "created_at",
    "updated_at",
)


class CursorProtocol(Protocol):
    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        ...

    def fetchone(self) -> dict[str, Any] | None:
        ...


class ConnectionProtocol(Protocol):
    def cursor(self, *, row_factory: Any = None) -> Any:
        ...


@dataclass(frozen=True)
class GenerationRunRecord:
    generation_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    analysis_id: str
    content_option_count: int
    operator_instruction: str | None
    status: str
    prompt_context_json: dict[str, Any] = field(default_factory=dict)
    report_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_db_params(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "project_id": self.project_id,
            "campaign_id": self.campaign_id,
            "promotion_id": self.promotion_id,
            "analysis_id": self.analysis_id,
            "content_option_count": self.content_option_count,
            "operator_instruction": self.operator_instruction,
            "prompt_context_json": Jsonb(self.prompt_context_json),
            "report_json": Jsonb(self.report_json),
            "status": self.status,
        }


class GenerationRunRepository:
    INSERT_SQL = """
        INSERT INTO generation_runs (
            generation_id,
            project_id,
            campaign_id,
            promotion_id,
            analysis_id,
            content_option_count,
            operator_instruction,
            prompt_context_json,
            report_json,
            status
        )
        VALUES (
            %(generation_id)s,
            %(project_id)s,
            %(campaign_id)s,
            %(promotion_id)s,
            %(analysis_id)s,
            %(content_option_count)s,
            %(operator_instruction)s,
            %(prompt_context_json)s,
            %(report_json)s,
            %(status)s
        )
        RETURNING
            generation_id,
            project_id,
            campaign_id,
            promotion_id,
            analysis_id,
            content_option_count,
            operator_instruction,
            prompt_context_json,
            report_json,
            status,
            created_at,
            updated_at
    """

    SELECT_BY_ID_SQL = """
        SELECT
            generation_id,
            project_id,
            campaign_id,
            promotion_id,
            analysis_id,
            content_option_count,
            operator_instruction,
            prompt_context_json,
            report_json,
            status,
            created_at,
            updated_at
        FROM generation_runs
        WHERE generation_id = %(generation_id)s
    """

    def __init__(self, connection: ConnectionProtocol):
        self._connection = connection

    def create(self, record: GenerationRunRecord) -> dict[str, Any]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(self.INSERT_SQL, record.to_db_params())
            created = cursor.fetchone()

        if created is None:
            raise RuntimeError("generation_runs insert returned no row")
        return created

    def get_by_id(self, generation_id: str) -> dict[str, Any] | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(self.SELECT_BY_ID_SQL, {"generation_id": generation_id})
            return cursor.fetchone()
