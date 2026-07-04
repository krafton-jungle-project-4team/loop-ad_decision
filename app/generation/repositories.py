from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.generation.schemas import ContentChannel, missing_channel_fields


GENERATION_RUN_COLUMNS: tuple[str, ...] = (
    "generation_id",
    "analysis_id",
    "project_id",
    "campaign_id",
    "promotion_id",
    "content_option_count",
    "operator_instruction",
    "input_json",
    "output_json",
    "generation_report_json",
    "status",
    "created_at",
    "updated_at",
)


CONTENT_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "content_id",
    "content_option_id",
    "generation_id",
    "analysis_id",
    "project_id",
    "campaign_id",
    "promotion_id",
    "segment_id",
    "channel",
    "subject",
    "preheader",
    "title",
    "body",
    "cta",
    "message",
    "image_prompt",
    "image_url",
    "landing_url",
    "generation_prompt",
    "reason_summary",
    "data_evidence_json",
    "message_strategy",
    "metadata_json",
    "status",
    "created_at",
    "updated_at",
)


class CursorProtocol(Protocol):
    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        ...

    def fetchone(self) -> dict[str, Any] | None:
        ...

    def fetchall(self) -> list[dict[str, Any]]:
        ...


class ConnectionProtocol(Protocol):
    def cursor(self, *, row_factory: Any = None) -> Any:
        ...


@dataclass(frozen=True)
class GenerationRunRecord:
    generation_id: str
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    content_option_count: int
    operator_instruction: str | None
    status: str
    input_json: dict[str, Any] = field(default_factory=dict)
    output_json: dict[str, Any] | None = None
    generation_report_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_db_params(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "analysis_id": self.analysis_id,
            "project_id": self.project_id,
            "campaign_id": self.campaign_id,
            "promotion_id": self.promotion_id,
            "content_option_count": self.content_option_count,
            "operator_instruction": self.operator_instruction,
            "input_json": Jsonb(self.input_json),
            "output_json": Jsonb(self.output_json) if self.output_json is not None else None,
            "generation_report_json": Jsonb(self.generation_report_json),
            "status": self.status,
        }


class GenerationRunRepository:
    INSERT_SQL = """
        INSERT INTO generation_runs (
            generation_id,
            analysis_id,
            project_id,
            campaign_id,
            promotion_id,
            content_option_count,
            operator_instruction,
            input_json,
            output_json,
            generation_report_json,
            status
        )
        VALUES (
            %(generation_id)s,
            %(analysis_id)s,
            %(project_id)s,
            %(campaign_id)s,
            %(promotion_id)s,
            %(content_option_count)s,
            %(operator_instruction)s,
            %(input_json)s,
            %(output_json)s,
            %(generation_report_json)s,
            %(status)s
        )
        RETURNING
            generation_id,
            analysis_id,
            project_id,
            campaign_id,
            promotion_id,
            content_option_count,
            operator_instruction,
            input_json,
            output_json,
            generation_report_json,
            status,
            created_at,
            updated_at
    """

    SELECT_BY_ID_SQL = """
        SELECT
            generation_id,
            analysis_id,
            project_id,
            campaign_id,
            promotion_id,
            content_option_count,
            operator_instruction,
            input_json,
            output_json,
            generation_report_json,
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


@dataclass(frozen=True)
class ContentCandidateRecord:
    content_id: str
    content_option_id: str
    generation_id: str
    analysis_id: str
    project_id: str
    campaign_id: str
    promotion_id: str
    segment_id: str
    channel: ContentChannel
    status: str = "draft"
    subject: str | None = None
    preheader: str | None = None
    title: str | None = None
    body: str | None = None
    cta: str | None = None
    message: str | None = None
    image_prompt: str | None = None
    image_url: str | None = None
    landing_url: str | None = None
    generation_prompt: str | None = None
    reason_summary: str | None = None
    data_evidence_json: dict[str, Any] = field(default_factory=dict)
    message_strategy: str | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        channel = (
            self.channel
            if isinstance(self.channel, ContentChannel)
            else ContentChannel(self.channel)
        )
        object.__setattr__(self, "channel", channel)
        missing = missing_channel_fields(channel, self.to_public_values())
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                f"{self.channel.value} content candidate is missing required fields: "
                f"{missing_fields}"
            )

    def to_public_values(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "content_option_id": self.content_option_id,
            "segment_id": self.segment_id,
            "channel": self.channel,
            "subject": self.subject,
            "preheader": self.preheader,
            "title": self.title,
            "body": self.body,
            "cta": self.cta,
            "message": self.message,
            "image_prompt": self.image_prompt,
            "image_url": self.image_url,
            "landing_url": self.landing_url,
            "status": self.status,
        }

    def to_db_params(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "content_option_id": self.content_option_id,
            "generation_id": self.generation_id,
            "analysis_id": self.analysis_id,
            "project_id": self.project_id,
            "campaign_id": self.campaign_id,
            "promotion_id": self.promotion_id,
            "segment_id": self.segment_id,
            "channel": self.channel.value,
            "subject": self.subject,
            "preheader": self.preheader,
            "title": self.title,
            "body": self.body,
            "cta": self.cta,
            "message": self.message,
            "image_prompt": self.image_prompt,
            "image_url": self.image_url,
            "landing_url": self.landing_url,
            "generation_prompt": self.generation_prompt,
            "reason_summary": self.reason_summary,
            "data_evidence_json": Jsonb(self.data_evidence_json),
            "message_strategy": self.message_strategy,
            "metadata_json": Jsonb(self.metadata_json),
            "status": self.status,
        }


class ContentCandidateRepository:
    INSERT_SQL = """
        INSERT INTO content_candidates (
            content_id,
            content_option_id,
            generation_id,
            analysis_id,
            project_id,
            campaign_id,
            promotion_id,
            segment_id,
            channel,
            subject,
            preheader,
            title,
            body,
            cta,
            message,
            image_prompt,
            image_url,
            landing_url,
            generation_prompt,
            reason_summary,
            data_evidence_json,
            message_strategy,
            metadata_json,
            status
        )
        VALUES (
            %(content_id)s,
            %(content_option_id)s,
            %(generation_id)s,
            %(analysis_id)s,
            %(project_id)s,
            %(campaign_id)s,
            %(promotion_id)s,
            %(segment_id)s,
            %(channel)s,
            %(subject)s,
            %(preheader)s,
            %(title)s,
            %(body)s,
            %(cta)s,
            %(message)s,
            %(image_prompt)s,
            %(image_url)s,
            %(landing_url)s,
            %(generation_prompt)s,
            %(reason_summary)s,
            %(data_evidence_json)s,
            %(message_strategy)s,
            %(metadata_json)s,
            %(status)s
        )
        RETURNING
            content_id,
            content_option_id,
            generation_id,
            analysis_id,
            project_id,
            campaign_id,
            promotion_id,
            segment_id,
            channel,
            subject,
            preheader,
            title,
            body,
            cta,
            message,
            image_prompt,
            image_url,
            landing_url,
            generation_prompt,
            reason_summary,
            data_evidence_json,
            message_strategy,
            metadata_json,
            status,
            created_at,
            updated_at
    """

    SELECT_BY_GENERATION_SQL = """
        SELECT
            content_id,
            content_option_id,
            generation_id,
            analysis_id,
            project_id,
            campaign_id,
            promotion_id,
            segment_id,
            channel,
            subject,
            preheader,
            title,
            body,
            cta,
            message,
            image_prompt,
            image_url,
            landing_url,
            generation_prompt,
            reason_summary,
            data_evidence_json,
            message_strategy,
            metadata_json,
            status,
            created_at,
            updated_at
        FROM content_candidates
        WHERE generation_id = %(generation_id)s
        ORDER BY segment_id, content_option_id
    """

    def __init__(self, connection: ConnectionProtocol):
        self._connection = connection

    def create(self, record: ContentCandidateRecord) -> dict[str, Any]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(self.INSERT_SQL, record.to_db_params())
            created = cursor.fetchone()

        if created is None:
            raise RuntimeError("content_candidates insert returned no row")
        return created

    def list_by_generation(self, generation_id: str) -> list[dict[str, Any]]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.SELECT_BY_GENERATION_SQL,
                {"generation_id": generation_id},
            )
            return cursor.fetchall()
