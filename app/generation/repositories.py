from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.generation.artifacts import (
    attribution_for_candidate,
    creative_format_for_channel,
    default_artifact,
    source_for_channel,
)
from app.generation.prompt_builder import (
    PromotionPromptInput,
    TargetSegmentPromptInput,
)
from app.generation.schemas import (
    ContentChannel,
    GenerationRequest,
    missing_channel_fields,
)


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

    LIST_IDS_BY_PROMOTION_SQL = """
        SELECT generation_id
        FROM generation_runs
        WHERE promotion_id = %(promotion_id)s
        ORDER BY created_at ASC, generation_id ASC
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

    def list_ids_by_promotion(self, promotion_id: str) -> list[str]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.LIST_IDS_BY_PROMOTION_SQL,
                {"promotion_id": promotion_id},
            )
            rows = cursor.fetchall()
        return [str(row["generation_id"]) for row in rows]


class GenerationInputRepository:
    CONFIRMED_TARGET_SEGMENT_STATUS_FILTER = "AND pts.status = 'approved'"

    SELECT_PROMOTION_SQL = """
        SELECT
            project_id,
            campaign_id,
            promotion_id,
            channel,
            goal_metric,
            goal_target_value,
            goal_basis,
            message_brief,
            landing_url
        FROM promotions
        WHERE project_id = %(project_id)s
          AND campaign_id = %(campaign_id)s
          AND promotion_id = %(promotion_id)s
    """

    SELECT_TARGET_SEGMENTS_BASE_SQL = """
        SELECT
            pts.analysis_id,
            pts.promotion_id,
            pts.segment_id,
            pts.segment_name,
            pts.content_brief_json,
            pts.data_evidence_json,
            pts.segment_vector_id,
            pts.estimated_size,
            pts.priority,
            pts.status,
            sd.source AS segment_source,
            sd.query_preview_id,
            sd.natural_language_query,
            sd.generated_sql,
            sd.sample_size AS segment_sample_size,
            sd.sample_ratio AS segment_sample_ratio
        FROM promotion_target_segments pts
        LEFT JOIN segment_definitions sd
          ON sd.project_id = pts.project_id
         AND sd.segment_id = pts.segment_id
        WHERE pts.project_id = %(project_id)s
          AND pts.campaign_id = %(campaign_id)s
          AND pts.promotion_id = %(promotion_id)s
          AND pts.analysis_id = %(analysis_id)s
          {status_filter}
        ORDER BY
            CASE pts.priority
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                WHEN 'low' THEN 3
                ELSE 4
            END,
            pts.segment_id
    """

    def __init__(self, connection: ConnectionProtocol):
        self._connection = connection

    def get_promotion_input(
        self,
        request: GenerationRequest,
    ) -> PromotionPromptInput | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.SELECT_PROMOTION_SQL,
                {
                    "project_id": request.project_id,
                    "campaign_id": request.campaign_id,
                    "promotion_id": request.promotion_id,
                },
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return PromotionPromptInput(
            project_id=str(row["project_id"]),
            campaign_id=str(row["campaign_id"]),
            promotion_id=str(row["promotion_id"]),
            channel=ContentChannel(str(row["channel"])),
            goal_metric=str(row["goal_metric"]),
            goal_target_value=str(row["goal_target_value"]),
            goal_basis=str(row["goal_basis"]),
            message_brief=_optional_text(row.get("message_brief")),
            landing_url=_optional_text(row.get("landing_url")),
        )

    def list_target_segment_inputs(
        self,
        request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        return self._list_target_segment_inputs(
            request=request,
            confirmed_only=True,
        )

    def list_focus_target_segment_inputs(
        self,
        request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        return self._list_target_segment_inputs(
            request=request,
            confirmed_only=False,
        )

    def _list_target_segment_inputs(
        self,
        *,
        request: GenerationRequest,
        confirmed_only: bool,
    ) -> list[TargetSegmentPromptInput]:
        query = self.SELECT_TARGET_SEGMENTS_BASE_SQL.format(
            status_filter=(
                self.CONFIRMED_TARGET_SEGMENT_STATUS_FILTER
                if confirmed_only
                else ""
            )
        )
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                query,
                {
                    "project_id": request.project_id,
                    "campaign_id": request.campaign_id,
                    "promotion_id": request.promotion_id,
                    "analysis_id": request.analysis_id,
                },
            )
            rows = cursor.fetchall()

        return [_target_segment_prompt_input(row) for row in rows]


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
        missing = missing_channel_fields(channel, self.to_record_values())
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                f"{self.channel.value} content candidate is missing required fields: "
                f"{missing_fields}"
            )

    def to_record_values(self) -> dict[str, Any]:
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

    def to_public_values(self) -> dict[str, Any]:
        content_values = self.to_record_values()
        creative = _creative_metadata(self.metadata_json)
        return {
            "channel": self.channel,
            "creative_format": creative_format_for_channel(self.channel),
            "attribution": attribution_for_candidate(
                project_id=self.project_id,
                campaign_id=self.campaign_id,
                promotion_id=self.promotion_id,
                segment_id=self.segment_id,
                content_id=self.content_id,
                content_option_id=self.content_option_id,
                channel=self.channel,
                target_url=str(self.landing_url or ""),
            ),
            "source": creative.get("source")
            or source_for_channel(channel=self.channel, content_values=content_values),
            "artifact": creative.get("artifact") or default_artifact(self.channel),
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

    UPDATE_IMAGE_URL_SQL = """
        UPDATE content_candidates
        SET
            image_url = %(image_url)s::text,
            metadata_json = COALESCE(metadata_json, '{}'::jsonb) ||
                jsonb_build_object(
                    'image_url', %(image_url)s::text,
                    'image_generation_status', 'completed'
                ),
            updated_at = now()
        WHERE content_id = %(content_id)s
        RETURNING content_id
    """

    MARK_IMAGE_GENERATION_FAILED_SQL = """
        UPDATE content_candidates
        SET
            metadata_json = COALESCE(metadata_json, '{}'::jsonb) ||
                jsonb_build_object(
                    'image_generation_status', 'failed',
                    'image_generation_error_code', %(error_code)s::text
                ),
            updated_at = now()
        WHERE content_id = %(content_id)s
        RETURNING content_id
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

    def update_image_url(self, *, content_id: str, image_url: str) -> dict[str, Any]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.UPDATE_IMAGE_URL_SQL,
                {"content_id": content_id, "image_url": image_url},
            )
            updated = cursor.fetchone()

        if updated is None:
            raise RuntimeError("content_candidates image_url update returned no row")
        return updated

    def mark_image_generation_failed(
        self,
        *,
        content_id: str,
        error_code: str,
    ) -> dict[str, Any]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.MARK_IMAGE_GENERATION_FAILED_SQL,
                {"content_id": content_id, "error_code": error_code},
            )
            updated = cursor.fetchone()

        if updated is None:
            raise RuntimeError(
                "content_candidates image generation failure update returned no row"
            )
        return updated

    def list_by_generation(self, generation_id: str) -> list[dict[str, Any]]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.SELECT_BY_GENERATION_SQL,
                {"generation_id": generation_id},
            )
            return cursor.fetchall()


def _target_segment_prompt_input(
    row: Mapping[str, Any],
) -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id=str(row["analysis_id"]),
        promotion_id=str(row["promotion_id"]),
        segment_id=str(row["segment_id"]),
        segment_name=str(row.get("segment_name") or row["segment_id"]),
        content_brief_json=_content_brief_json(row),
        segment_vector_id=_optional_text(row.get("segment_vector_id")),
        estimated_size=_positive_int(
            row.get("estimated_size"),
            fallback=_positive_int(row.get("segment_sample_size")),
        ),
        priority=_optional_text(row.get("priority")),
        natural_language_query=_optional_text(row.get("natural_language_query")),
        generated_sql=_optional_text(row.get("generated_sql")),
        sample_ratio=(
            _optional_text(row.get("segment_sample_ratio"))
            or _optional_text(
                _json_object(row.get("data_evidence_json")).get("sample_ratio")
            )
        ),
        source=_optional_text(row.get("segment_source"))
        or _optional_text(_json_object(row.get("data_evidence_json")).get("source")),
        query_preview_id=_optional_text(row.get("query_preview_id")),
        status=_optional_text(row.get("status")),
    )


def _content_brief_json(row: Mapping[str, Any]) -> dict[str, Any]:
    content_brief = _json_object(row.get("content_brief_json"))
    data_evidence = _json_object(row.get("data_evidence_json"))
    for key in (
        "booking_conversion_rate",
        "comparison_group_conversion_rate",
        "top_common_features",
        "keywords",
    ):
        if key not in content_brief and key in data_evidence:
            content_brief[key] = data_evidence[key]
    return content_brief


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _creative_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    creative = value.get("creative") if isinstance(value, Mapping) else None
    return creative if isinstance(creative, Mapping) else {}


def _positive_int(value: object, *, fallback: int = 0) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return max(fallback, 0)
    return max(number, 0)
