from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.content_brief import CONTENT_BRIEF_SCHEMA_VERSION
from app.generation.artifacts import (
    attribution_for_candidate,
    creative_format_for_channel,
    default_artifact,
    source_for_channel,
)
from app.generation.prompt_builder import (
    PromotionOfferLink,
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
    "started_at",
    "finished_at",
    "retry_count",
    "next_retry_at",
    "last_error_code",
    "last_error_message",
    "worker_id",
    "lease_token",
    "heartbeat_at",
    "lease_expires_at",
    "idempotency_key",
    "request_fingerprint",
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
    "creative_format",
    "image_generation_status",
    "artifact_status",
    "artifact_storage_key",
    "artifact_public_url",
    "artifact_sha256",
    "artifact_content_type",
    "artifact_error_code",
    "artifact_published_at",
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


class GenerationIdempotencyMismatch(ValueError):
    """An idempotency key already exists with another request fingerprint."""


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
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retry_count: int = 0
    next_retry_at: datetime | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    worker_id: str | None = None
    lease_token: UUID | None = None
    heartbeat_at: datetime | None = None
    lease_expires_at: datetime | None = None
    idempotency_key: str | None = None
    request_fingerprint: str | None = None

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
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "retry_count": self.retry_count,
            "next_retry_at": self.next_retry_at,
            "last_error_code": self.last_error_code,
            "last_error_message": self.last_error_message,
            "worker_id": self.worker_id,
            "lease_token": self.lease_token,
            "heartbeat_at": self.heartbeat_at,
            "lease_expires_at": self.lease_expires_at,
            "idempotency_key": self.idempotency_key,
            "request_fingerprint": self.request_fingerprint,
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
            status,
            started_at,
            finished_at,
            retry_count,
            next_retry_at,
            last_error_code,
            last_error_message,
            worker_id,
            lease_token,
            heartbeat_at,
            lease_expires_at,
            idempotency_key,
            request_fingerprint
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
            %(status)s,
            %(started_at)s,
            CASE
                WHEN %(status)s::varchar IN ('completed', 'failed')
                THEN GREATEST(
                    COALESCE(%(finished_at)s::timestamptz, now()),
                    now()
                )
                ELSE %(finished_at)s::timestamptz
            END,
            %(retry_count)s,
            %(next_retry_at)s,
            %(last_error_code)s,
            %(last_error_message)s,
            %(worker_id)s,
            %(lease_token)s,
            %(heartbeat_at)s,
            %(lease_expires_at)s,
            %(idempotency_key)s,
            %(request_fingerprint)s
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
            updated_at,
            started_at,
            finished_at,
            retry_count,
            next_retry_at,
            last_error_code,
            last_error_message,
            worker_id,
            lease_token,
            heartbeat_at,
            lease_expires_at,
            idempotency_key,
            request_fingerprint
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
            updated_at,
            started_at,
            finished_at,
            retry_count,
            next_retry_at,
            last_error_code,
            last_error_message,
            worker_id,
            lease_token,
            heartbeat_at,
            lease_expires_at,
            idempotency_key,
            request_fingerprint
        FROM generation_runs
        WHERE generation_id = %(generation_id)s
    """

    SELECT_BY_IDEMPOTENCY_SQL = """
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
            updated_at,
            started_at,
            finished_at,
            retry_count,
            next_retry_at,
            last_error_code,
            last_error_message,
            worker_id,
            lease_token,
            heartbeat_at,
            lease_expires_at,
            idempotency_key,
            request_fingerprint
        FROM generation_runs
        WHERE project_id = %(project_id)s
          AND idempotency_key = %(idempotency_key)s
    """

    INSERT_IDEMPOTENT_SQL = INSERT_SQL.replace(
        "RETURNING\n",
        "ON CONFLICT (project_id, idempotency_key)\n"
        "        WHERE idempotency_key IS NOT NULL\n"
        "        DO NOTHING\n"
        "        RETURNING\n",
        1,
    )

    CLAIM_NEXT_SQL = """
        WITH next_job AS (
            SELECT generation_id
            FROM generation_runs
            WHERE status = 'requested'
              AND (next_retry_at IS NULL OR next_retry_at <= now())
            ORDER BY
                COALESCE(next_retry_at, created_at),
                created_at,
                generation_id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE generation_runs AS run
        SET
            status = 'running',
            started_at = COALESCE(run.started_at, now()),
            finished_at = NULL,
            next_retry_at = NULL,
            worker_id = %(worker_id)s,
            lease_token = %(lease_token)s,
            heartbeat_at = now(),
            lease_expires_at = now()
                + make_interval(secs => %(lease_seconds)s),
            updated_at = now()
        FROM next_job
        WHERE run.generation_id = next_job.generation_id
        RETURNING run.*
    """

    HEARTBEAT_SQL = """
        UPDATE generation_runs
        SET
            heartbeat_at = now(),
            lease_expires_at = now()
                + make_interval(secs => %(lease_seconds)s),
            updated_at = now()
        WHERE generation_id = %(generation_id)s
          AND status = 'running'
          AND worker_id = %(worker_id)s
          AND lease_token = %(lease_token)s
          AND lease_expires_at > now()
        RETURNING generation_id
    """

    RECOVER_EXPIRED_SQL = """
        WITH expired AS (
            SELECT generation_id, retry_count
            FROM generation_runs
            WHERE status = 'running'
              AND lease_expires_at <= now()
            ORDER BY lease_expires_at, generation_id
            FOR UPDATE SKIP LOCKED
            LIMIT %(limit)s
        )
        UPDATE generation_runs AS run
        SET
            status = CASE
                WHEN expired.retry_count < %(max_retries)s
                THEN 'requested'
                ELSE 'failed'
            END,
            finished_at = CASE
                WHEN expired.retry_count < %(max_retries)s
                THEN NULL
                ELSE now()
            END,
            retry_count = CASE
                WHEN expired.retry_count < %(max_retries)s
                THEN expired.retry_count + 1
                ELSE expired.retry_count
            END,
            next_retry_at = CASE
                WHEN expired.retry_count < %(max_retries)s
                THEN now() + make_interval(
                    secs => (%(retry_backoff_seconds)s::integer[])[LEAST(
                        expired.retry_count + 1,
                        cardinality(%(retry_backoff_seconds)s::integer[])
                    )]
                )
                ELSE NULL
            END,
            last_error_code = 'generation_lease_expired',
            last_error_message = 'generation worker lease expired',
            worker_id = NULL,
            lease_token = NULL,
            heartbeat_at = NULL,
            lease_expires_at = NULL,
            updated_at = now()
        FROM expired
        WHERE run.generation_id = expired.generation_id
        RETURNING run.*
    """

    SCHEDULE_RETRY_FENCED_SQL = """
        UPDATE generation_runs
        SET
            status = 'requested',
            finished_at = NULL,
            retry_count = retry_count + 1,
            next_retry_at = %(next_retry_at)s,
            last_error_code = %(error_code)s,
            last_error_message = %(error_message)s,
            worker_id = NULL,
            lease_token = NULL,
            heartbeat_at = NULL,
            lease_expires_at = NULL,
            updated_at = now()
        WHERE generation_id = %(generation_id)s
          AND status = 'running'
          AND worker_id = %(worker_id)s
          AND lease_token = %(lease_token)s
          AND lease_expires_at > now()
        RETURNING generation_id
    """

    MARK_FAILED_FENCED_SQL = """
        UPDATE generation_runs
        SET
            status = 'failed',
            finished_at = now(),
            next_retry_at = NULL,
            last_error_code = %(error_code)s,
            last_error_message = %(error_message)s,
            worker_id = NULL,
            lease_token = NULL,
            heartbeat_at = NULL,
            lease_expires_at = NULL,
            updated_at = now()
        WHERE generation_id = %(generation_id)s
          AND status = 'running'
          AND worker_id = %(worker_id)s
          AND lease_token = %(lease_token)s
          AND lease_expires_at > now()
        RETURNING generation_id
    """

    COMPLETE_IF_READY_FENCED_SQL = """
        WITH locked_run AS MATERIALIZED (
            SELECT
                generation_id,
                content_option_count,
                input_json
            FROM generation_runs
            WHERE generation_id = %(generation_id)s
              AND status = 'running'
              AND worker_id = %(worker_id)s
              AND lease_token = %(lease_token)s
              AND lease_expires_at > now()
            FOR UPDATE
        ), target_values AS MATERIALIZED (
            SELECT target.value
            FROM locked_run AS run
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(run.input_json -> 'target_segments') = 'array'
                    THEN run.input_json -> 'target_segments'
                    ELSE '[]'::jsonb
                END
            ) AS target(value)
        ), expected_segments AS MATERIALIZED (
            SELECT btrim(value ->> 'segment_id') AS segment_id
            FROM target_values
            WHERE jsonb_typeof(value) = 'object'
              AND NULLIF(btrim(value ->> 'segment_id'), '') IS NOT NULL
        ), ready_run AS (
            SELECT run.generation_id
            FROM locked_run AS run
            WHERE run.input_json ->> 'schema_version' = 'generation.request.v1'
              AND jsonb_typeof(run.input_json -> 'target_segments') = 'array'
              AND jsonb_array_length(
                    CASE
                        WHEN jsonb_typeof(
                            run.input_json -> 'target_segments'
                        ) = 'array'
                        THEN run.input_json -> 'target_segments'
                        ELSE '[]'::jsonb
                    END
              ) > 0
              AND NOT EXISTS (
                    SELECT 1
                    FROM target_values
                    WHERE jsonb_typeof(value) <> 'object'
                       OR NULLIF(btrim(value ->> 'segment_id'), '') IS NULL
              )
              AND (
                    SELECT count(*) = count(DISTINCT segment_id)
                    FROM expected_segments
              )
              AND EXISTS (
                    SELECT 1
                    FROM content_candidates AS candidate
                    WHERE candidate.generation_id = run.generation_id
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM expected_segments AS expected
                    WHERE (
                        SELECT count(*)
                        FROM content_candidates AS candidate
                        WHERE candidate.generation_id = run.generation_id
                          AND candidate.segment_id = expected.segment_id
                    ) <> run.content_option_count
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM content_candidates AS candidate
                    WHERE candidate.generation_id = run.generation_id
                      AND NOT EXISTS (
                            SELECT 1
                            FROM expected_segments AS expected
                            WHERE expected.segment_id = candidate.segment_id
                      )
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM content_candidates AS candidate
                    WHERE candidate.generation_id = run.generation_id
                      AND (
                            (
                                candidate.channel = 'sms'
                                AND candidate.creative_format = 'sms_text'
                                AND candidate.message IS NOT NULL
                                AND candidate.image_generation_status = 'not_required'
                                AND candidate.artifact_status = 'not_required'
                            )
                            OR (
                                candidate.channel IN ('email', 'onsite_banner')
                                AND candidate.creative_format = CASE candidate.channel
                                    WHEN 'email' THEN 'email_html'
                                    ELSE 'banner_html'
                                END
                                AND candidate.image_generation_status = 'completed'
                                AND candidate.image_url IS NOT NULL
                                AND candidate.artifact_status = 'published'
                                AND candidate.artifact_storage_key IS NOT NULL
                                AND candidate.artifact_public_url IS NOT NULL
                                AND candidate.artifact_sha256 IS NOT NULL
                                AND candidate.artifact_content_type IS NOT NULL
                                AND candidate.artifact_published_at IS NOT NULL
                                AND candidate.created_at
                                    <= candidate.artifact_published_at
                                AND candidate.artifact_published_at <= now()
                            )
                      ) IS NOT TRUE
              )
        )
        UPDATE generation_runs AS run
        SET
            output_json = %(output_json)s,
            generation_report_json = %(generation_report_json)s,
            status = 'completed',
            finished_at = now(),
            next_retry_at = NULL,
            last_error_code = NULL,
            last_error_message = NULL,
            worker_id = NULL,
            lease_token = NULL,
            heartbeat_at = NULL,
            lease_expires_at = NULL,
            updated_at = now()
        FROM ready_run
        WHERE run.generation_id = ready_run.generation_id
        RETURNING run.*
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

    def create_or_get_idempotent(
        self,
        record: GenerationRunRecord,
    ) -> tuple[dict[str, Any], bool]:
        if not record.idempotency_key:
            raise ValueError("idempotency_key is required for a Generation v1 request")
        if not record.request_fingerprint:
            raise ValueError(
                "request_fingerprint is required for an idempotent Generation request"
            )

        params = record.to_db_params()
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(self.INSERT_IDEMPOTENT_SQL, params)
            created = cursor.fetchone()
            if created is not None:
                return created, True

            cursor.execute(
                self.SELECT_BY_IDEMPOTENCY_SQL,
                {
                    "project_id": record.project_id,
                    "idempotency_key": record.idempotency_key,
                },
            )
            existing = cursor.fetchone()

        if existing is None:
            raise RuntimeError(
                "generation idempotency conflict returned no existing row"
            )
        if existing.get("request_fingerprint") != record.request_fingerprint:
            raise GenerationIdempotencyMismatch(
                "idempotency_key request_fingerprint conflicts with a different request"
            )
        return existing, False

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_token: UUID,
        lease_seconds: int,
    ) -> dict[str, Any] | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.CLAIM_NEXT_SQL,
                {
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "lease_seconds": lease_seconds,
                },
            )
            return cursor.fetchone()

    def heartbeat(
        self,
        *,
        generation_id: str,
        worker_id: str,
        lease_token: UUID,
        lease_seconds: int,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.HEARTBEAT_SQL,
                {
                    "generation_id": generation_id,
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "lease_seconds": lease_seconds,
                },
            )
            return cursor.fetchone() is not None

    def recover_expired(
        self,
        *,
        max_retries: int,
        retry_backoff_seconds: Sequence[int],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        backoffs = list(retry_backoff_seconds)
        if not backoffs or any(seconds <= 0 for seconds in backoffs):
            raise ValueError("retry_backoff_seconds must contain positive values")
        if limit <= 0:
            raise ValueError("limit must be positive")

        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.RECOVER_EXPIRED_SQL,
                {
                    "max_retries": max_retries,
                    "retry_backoff_seconds": backoffs,
                    "limit": limit,
                },
            )
            return cursor.fetchall()

    def schedule_retry_fenced(
        self,
        *,
        generation_id: str,
        worker_id: str,
        lease_token: UUID,
        next_retry_at: datetime,
        error_code: str,
        error_message: str,
    ) -> bool:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.SCHEDULE_RETRY_FENCED_SQL,
                {
                    "generation_id": generation_id,
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "next_retry_at": next_retry_at,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
            return cursor.fetchone() is not None

    def mark_failed_fenced(
        self,
        *,
        generation_id: str,
        worker_id: str,
        lease_token: UUID,
        error_code: str,
        error_message: str,
    ) -> bool:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.MARK_FAILED_FENCED_SQL,
                {
                    "generation_id": generation_id,
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
            return cursor.fetchone() is not None

    def complete_if_ready_fenced(
        self,
        *,
        generation_id: str,
        worker_id: str,
        lease_token: UUID,
        output_json: Mapping[str, Any],
        generation_report_json: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.COMPLETE_IF_READY_FENCED_SQL,
                {
                    "generation_id": generation_id,
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "output_json": Jsonb(dict(output_json)),
                    "generation_report_json": Jsonb(dict(generation_report_json)),
                },
            )
            return cursor.fetchone()

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
    REQUESTED_SEGMENT_IDS_FILTER = "AND pts.segment_id = ANY(%(segment_ids)s)"

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
            landing_url,
            offer_type,
            landing_type,
            metadata_json
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
            offer_type=_optional_text(row.get("offer_type")),
            landing_type=_optional_text(row.get("landing_type")),
            offer_links=_promotion_offer_links(row.get("metadata_json")),
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
        filters = [
            self.CONFIRMED_TARGET_SEGMENT_STATUS_FILTER if confirmed_only else "",
            (
                self.REQUESTED_SEGMENT_IDS_FILTER
                if confirmed_only and request.segment_ids is not None
                else ""
            ),
        ]
        query = self.SELECT_TARGET_SEGMENTS_BASE_SQL.format(
            status_filter="\n          ".join(filter(None, filters)),
        )
        params: dict[str, Any] = {
            "project_id": request.project_id,
            "campaign_id": request.campaign_id,
            "promotion_id": request.promotion_id,
            "analysis_id": request.analysis_id,
        }
        if confirmed_only and request.segment_ids is not None:
            params["segment_ids"] = request.segment_ids
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
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
    creative_format: str | None = None
    image_generation_status: str | None = None
    artifact_status: str | None = None
    artifact_storage_key: str | None = None
    artifact_public_url: str | None = None
    artifact_sha256: str | None = None
    artifact_content_type: str | None = None
    artifact_error_code: str | None = None
    artifact_published_at: datetime | None = None

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
            "source": _public_source_metadata(
                channel=self.channel,
                content_values=content_values,
                creative=creative,
            ),
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
            "creative_format": self.creative_format,
            "image_generation_status": self.image_generation_status,
            "artifact_status": self.artifact_status,
            "artifact_storage_key": self.artifact_storage_key,
            "artifact_public_url": self.artifact_public_url,
            "artifact_sha256": self.artifact_sha256,
            "artifact_content_type": self.artifact_content_type,
            "artifact_error_code": self.artifact_error_code,
            "artifact_published_at": self.artifact_published_at,
        }


def content_candidate_record_from_row(
    row: Mapping[str, Any],
) -> ContentCandidateRecord:
    return ContentCandidateRecord(
        content_id=str(row["content_id"]),
        content_option_id=str(row["content_option_id"]),
        generation_id=str(row["generation_id"]),
        analysis_id=str(row["analysis_id"]),
        project_id=str(row["project_id"]),
        campaign_id=str(row["campaign_id"]),
        promotion_id=str(row["promotion_id"]),
        segment_id=str(row["segment_id"]),
        channel=ContentChannel(str(row["channel"])),
        status=str(row.get("status") or "draft"),
        subject=_optional_text(row.get("subject")),
        preheader=_optional_text(row.get("preheader")),
        title=_optional_text(row.get("title")),
        body=_optional_text(row.get("body")),
        cta=_optional_text(row.get("cta")),
        message=_optional_text(row.get("message")),
        image_prompt=_optional_text(row.get("image_prompt")),
        image_url=_optional_text(row.get("image_url")),
        landing_url=_optional_text(row.get("landing_url")),
        generation_prompt=_optional_text(row.get("generation_prompt")),
        reason_summary=_optional_text(row.get("reason_summary")),
        data_evidence_json=dict(row.get("data_evidence_json") or {}),
        message_strategy=_optional_text(row.get("message_strategy")),
        metadata_json=dict(row.get("metadata_json") or {}),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        creative_format=_optional_text(row.get("creative_format")),
        image_generation_status=_optional_text(
            row.get("image_generation_status")
        ),
        artifact_status=_optional_text(row.get("artifact_status")),
        artifact_storage_key=_optional_text(row.get("artifact_storage_key")),
        artifact_public_url=_optional_text(row.get("artifact_public_url")),
        artifact_sha256=_optional_text(row.get("artifact_sha256")),
        artifact_content_type=_optional_text(row.get("artifact_content_type")),
        artifact_error_code=_optional_text(row.get("artifact_error_code")),
        artifact_published_at=row.get("artifact_published_at"),
    )


class ContentCandidateRepository:
    INSERT_SQL = """
        WITH write_clock AS MATERIALIZED (
            SELECT now() AS db_now
        )
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
            status,
            creative_format,
            image_generation_status,
            artifact_status,
            artifact_storage_key,
            artifact_public_url,
            artifact_sha256,
            artifact_content_type,
            artifact_error_code,
            artifact_published_at
        )
        SELECT
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
            CASE
                WHEN %(artifact_status)s::varchar = 'published'
                THEN COALESCE(%(metadata_json)s::jsonb, '{}'::jsonb)
                    || jsonb_build_object(
                        'creative',
                        COALESCE(
                            %(metadata_json)s::jsonb -> 'creative',
                            '{}'::jsonb
                        )
                        || jsonb_build_object(
                            'artifact',
                            COALESCE(
                                %(metadata_json)s::jsonb
                                    #> '{creative,artifact}',
                                '{}'::jsonb
                            )
                            || jsonb_build_object(
                                'published_at',
                                write_clock.db_now
                            )
                        )
                    )
                ELSE COALESCE(%(metadata_json)s::jsonb, '{}'::jsonb)
                    #- '{creative,artifact,published_at}'
            END,
            %(status)s,
            %(creative_format)s,
            %(image_generation_status)s,
            %(artifact_status)s::varchar,
            %(artifact_storage_key)s,
            %(artifact_public_url)s,
            %(artifact_sha256)s,
            %(artifact_content_type)s,
            %(artifact_error_code)s,
            CASE
                WHEN %(artifact_status)s::varchar = 'published'
                THEN write_clock.db_now
                ELSE NULL::timestamptz
            END
        FROM write_clock
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
            updated_at,
            creative_format,
            image_generation_status,
            artifact_status,
            artifact_storage_key,
            artifact_public_url,
            artifact_sha256,
            artifact_content_type,
            artifact_error_code,
            artifact_published_at
    """

    UPSERT_FENCED_SQL = """
        WITH fenced_run AS MATERIALIZED (
            SELECT generation_id
            FROM generation_runs
            WHERE generation_id = %(generation_id)s
              AND status = 'running'
              AND worker_id = %(worker_id)s
              AND lease_token = %(lease_token)s
              AND lease_expires_at > now()
            FOR UPDATE
        ),
        write_clock AS MATERIALIZED (
            SELECT now() AS db_now
        )
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
            status,
            creative_format,
            image_generation_status,
            artifact_status,
            artifact_storage_key,
            artifact_public_url,
            artifact_sha256,
            artifact_content_type,
            artifact_error_code,
            artifact_published_at
        )
        SELECT
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
            CASE
                WHEN %(artifact_status)s::varchar = 'published'
                THEN COALESCE(%(metadata_json)s::jsonb, '{}'::jsonb)
                    || jsonb_build_object(
                        'creative',
                        COALESCE(
                            %(metadata_json)s::jsonb -> 'creative',
                            '{}'::jsonb
                        )
                        || jsonb_build_object(
                            'artifact',
                            COALESCE(
                                %(metadata_json)s::jsonb
                                    #> '{creative,artifact}',
                                '{}'::jsonb
                            )
                            || jsonb_build_object(
                                'published_at',
                                write_clock.db_now
                            )
                        )
                    )
                ELSE COALESCE(%(metadata_json)s::jsonb, '{}'::jsonb)
                    #- '{creative,artifact,published_at}'
            END,
            %(status)s,
            %(creative_format)s,
            %(image_generation_status)s,
            %(artifact_status)s::varchar,
            %(artifact_storage_key)s,
            %(artifact_public_url)s,
            %(artifact_sha256)s,
            %(artifact_content_type)s,
            %(artifact_error_code)s,
            CASE
                WHEN %(artifact_status)s::varchar = 'published'
                THEN write_clock.db_now
                ELSE NULL::timestamptz
            END
        FROM fenced_run
        CROSS JOIN write_clock
        ON CONFLICT (generation_id, segment_id, content_option_id)
        DO UPDATE SET
            subject = EXCLUDED.subject,
            preheader = EXCLUDED.preheader,
            title = EXCLUDED.title,
            body = EXCLUDED.body,
            cta = EXCLUDED.cta,
            message = EXCLUDED.message,
            image_prompt = EXCLUDED.image_prompt,
            image_url = EXCLUDED.image_url,
            landing_url = EXCLUDED.landing_url,
            generation_prompt = EXCLUDED.generation_prompt,
            reason_summary = EXCLUDED.reason_summary,
            data_evidence_json = EXCLUDED.data_evidence_json,
            message_strategy = EXCLUDED.message_strategy,
            metadata_json = CASE
                WHEN EXCLUDED.artifact_status = 'published'
                THEN EXCLUDED.metadata_json
                    || jsonb_build_object(
                        'creative',
                        COALESCE(
                            EXCLUDED.metadata_json -> 'creative',
                            '{}'::jsonb
                        )
                        || jsonb_build_object(
                            'artifact',
                            COALESCE(
                                EXCLUDED.metadata_json
                                    #> '{creative,artifact}',
                                '{}'::jsonb
                            )
                            || jsonb_build_object(
                                'published_at',
                                COALESCE(
                                    content_candidates.artifact_published_at,
                                    EXCLUDED.artifact_published_at
                                )
                            )
                        )
                    )
                ELSE EXCLUDED.metadata_json
                    #- '{creative,artifact,published_at}'
            END,
            status = EXCLUDED.status,
            creative_format = EXCLUDED.creative_format,
            image_generation_status = EXCLUDED.image_generation_status,
            artifact_status = EXCLUDED.artifact_status,
            artifact_storage_key = EXCLUDED.artifact_storage_key,
            artifact_public_url = EXCLUDED.artifact_public_url,
            artifact_sha256 = EXCLUDED.artifact_sha256,
            artifact_content_type = EXCLUDED.artifact_content_type,
            artifact_error_code = EXCLUDED.artifact_error_code,
            artifact_published_at = CASE
                WHEN EXCLUDED.artifact_status = 'published'
                THEN COALESCE(
                    content_candidates.artifact_published_at,
                    EXCLUDED.artifact_published_at
                )
                ELSE NULL::timestamptz
            END,
            updated_at = now()
        WHERE content_candidates.content_id = EXCLUDED.content_id
          AND content_candidates.analysis_id = EXCLUDED.analysis_id
          AND content_candidates.project_id = EXCLUDED.project_id
          AND content_candidates.campaign_id = EXCLUDED.campaign_id
          AND content_candidates.promotion_id = EXCLUDED.promotion_id
          AND content_candidates.channel = EXCLUDED.channel
        RETURNING content_candidates.*
    """

    UPDATE_IMAGE_URL_SQL = """
        UPDATE content_candidates
        SET
            image_url = %(image_url)s::text,
            image_generation_status = 'completed',
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
            image_generation_status = 'failed',
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
            updated_at,
            creative_format,
            image_generation_status,
            artifact_status,
            artifact_storage_key,
            artifact_public_url,
            artifact_sha256,
            artifact_content_type,
            artifact_error_code,
            artifact_published_at
        FROM content_candidates
        WHERE generation_id = %(generation_id)s
        ORDER BY segment_id, content_option_id, content_id
    """

    SELECT_BY_GENERATION_FOR_UPDATE_SQL = """
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
        ORDER BY segment_id, content_option_id, content_id
        FOR UPDATE
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

    def upsert_fenced(
        self,
        record: ContentCandidateRecord,
        *,
        worker_id: str,
        lease_token: UUID,
    ) -> dict[str, Any] | None:
        params = {
            **record.to_db_params(),
            "worker_id": worker_id,
            "lease_token": lease_token,
        }
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(self.UPSERT_FENCED_SQL, params)
            return cursor.fetchone()

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

    def list_by_generation_for_update(
        self,
        generation_id: str,
    ) -> list[dict[str, Any]]:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.SELECT_BY_GENERATION_FOR_UPDATE_SQL,
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
        source_content_brief_json=_json_object(row.get("content_brief_json")),
        data_evidence_json=_json_object(row.get("data_evidence_json")),
    )


def _content_brief_json(row: Mapping[str, Any]) -> dict[str, Any]:
    content_brief = _json_object(row.get("content_brief_json"))
    if content_brief.get("schema_version") == CONTENT_BRIEF_SCHEMA_VERSION:
        return content_brief
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


def _promotion_offer_links(value: object) -> tuple[PromotionOfferLink, ...]:
    metadata = _json_object(value)
    raw_links = metadata.get("offer_links")
    if raw_links is None:
        return ()
    if not isinstance(raw_links, list):
        raise ValueError("promotion metadata_json.offer_links must be an array")
    links: list[PromotionOfferLink] = []
    for raw_link in raw_links:
        if not isinstance(raw_link, Mapping):
            raise ValueError("promotion offer_links entries must be objects")
        links.append(
            PromotionOfferLink(
                offer_id=str(raw_link.get("offer_id") or ""),
                destination_url=str(raw_link.get("destination_url") or ""),
            )
        )
    if len(links) > 8:
        raise ValueError("promotion offer_links must contain at most 8 items")
    offer_ids = [link.offer_id for link in links]
    if len(offer_ids) != len(set(offer_ids)):
        raise ValueError("promotion offer_links must not contain duplicate offer_id")
    return tuple(links)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _creative_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    creative = value.get("creative") if isinstance(value, Mapping) else None
    return creative if isinstance(creative, Mapping) else {}


def _public_source_metadata(
    *,
    channel: ContentChannel,
    content_values: Mapping[str, Any],
    creative: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = creative.get("source")
    if isinstance(source, Mapping):
        public_source = dict(source)
        public_source.pop("html_body", None)
        if public_source:
            return public_source
    return source_for_channel(channel=channel, content_values=content_values)


def _positive_int(value: object, *, fallback: int = 0) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return max(fallback, 0)
    return max(number, 0)
