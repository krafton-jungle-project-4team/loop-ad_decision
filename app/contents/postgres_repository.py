from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from app.contents.repository import GenerationLock, GenerationLockUnavailable
from app.contents.types import (
    ACTION_STATUS_CONTENT_GENERATED,
    ACTION_STATUS_FAILED,
    GeneratedContentDraft,
    GeneratedContentRecord,
    RecommendationActionTarget,
    SegmentContext,
)


class CursorLike(Protocol):
    rowcount: int

    def fetchone(self) -> Any:
        ...

    def fetchall(self) -> list[Any]:
        ...


class PostgresConnectionLike(Protocol):
    def execute(self, query: str, params: dict[str, Any] | None = None) -> CursorLike:
        ...


@dataclass
class PostgresGenerationLock(AbstractContextManager[None]):
    connection: PostgresConnectionLike
    project_id: int | str
    recommendation_action_id: int
    lock_key: int | None = None

    def __enter__(self) -> None:
        self.lock_key = advisory_lock_key(
            self.project_id,
            self.recommendation_action_id,
        )
        cursor = self.connection.execute(
            "SELECT pg_try_advisory_lock(%(lock_key)s) AS acquired",
            {"lock_key": self.lock_key},
        )
        row = cursor.fetchone()
        if row is None or not bool(_row_get(row, "acquired")):
            raise GenerationLockUnavailable(
                f"content generation already running for recommendation_action_id={self.recommendation_action_id}"
            )
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.lock_key is not None:
            self.connection.execute(
                "SELECT pg_advisory_unlock(%(lock_key)s)",
                {"lock_key": self.lock_key},
            )
        return None


class PostgresContentRepository:
    """PostgreSQL implementation for the content generation repository boundary.

    The connection should return dict-like rows. With psycopg v3, create it with
    `row_factory=psycopg.rows.dict_row`.
    """

    def __init__(self, connection: PostgresConnectionLike) -> None:
        self.connection = connection

    def list_generation_targets(
        self,
        *,
        project_id: int | str,
        analysis_date: str,
        eligible_statuses: tuple[str, ...],
    ) -> Iterable[RecommendationActionTarget]:
        cursor = self.connection.execute(
            """
            SELECT
                ra.id AS recommendation_action_id,
                ra.project_id,
                ra.recommendation_result_id,
                ra.action_key,
                ra.status AS action_status,
                COALESCE(ra.metadata->>'action_type', ac.default_channel, 'banner') AS action_type,
                ra.title AS action_title,
                ra.description AS action_description,
                COALESCE(ra.metadata->>'content_type', ac.default_channel, 'banner') AS content_type,
                COALESCE(ra.metadata, '{}'::jsonb) AS action_metadata,
                rr.analysis_date,
                s.id AS segment_id,
                s.segment_key,
                s.name AS segment_name,
                s.is_default AS segment_is_default,
                COALESCE(s.description, '') AS segment_description,
                COALESCE(s.rule_json, '{}'::jsonb) AS segment_attributes,
                COALESCE(to_jsonb(sdm) - 'created_at' - 'updated_at', '{}'::jsonb) AS metrics,
                COALESCE(rc.root_cause, '{}'::jsonb) AS root_cause
            FROM recommendation_actions ra
            JOIN recommendation_results rr
                ON rr.id = ra.recommendation_result_id
            JOIN segments s
                ON s.id = rr.segment_id
            LEFT JOIN action_catalog ac
                ON ac.id = ra.action_catalog_id
            LEFT JOIN segment_daily_metrics sdm
                ON sdm.project_id = ra.project_id
                AND sdm.segment_id = s.id
                AND sdm.analysis_date = rr.analysis_date
            LEFT JOIN LATERAL (
                SELECT to_jsonb(rcc) - 'created_at' - 'updated_at' AS root_cause
                FROM root_cause_candidates rcc
                WHERE rcc.anomaly_id = rr.anomaly_id
                ORDER BY rcc.id
                LIMIT 1
            ) rc ON TRUE
            WHERE ra.project_id = %(project_id)s
                AND rr.analysis_date = %(analysis_date)s
                AND ra.status = ANY(%(eligible_statuses)s)
                AND s.is_default = false
                AND rr.anomaly_id IS NOT NULL
            ORDER BY ra.id
            """,
            {
                "project_id": project_id,
                "analysis_date": analysis_date,
                "eligible_statuses": list(eligible_statuses),
            },
        )
        return [self._target_from_row(row) for row in cursor.fetchall()]

    def generation_lock(
        self,
        *,
        project_id: int | str,
        recommendation_action_id: int,
    ) -> GenerationLock:
        return PostgresGenerationLock(
            connection=self.connection,
            project_id=project_id,
            recommendation_action_id=recommendation_action_id,
        )

    def get_generated_content(
        self,
        *,
        project_id: int | str,
        recommendation_action_id: int,
        variant_key: str,
    ) -> GeneratedContentRecord | None:
        cursor = self.connection.execute(
            """
            SELECT
                id,
                project_id,
                recommendation_action_id,
                segment_id,
                variant_key,
                generation_status,
                created_run_id,
                COALESCE(metadata, '{}'::jsonb) AS metadata
            FROM generated_contents
            WHERE project_id = %(project_id)s
                AND recommendation_action_id = %(recommendation_action_id)s
                AND variant_key = %(variant_key)s
            """,
            {
                "project_id": project_id,
                "recommendation_action_id": recommendation_action_id,
                "variant_key": variant_key,
            },
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def upsert_generated_content(
        self,
        *,
        draft: GeneratedContentDraft,
        force: bool,
    ) -> GeneratedContentRecord:
        if force:
            return self._force_upsert_generated_content(draft)
        inserted = self._insert_generated_content_if_absent(draft)
        if inserted is not None:
            return inserted
        existing = self.get_generated_content(
            project_id=draft.project_id,
            recommendation_action_id=draft.recommendation_action_id,
            variant_key=draft.variant_key,
        )
        if existing is None:
            raise RuntimeError("generated_contents insert conflict could not be resolved")
        return existing

    def mark_action_content_generated(self, *, recommendation_action_id: int) -> None:
        self.connection.execute(
            """
            UPDATE recommendation_actions
            SET
                status = %(status)s,
                metadata = COALESCE(metadata, '{}'::jsonb) || CAST(%(metadata_patch)s AS jsonb),
                updated_at = now()
            WHERE id = %(recommendation_action_id)s
            """,
            {
                "recommendation_action_id": recommendation_action_id,
                "status": ACTION_STATUS_CONTENT_GENERATED,
                "metadata_patch": _json(
                    {
                        "error_type": None,
                        "error_message": None,
                    }
                ),
            },
        )

    def mark_action_failed(
        self,
        *,
        recommendation_action_id: int,
        error_type: str,
        error_message: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE recommendation_actions
            SET
                status = %(status)s,
                metadata = COALESCE(metadata, '{}'::jsonb) || CAST(%(metadata_patch)s AS jsonb),
                updated_at = now()
            WHERE id = %(recommendation_action_id)s
            """,
            {
                "recommendation_action_id": recommendation_action_id,
                "status": ACTION_STATUS_FAILED,
                "metadata_patch": _json(
                    {
                        "error_type": error_type,
                        "error_message": error_message,
                    }
                ),
            },
        )

    def _insert_generated_content_if_absent(
        self,
        draft: GeneratedContentDraft,
    ) -> GeneratedContentRecord | None:
        cursor = self.connection.execute(
            f"""
            INSERT INTO generated_contents ({_generated_content_columns()})
            VALUES ({_generated_content_value_placeholders()})
            ON CONFLICT (project_id, recommendation_action_id, variant_key)
                WHERE recommendation_action_id IS NOT NULL
                DO NOTHING
            RETURNING
                id,
                project_id,
                recommendation_action_id,
                segment_id,
                variant_key,
                generation_status,
                created_run_id,
                COALESCE(metadata, '{{}}'::jsonb) AS metadata
            """,
            _draft_params(draft),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def _force_upsert_generated_content(
        self,
        draft: GeneratedContentDraft,
    ) -> GeneratedContentRecord:
        cursor = self.connection.execute(
            f"""
            INSERT INTO generated_contents ({_generated_content_columns()})
            VALUES ({_generated_content_value_placeholders()})
            ON CONFLICT (project_id, recommendation_action_id, variant_key)
                WHERE recommendation_action_id IS NOT NULL
                DO UPDATE SET
                    content_type = EXCLUDED.content_type,
                    title = EXCLUDED.title,
                    body = EXCLUDED.body,
                    cta_label = EXCLUDED.cta_label,
                    landing_url = EXCLUDED.landing_url,
                    image_url = EXCLUDED.image_url,
                    media_s3_key = EXCLUDED.media_s3_key,
                    image_prompt = EXCLUDED.image_prompt,
                    generation_model = EXCLUDED.generation_model,
                    generation_status = EXCLUDED.generation_status,
                    created_run_id = EXCLUDED.created_run_id,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
            RETURNING
                id,
                project_id,
                recommendation_action_id,
                segment_id,
                variant_key,
                generation_status,
                created_run_id,
                COALESCE(metadata, '{{}}'::jsonb) AS metadata
            """,
            _draft_params(draft),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("generated_contents upsert returned no row")
        return self._record_from_row(row)

    def _target_from_row(self, row: Any) -> RecommendationActionTarget:
        return RecommendationActionTarget(
            id=int(_row_get(row, "recommendation_action_id")),
            project_id=_row_get(row, "project_id"),
            recommendation_result_id=int(_row_get(row, "recommendation_result_id")),
            action_key=str(_row_get(row, "action_key")),
            status=str(_row_get(row, "action_status")),
            segment=SegmentContext(
                id=int(_row_get(row, "segment_id")),
                segment_key=str(_row_get(row, "segment_key")),
                name=str(_row_get(row, "segment_name")),
                is_default=bool(_row_get(row, "segment_is_default")),
                description=_row_get(row, "segment_description"),
                attributes=_as_dict(_row_get(row, "segment_attributes")),
            ),
            analysis_date=str(_row_get(row, "analysis_date")),
            action_type=_row_get(row, "action_type"),
            action_title=_row_get(row, "action_title"),
            action_description=_row_get(row, "action_description"),
            content_type=_row_get(row, "content_type"),
            metrics=_as_dict(_row_get(row, "metrics")),
            root_cause=_as_dict(_row_get(row, "root_cause")),
            metadata=_as_dict(_row_get(row, "action_metadata")),
        )

    def _record_from_row(self, row: Any) -> GeneratedContentRecord:
        return GeneratedContentRecord(
            id=int(_row_get(row, "id")),
            project_id=_row_get(row, "project_id"),
            recommendation_action_id=_row_get(row, "recommendation_action_id"),
            segment_id=_row_get(row, "segment_id"),
            variant_key=str(_row_get(row, "variant_key")),
            generation_status=str(_row_get(row, "generation_status")),
            created_run_id=_row_get(row, "created_run_id"),
            metadata=_as_dict(_row_get(row, "metadata")),
        )


def advisory_lock_key(project_id: int | str, recommendation_action_id: int) -> int:
    raw = f"{project_id}:{recommendation_action_id}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _generated_content_columns() -> str:
    return (
        "project_id, segment_id, recommendation_action_id, variant_key, content_type, "
        "title, body, cta_label, landing_url, image_url, media_s3_key, image_prompt, "
        "generation_model, generation_status, metadata, created_run_id"
    )


def _generated_content_value_placeholders() -> str:
    return (
        "%(project_id)s, %(segment_id)s, %(recommendation_action_id)s, %(variant_key)s, "
        "%(content_type)s, %(title)s, %(body)s, %(cta_label)s, %(landing_url)s, "
        "%(image_url)s, %(media_s3_key)s, %(image_prompt)s, %(generation_model)s, "
        "%(generation_status)s, CAST(%(metadata)s AS jsonb), %(created_run_id)s"
    )


def _draft_params(draft: GeneratedContentDraft) -> dict[str, Any]:
    return {
        "project_id": draft.project_id,
        "segment_id": draft.segment_id,
        "recommendation_action_id": draft.recommendation_action_id,
        "variant_key": draft.variant_key,
        "content_type": draft.content_type,
        "title": draft.title,
        "body": draft.body,
        "cta_label": draft.cta_label,
        "landing_url": draft.landing_url,
        "image_url": draft.image_url,
        "media_s3_key": draft.media_s3_key,
        "image_prompt": draft.image_prompt,
        "generation_model": draft.generation_model,
        "generation_status": draft.generation_status,
        "metadata": _json(draft.metadata),
        "created_run_id": draft.created_run_id,
    }


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row[key]
    return getattr(row, key)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"expected dict-compatible JSON value, got {type(value).__name__}")


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
