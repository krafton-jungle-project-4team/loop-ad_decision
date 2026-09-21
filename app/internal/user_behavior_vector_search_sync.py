from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence

from app.logging import duration_ms, log, log_context_scope, now_ms


VECTOR_DIM = 64
GENERATION_ACTIVATION_LOCK_NAMESPACE = (
    "user-behavior-vector-generation-activation-v1"
)


class ClickHouseClient(Protocol):
    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class PostgresExecutor(Protocol):
    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        ...

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class SearchVectorRevision:
    project_id: str
    user_id: str
    vector_dim: int
    vector_values: tuple[float, ...]
    vector_version: str
    source: str
    window_start: datetime
    window_end: datetime
    updated_at: datetime
    vector_row_id: str
    ingested_at: datetime


@dataclass(frozen=True, slots=True)
class VectorSyncCursor:
    user_id: str | None = None


@dataclass(frozen=True, slots=True)
class VectorSearchGeneration:
    vector_generation_id: str
    project_id: str
    vector_version: str
    manifest_hash: str
    window_start: datetime
    window_end: datetime
    source_revision_cutoff: datetime
    expected_user_count: int
    synced_user_count: int
    invalid_user_count: int
    cursor: VectorSyncCursor
    status: str


@dataclass(frozen=True, slots=True)
class UserBehaviorVectorSearchSyncResult:
    project_id: str
    vector_version: str
    vector_generation_id: str
    synced_user_count: int
    expected_user_count: int
    active_generation_id: str | None
    source_cutoff: datetime | None
    status: str

    @property
    def synced_vector_count(self) -> int:
        """Temporary compatibility alias for internal callers during rollout."""

        return self.synced_user_count


class UserBehaviorVectorSearchSyncRepository:
    def __init__(
        self,
        *,
        clickhouse: ClickHouseClient,
        postgres: PostgresExecutor,
    ) -> None:
        self._clickhouse = clickhouse
        self._postgres = postgres

    def register_generation(
        self,
        *,
        vector_generation_id: str,
        project_id: str,
        vector_version: str,
        manifest_hash: str,
        window_start: datetime,
        window_end: datetime,
        expected_user_count: int,
        source_revision_cutoff: datetime,
    ) -> None:
        self._postgres.execute(
            """
            INSERT INTO user_behavior_vector_search_generations (
                vector_generation_id, project_id, vector_version, manifest_hash,
                window_start, window_end, source_revision_cutoff,
                expected_user_count, synced_user_count, invalid_user_count,
                status, is_active, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0,
                    'in_progress', false, now(), now())
            ON CONFLICT (vector_generation_id) DO NOTHING
            """,
            (
                vector_generation_id,
                project_id,
                vector_version,
                manifest_hash,
                window_start,
                window_end,
                source_revision_cutoff,
                expected_user_count,
            ),
        )
        registered = self.get_generation(vector_generation_id=vector_generation_id)
        expected_identity = (
            project_id,
            vector_version,
            manifest_hash,
            window_start,
            window_end,
            source_revision_cutoff,
            expected_user_count,
        )
        actual_identity = (
            registered.project_id,
            registered.vector_version,
            registered.manifest_hash,
            registered.window_start,
            registered.window_end,
            registered.source_revision_cutoff,
            registered.expected_user_count,
        )
        if actual_identity != expected_identity:
            raise RuntimeError(
                "vector generation id conflicts with different build inputs"
            )

    def get_generation(self, *, vector_generation_id: str) -> VectorSearchGeneration:
        row = self._postgres.fetchone(
            """
            SELECT
                vector_generation_id, project_id, vector_version, manifest_hash,
                window_start, window_end, source_revision_cutoff,
                expected_user_count, synced_user_count, invalid_user_count,
                last_user_id, status
            FROM user_behavior_vector_search_generations
            WHERE vector_generation_id = %s
            """,
            (vector_generation_id,),
        )
        if row is None:
            raise RuntimeError("user behavior vector search generation does not exist")
        return VectorSearchGeneration(
            vector_generation_id=str(row["vector_generation_id"]),
            project_id=str(row["project_id"]),
            vector_version=str(row["vector_version"]),
            manifest_hash=str(row["manifest_hash"]),
            window_start=row["window_start"],
            window_end=row["window_end"],
            source_revision_cutoff=row["source_revision_cutoff"],
            expected_user_count=int(row["expected_user_count"]),
            synced_user_count=int(row["synced_user_count"]),
            invalid_user_count=int(row["invalid_user_count"]),
            cursor=VectorSyncCursor(
                str(row["last_user_id"])
                if row.get("last_user_id") is not None
                else None
            ),
            status=str(row["status"]),
        )

    def list_revisions(
        self,
        *,
        generation: VectorSearchGeneration,
        limit: int,
    ) -> list[SearchVectorRevision]:
        result = self._clickhouse.query(
            """
            SELECT
                project_id, user_id, vector_dim, vector_values, vector_version,
                source, window_start, window_end, updated_at, vector_row_id,
                ingested_at
            FROM (
                SELECT
                    project_id, user_id, vector_dim, vector_values,
                    vector_version, toString(source) AS source, window_start,
                    window_end, updated_at, vector_row_id, ingested_at,
                    row_number() OVER (
                        PARTITION BY user_id
                        ORDER BY ingested_at DESC, vector_row_id DESC
                    ) AS revision_rank
                FROM user_behavior_vector_revisions
                WHERE project_id = {project_id:String}
                  AND vector_version = {vector_version:String}
                  AND window_start = {window_start:DateTime64(3, 'UTC')}
                  AND window_end = {window_end:DateTime64(3, 'UTC')}
                  AND ingested_at <= {source_revision_cutoff:DateTime64(6, 'UTC')}
            )
            WHERE revision_rank = 1
              AND ({last_user_id:String} = '' OR user_id > {last_user_id:String})
            ORDER BY user_id ASC
            LIMIT {limit:UInt32}
            """,
            parameters={
                "project_id": generation.project_id,
                "vector_version": generation.vector_version,
                "window_start": generation.window_start,
                "window_end": generation.window_end,
                "source_revision_cutoff": generation.source_revision_cutoff,
                "last_user_id": generation.cursor.user_id or "",
                "limit": limit,
            },
        )
        rows = (
            list(result.named_results())
            if hasattr(result, "named_results")
            else list(result.result_rows)
        )
        return [_revision_from_row(row) for row in rows]

    def count_source_users(self, *, generation: VectorSearchGeneration) -> int:
        result = self._clickhouse.query(
            """
            SELECT uniqExact(user_id) AS source_user_count
            FROM user_behavior_vector_revisions
            WHERE project_id = {project_id:String}
              AND vector_version = {vector_version:String}
              AND window_start = {window_start:DateTime64(3, 'UTC')}
              AND window_end = {window_end:DateTime64(3, 'UTC')}
              AND ingested_at <= {source_revision_cutoff:DateTime64(6, 'UTC')}
            """,
            parameters={
                "project_id": generation.project_id,
                "vector_version": generation.vector_version,
                "window_start": generation.window_start,
                "window_end": generation.window_end,
                "source_revision_cutoff": generation.source_revision_cutoff,
            },
        )
        rows = (
            list(result.named_results())
            if hasattr(result, "named_results")
            else list(result.result_rows)
        )
        if not rows:
            return 0
        row = rows[0]
        return int(row["source_user_count"] if isinstance(row, Mapping) else row[0])

    def bulk_upsert_revisions(
        self,
        *,
        generation: VectorSearchGeneration,
        revisions: Sequence[SearchVectorRevision],
    ) -> None:
        if not revisions:
            return
        for revision in revisions:
            _validate_revision(revision, generation)
        self._postgres.execute(
            """
            INSERT INTO user_behavior_vector_search (
                vector_generation_id, project_id, user_id, vector_version,
                vector_dim, embedding, window_start, window_end,
                source_vector_row_id, source_updated_at, source_ingested_at,
                synced_at
            )
            SELECT
                %s, %s, rows.user_id, %s, 64,
                rows.embedding_text::vector, %s, %s,
                rows.source_vector_row_id, rows.source_updated_at,
                rows.source_ingested_at, now()
            FROM unnest(
                %s::text[], %s::text[], %s::text[],
                %s::timestamptz[], %s::timestamptz[]
            ) AS rows(
                user_id, embedding_text, source_vector_row_id,
                source_updated_at, source_ingested_at
            )
            ON CONFLICT (vector_generation_id, user_id)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                source_vector_row_id = EXCLUDED.source_vector_row_id,
                source_updated_at = EXCLUDED.source_updated_at,
                source_ingested_at = EXCLUDED.source_ingested_at,
                synced_at = now()
            WHERE (
                user_behavior_vector_search.source_ingested_at,
                user_behavior_vector_search.source_vector_row_id
            ) <= (
                EXCLUDED.source_ingested_at,
                EXCLUDED.source_vector_row_id
            )
            """,
            (
                generation.vector_generation_id,
                generation.project_id,
                generation.vector_version,
                generation.window_start,
                generation.window_end,
                [revision.user_id for revision in revisions],
                [_vector_literal(revision.vector_values) for revision in revisions],
                [revision.vector_row_id for revision in revisions],
                [revision.updated_at for revision in revisions],
                [revision.ingested_at for revision in revisions],
            ),
        )

    def save_progress(
        self,
        *,
        generation: VectorSearchGeneration,
        last_user_id: str,
    ) -> None:
        self._postgres.execute(
            """
            UPDATE user_behavior_vector_search_generations
            SET last_user_id = %s,
                synced_user_count = (
                    SELECT count(*)
                    FROM user_behavior_vector_search
                    WHERE vector_generation_id = %s
                ),
                updated_at = now()
            WHERE vector_generation_id = %s AND status = 'in_progress'
            """,
            (
                last_user_id,
                generation.vector_generation_id,
                generation.vector_generation_id,
            ),
        )

    def count_synced_users(self, *, vector_generation_id: str) -> int:
        row = self._postgres.fetchone(
            """
            SELECT count(*) AS synced_user_count
            FROM user_behavior_vector_search
            WHERE vector_generation_id = %s
            """,
            (vector_generation_id,),
        )
        return int(row["synced_user_count"]) if row is not None else 0

    def mark_failed(
        self,
        *,
        vector_generation_id: str,
        invalid_user_count: int,
        reason: str,
    ) -> None:
        self._postgres.execute(
            """
            UPDATE user_behavior_vector_search_generations
            SET status = 'failed', is_active = false,
                invalid_user_count = %s,
                failure_reason = %s,
                updated_at = now()
            WHERE vector_generation_id = %s AND status <> 'activated'
            """,
            (invalid_user_count, reason[:1000], vector_generation_id),
        )

    def activate_generation(self, *, generation: VectorSearchGeneration) -> None:
        activation_key = f"{generation.project_id}:{generation.vector_version}"
        self._postgres.execute(
            """
            SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))
            """,
            (GENERATION_ACTIVATION_LOCK_NAMESPACE, activation_key),
        )
        locked_row = self._postgres.fetchone(
            """
            SELECT
                vector_generation_id, project_id, vector_version, manifest_hash,
                window_start, window_end, source_revision_cutoff,
                expected_user_count, synced_user_count, invalid_user_count,
                last_user_id, status
            FROM user_behavior_vector_search_generations
            WHERE vector_generation_id = %s
            FOR UPDATE
            """,
            (generation.vector_generation_id,),
        )
        if locked_row is None:
            raise RuntimeError("vector generation disappeared during activation")
        locked = _generation_from_row(locked_row)
        if (
            locked.project_id != generation.project_id
            or locked.vector_version != generation.vector_version
            or locked.manifest_hash != generation.manifest_hash
            or locked.window_start != generation.window_start
            or locked.window_end != generation.window_end
            or locked.source_revision_cutoff != generation.source_revision_cutoff
            or locked.expected_user_count != generation.expected_user_count
        ):
            raise RuntimeError("vector generation changed before activation")
        synced_user_count = self.count_synced_users(
            vector_generation_id=locked.vector_generation_id
        )
        if (
            locked.status not in {"in_progress", "activated"}
            or locked.invalid_user_count != 0
            or synced_user_count != locked.expected_user_count
        ):
            raise RuntimeError("vector generation is incomplete and cannot be activated")
        self._postgres.execute(
            """
            UPDATE user_behavior_vector_search_generations
            SET status = 'superseded', is_active = false, updated_at = now()
            WHERE project_id = %s
              AND vector_version = %s
              AND is_active = true
              AND vector_generation_id <> %s
            """,
            (
                locked.project_id,
                locked.vector_version,
                locked.vector_generation_id,
            ),
        )
        self._postgres.execute(
            """
            UPDATE user_behavior_vector_search_generations
            SET status = 'activated', is_active = true,
                activated_at = COALESCE(activated_at, now()), updated_at = now()
            WHERE vector_generation_id = %s
              AND status IN ('in_progress', 'activated')
              AND invalid_user_count = 0
              AND expected_user_count = (
                  SELECT count(*)
                  FROM user_behavior_vector_search
                  WHERE vector_generation_id = %s
              )
            """,
            (locked.vector_generation_id, locked.vector_generation_id),
        )
        active = self._postgres.fetchone(
            """
            SELECT
                count(*) AS active_count,
                min(vector_generation_id) AS active_generation_id
            FROM user_behavior_vector_search_generations
            WHERE project_id = %s
              AND vector_version = %s
              AND status = 'activated'
              AND is_active = true
            """,
            (locked.project_id, locked.vector_version),
        )
        if (
            active is None
            or int(active["active_count"]) != 1
            or str(active["active_generation_id"]) != locked.vector_generation_id
        ):
            raise RuntimeError(
                "vector generation activation did not produce one active generation"
            )

    def get_active_generation_id(
        self,
        *,
        project_id: str,
        vector_version: str,
    ) -> str | None:
        row = self._postgres.fetchone(
            """
            SELECT vector_generation_id
            FROM user_behavior_vector_search_generations
            WHERE project_id = %s AND vector_version = %s
              AND status = 'activated' AND is_active = true
            ORDER BY activated_at DESC, vector_generation_id DESC
            LIMIT 1
            """,
            (project_id, vector_version),
        )
        return str(row["vector_generation_id"]) if row is not None else None


class UserBehaviorVectorSearchSyncService:
    def __init__(self, repository: UserBehaviorVectorSearchSyncRepository) -> None:
        self._repository = repository

    @log_context_scope
    def sync(
        self,
        *,
        project_id: str,
        vector_version: str,
        vector_generation_id: str,
        batch_size: int,
        max_batches: int,
    ) -> UserBehaviorVectorSearchSyncResult:
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": project_id,
                "vectorGenerationId": vector_generation_id,
                "vectorVersion": vector_version,
            }
        )
        log.info(
            "started",
            {"batchSize": batch_size, "maxBatches": max_batches},
        )
        generation = self._repository.get_generation(
            vector_generation_id=vector_generation_id
        )
        if (
            generation.project_id != project_id
            or generation.vector_version != vector_version
        ):
            log.warn(
                "vector_search_generation_mismatch",
                {
                    "actualProjectId": generation.project_id,
                    "actualVectorVersion": generation.vector_version,
                },
            )
            raise RuntimeError("vector generation does not belong to this project/version")
        if generation.status == "failed":
            log.info(
                "vector_search_sync_skipped",
                {"reason": "generation_already_failed"},
            )
            return self._completed_result(
                generation,
                status="failed",
                started_at=started_at,
                processed_batch_count=0,
            )
        if generation.status == "activated":
            log.info(
                "vector_search_sync_skipped",
                {"reason": "generation_already_activated"},
            )
            return self._completed_result(
                generation,
                status="activated",
                started_at=started_at,
                processed_batch_count=0,
            )
        if generation.status != "in_progress":
            log.warn(
                "vector_search_generation_not_syncable",
                {"status": generation.status},
            )
            raise RuntimeError("vector generation is not syncable")

        exhausted = False
        processed_batch_count = 0
        for batch_index in range(max_batches):
            generation = self._repository.get_generation(
                vector_generation_id=vector_generation_id
            )
            revisions = self._repository.list_revisions(
                generation=generation,
                limit=batch_size,
            )
            if not revisions:
                exhausted = True
                break
            try:
                self._repository.bulk_upsert_revisions(
                    generation=generation,
                    revisions=revisions,
                )
            except ValueError as exc:
                log.warn(
                    "vector_search_revision_invalid",
                    {
                        "batchIndex": batch_index,
                        "err": exc,
                        "revisionCount": len(revisions),
                    },
                )
                self._repository.mark_failed(
                    vector_generation_id=vector_generation_id,
                    invalid_user_count=1,
                    reason=str(exc),
                )
                failed = self._repository.get_generation(
                    vector_generation_id=vector_generation_id
                )
                log.info(
                    "vector_search_generation_failed",
                    {"reason": "revision_invalid"},
                )
                return self._completed_result(
                    failed,
                    status="failed",
                    started_at=started_at,
                    processed_batch_count=processed_batch_count,
                )
            self._repository.save_progress(
                generation=generation,
                last_user_id=revisions[-1].user_id,
            )
            processed_batch_count += 1
            log.info(
                "vector_search_sync_batch_completed",
                {
                    "batchIndex": batch_index,
                    "revisionCount": len(revisions),
                },
            )
            if len(revisions) < batch_size:
                exhausted = True
                break

        generation = self._repository.get_generation(
            vector_generation_id=vector_generation_id
        )
        if not exhausted:
            return self._completed_result(
                generation,
                status="in_progress",
                started_at=started_at,
                processed_batch_count=processed_batch_count,
            )

        source_user_count = self._repository.count_source_users(
            generation=generation
        )
        synced_user_count = self._repository.count_synced_users(
            vector_generation_id=vector_generation_id
        )
        if (
            source_user_count != generation.expected_user_count
            or synced_user_count != generation.expected_user_count
            or generation.invalid_user_count != 0
        ):
            log.warn(
                "vector_search_generation_completeness_mismatch",
                {
                    "expectedUserCount": generation.expected_user_count,
                    "invalidUserCount": generation.invalid_user_count,
                    "sourceUserCount": source_user_count,
                    "syncedUserCount": synced_user_count,
                },
            )
            self._repository.mark_failed(
                vector_generation_id=vector_generation_id,
                invalid_user_count=generation.invalid_user_count,
                reason=(
                    "generation completeness mismatch: "
                    f"expected={generation.expected_user_count}, "
                    f"source={source_user_count}, synced={synced_user_count}"
                ),
            )
            failed = self._repository.get_generation(
                vector_generation_id=vector_generation_id
            )
            log.info(
                "vector_search_generation_failed",
                {"reason": "completeness_mismatch"},
            )
            return self._completed_result(
                failed,
                status="failed",
                started_at=started_at,
                processed_batch_count=processed_batch_count,
            )

        self._repository.activate_generation(generation=generation)
        activated = self._repository.get_generation(
            vector_generation_id=vector_generation_id
        )
        if activated.status != "activated":
            log.warn(
                "vector_search_generation_activation_mismatch",
                {"status": activated.status},
            )
            raise RuntimeError("complete vector generation could not be activated")
        log.info(
            "vector_search_generation_activated",
            {"expectedUserCount": activated.expected_user_count},
        )
        return self._completed_result(
            activated,
            status="activated",
            started_at=started_at,
            processed_batch_count=processed_batch_count,
        )

    def _completed_result(
        self,
        generation: VectorSearchGeneration,
        *,
        status: str,
        started_at: float,
        processed_batch_count: int,
    ) -> UserBehaviorVectorSearchSyncResult:
        result = self._result(generation, status=status)
        log.info(
            "completed",
            {
                "durationMs": duration_ms(started_at),
                "expectedUserCount": result.expected_user_count,
                "processedBatchCount": processed_batch_count,
                "status": result.status,
                "syncedUserCount": result.synced_user_count,
            },
        )
        return result

    def _result(
        self,
        generation: VectorSearchGeneration,
        *,
        status: str,
    ) -> UserBehaviorVectorSearchSyncResult:
        return UserBehaviorVectorSearchSyncResult(
            project_id=generation.project_id,
            vector_version=generation.vector_version,
            vector_generation_id=generation.vector_generation_id,
            synced_user_count=self._repository.count_synced_users(
                vector_generation_id=generation.vector_generation_id
            ),
            expected_user_count=generation.expected_user_count,
            active_generation_id=self._repository.get_active_generation_id(
                project_id=generation.project_id,
                vector_version=generation.vector_version,
            ),
            source_cutoff=generation.window_end,
            status=status,
        )


def _revision_from_row(row: Any) -> SearchVectorRevision:
    if not isinstance(row, Mapping):
        raise ValueError("ClickHouse revision rows must be named")
    return SearchVectorRevision(
        project_id=str(row["project_id"]),
        user_id=str(row["user_id"]),
        vector_dim=int(row["vector_dim"]),
        vector_values=tuple(float(value) for value in row["vector_values"]),
        vector_version=str(row["vector_version"]),
        source=str(row["source"]),
        window_start=row["window_start"],
        window_end=row["window_end"],
        updated_at=row["updated_at"],
        vector_row_id=str(row["vector_row_id"]),
        ingested_at=row["ingested_at"],
    )


def _generation_from_row(row: Mapping[str, Any]) -> VectorSearchGeneration:
    return VectorSearchGeneration(
        vector_generation_id=str(row["vector_generation_id"]),
        project_id=str(row["project_id"]),
        vector_version=str(row["vector_version"]),
        manifest_hash=str(row["manifest_hash"]),
        window_start=row["window_start"],
        window_end=row["window_end"],
        source_revision_cutoff=row["source_revision_cutoff"],
        expected_user_count=int(row["expected_user_count"]),
        synced_user_count=int(row["synced_user_count"]),
        invalid_user_count=int(row["invalid_user_count"]),
        cursor=VectorSyncCursor(
            str(row["last_user_id"])
            if row.get("last_user_id") is not None
            else None
        ),
        status=str(row["status"]),
    )


def _validate_revision(
    revision: SearchVectorRevision,
    generation: VectorSearchGeneration,
) -> None:
    revision_window = (
        _normalize_vector_window(revision.window_start),
        _normalize_vector_window(revision.window_end),
    )
    generation_window = (
        _normalize_vector_window(generation.window_start),
        _normalize_vector_window(generation.window_end),
    )
    if (
        revision.project_id != generation.project_id
        or revision.vector_version != generation.vector_version
        or revision_window != generation_window
    ):
        raise ValueError("search vector revision does not belong to generation")
    if revision.vector_dim != VECTOR_DIM or len(revision.vector_values) != VECTOR_DIM:
        raise ValueError("search vector revision must contain 64 values")
    if not all(math.isfinite(value) for value in revision.vector_values):
        raise ValueError("search vector revision must contain finite values")
    if math.sqrt(sum(value * value for value in revision.vector_values)) == 0:
        raise ValueError("search vector revision must not be zero")


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".17g") for value in values) + "]"


def _normalize_vector_window(value: datetime) -> datetime:
    if value.tzinfo is None:
        utc_value = value.replace(tzinfo=timezone.utc)
    else:
        utc_value = value.astimezone(timezone.utc)
    millisecond = (utc_value.microsecond // 1000) * 1000
    return utc_value.replace(microsecond=millisecond)
