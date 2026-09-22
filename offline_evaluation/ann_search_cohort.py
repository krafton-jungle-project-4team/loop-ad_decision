"""Prepare one deterministic ANN benchmark cohort in a disposable PostgreSQL."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from app.analysis.repositories import PsycopgPostgresExecutor
from offline_evaluation.ann_search_experiment import HNSW_INDEX_NAME


@dataclass(frozen=True, slots=True)
class CohortPreparation:
    project_id: str
    vector_version: str
    manifest_hash: str
    window_start: datetime
    window_end: datetime
    source_revision_cutoff: datetime
    cohort_size: int
    cohort_seed: str

    def __post_init__(self) -> None:
        if not self.project_id or not self.vector_version or not self.manifest_hash:
            raise ValueError("cohort identity fields are required")
        if self.cohort_size <= 0 or not self.cohort_seed:
            raise ValueError("cohort size and seed are required")
        if self.window_start >= self.window_end:
            raise ValueError("cohort vector window is invalid")
        for value in (
            self.window_start,
            self.window_end,
            self.source_revision_cutoff,
        ):
            if value.tzinfo is None:
                raise ValueError("cohort timestamps must be timezone-aware")

    @property
    def vector_generation_id(self) -> str:
        identity = "|".join(
            (
                self.project_id,
                self.vector_version,
                self.window_start.isoformat(),
                self.window_end.isoformat(),
                str(self.cohort_size),
                self.cohort_seed,
            )
        )
        return "ann-benchmark-" + hashlib.sha256(identity.encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class CohortPreparationResult:
    vector_generation_id: str
    cohort_size: int
    cohort_sha256: str
    index_build_seconds: float
    index_size_bytes: int
    postgres_row_count: int
    source_user_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnnBenchmarkCohortPreparer:
    """Load a stable subset from ClickHouse revisions into an empty local PG."""

    def __init__(self, *, postgres_connection: Any, clickhouse: Any) -> None:
        self._connection = postgres_connection
        self._postgres = PsycopgPostgresExecutor(postgres_connection)
        self._clickhouse = clickhouse

    def prepare(self, config: CohortPreparation) -> CohortPreparationResult:
        self._require_empty_search_database()
        source_user_count = self._source_user_count(config)
        if source_user_count < config.cohort_size:
            raise RuntimeError(
                f"source generation has {source_user_count} users, "
                f"cannot prepare {config.cohort_size}"
            )
        with self._connection.transaction():
            self._postgres.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX_NAME}")
            self._postgres.execute(
                """
                INSERT INTO projects (
                    project_id, project_name, domain, write_key, industry, status
                ) VALUES (%s, %s, %s, %s, 'hotel_booking', 'active')
                """,
                (
                    config.project_id,
                    "Expedia ANN benchmark",
                    "benchmark.local",
                    f"ann-benchmark-{config.cohort_size}-{config.cohort_seed}",
                ),
            )
            self._postgres.execute(
                """
                INSERT INTO user_behavior_vector_search_generations (
                    vector_generation_id, project_id, vector_version,
                    manifest_hash, window_start, window_end,
                    source_revision_cutoff, expected_user_count,
                    synced_user_count, invalid_user_count, status, is_active
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0,
                          'in_progress', false)
                """,
                (
                    config.vector_generation_id,
                    config.project_id,
                    config.vector_version,
                    config.manifest_hash,
                    config.window_start,
                    config.window_end,
                    config.source_revision_cutoff,
                    config.cohort_size,
                ),
            )

        cohort_hash = hashlib.sha256()
        inserted = 0
        stream = self._clickhouse.query_row_block_stream(
            _cohort_query(),
            parameters={
                "project_id": config.project_id,
                "vector_version": config.vector_version,
                "window_start": config.window_start,
                "window_end": config.window_end,
                "source_revision_cutoff": config.source_revision_cutoff,
                "cohort_seed": config.cohort_seed,
                "cohort_size": config.cohort_size,
            },
        )
        with stream as blocks:
            for rows in blocks:
                if not rows:
                    continue
                parsed = [_revision_row(row) for row in rows]
                for row in parsed:
                    cohort_hash.update(row[0].encode("utf-8"))
                    cohort_hash.update(b"\n")
                with self._connection.transaction():
                    self._insert_rows(config, parsed)
                inserted += len(parsed)
        if inserted != config.cohort_size:
            raise RuntimeError(
                f"cohort stream returned {inserted}, expected {config.cohort_size}"
            )

        started = time.perf_counter()
        with self._connection.transaction():
            self._postgres.execute(
                f"""
                CREATE INDEX {HNSW_INDEX_NAME}
                ON user_behavior_vector_search
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            self._postgres.execute("ANALYZE user_behavior_vector_search")
            self._postgres.execute(
                """
                UPDATE user_behavior_vector_search_generations
                SET synced_user_count = %s,
                    status = 'activated', is_active = true,
                    activated_at = now(), updated_at = now()
                WHERE vector_generation_id = %s
                """,
                (inserted, config.vector_generation_id),
            )
        index_build_seconds = time.perf_counter() - started
        with self._connection.transaction():
            row = self._postgres.fetchone(
                """
                SELECT count(*) AS row_count,
                       pg_relation_size(%s::regclass) AS index_size_bytes
                FROM user_behavior_vector_search
                WHERE vector_generation_id = %s
                """,
                (HNSW_INDEX_NAME, config.vector_generation_id),
            )
        if row is None or int(row["row_count"]) != config.cohort_size:
            raise RuntimeError("prepared PostgreSQL cohort row count mismatches")
        return CohortPreparationResult(
            vector_generation_id=config.vector_generation_id,
            cohort_size=config.cohort_size,
            cohort_sha256=cohort_hash.hexdigest(),
            index_build_seconds=index_build_seconds,
            index_size_bytes=int(row["index_size_bytes"]),
            postgres_row_count=int(row["row_count"]),
            source_user_count=source_user_count,
        )

    def _require_empty_search_database(self) -> None:
        with self._connection.transaction():
            row = self._postgres.fetchone(
                """
                SELECT
                    (SELECT count(*) FROM projects) AS project_count,
                    (SELECT count(*) FROM user_behavior_vector_search_generations)
                        AS generation_count,
                    (SELECT count(*) FROM user_behavior_vector_search)
                        AS search_row_count
                """
            )
        if row is None or any(int(value) != 0 for value in row.values()):
            raise RuntimeError(
                "prepare-cohort requires a fresh Data Contract PostgreSQL with "
                "no projects, generations, or search rows"
            )

    def _source_user_count(self, config: CohortPreparation) -> int:
        result = self._clickhouse.query(
            """
            SELECT uniqExact(user_id) AS user_count
            FROM user_behavior_vector_revisions
            WHERE project_id = {project_id:String}
              AND vector_version = {vector_version:String}
              AND window_start = {window_start:DateTime64(3, 'UTC')}
              AND window_end = {window_end:DateTime64(3, 'UTC')}
              AND ingested_at <= {source_revision_cutoff:DateTime64(6, 'UTC')}
            """,
            parameters={
                "project_id": config.project_id,
                "vector_version": config.vector_version,
                "window_start": config.window_start,
                "window_end": config.window_end,
                "source_revision_cutoff": config.source_revision_cutoff,
            },
        )
        rows = list(result.named_results())
        return int(rows[0]["user_count"]) if rows else 0

    def _insert_rows(
        self,
        config: CohortPreparation,
        rows: Sequence[tuple[str, str, str, datetime, datetime]],
    ) -> None:
        self._postgres.execute(
            """
            INSERT INTO user_behavior_vector_search (
                vector_generation_id, project_id, user_id, vector_version,
                vector_dim, embedding, window_start, window_end,
                source_vector_row_id, source_updated_at, source_ingested_at
            )
            SELECT %s, %s, source.user_id, %s, 64,
                   source.embedding::vector, %s, %s,
                   source.vector_row_id, source.updated_at, source.ingested_at
            FROM unnest(
                %s::text[], %s::text[], %s::text[],
                %s::timestamptz[], %s::timestamptz[]
            ) AS source(
                user_id, embedding, vector_row_id, updated_at, ingested_at
            )
            """,
            (
                config.vector_generation_id,
                config.project_id,
                config.vector_version,
                config.window_start,
                config.window_end,
                [row[0] for row in rows],
                [row[1] for row in rows],
                [row[2] for row in rows],
                [row[3] for row in rows],
                [row[4] for row in rows],
            ),
        )


def _cohort_query() -> str:
    return """
        SELECT user_id, vector_values, vector_row_id, updated_at, ingested_at
        FROM (
            SELECT
                user_id, vector_values, vector_row_id, updated_at, ingested_at,
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
        ORDER BY hex(SHA256(concat({cohort_seed:String}, '|', user_id))), user_id
        LIMIT {cohort_size:UInt32}
    """


def _revision_row(row: Sequence[Any]) -> tuple[str, str, str, datetime, datetime]:
    user_id, vector_values, vector_row_id, updated_at, ingested_at = row
    values = [float(value) for value in vector_values]
    if len(values) != 64 or any(not math_is_finite(value) for value in values):
        raise ValueError("source revision contains an invalid 64D vector")
    embedding = "[" + ",".join(format(value, ".17g") for value in values) + "]"
    return (
        str(user_id),
        embedding,
        str(vector_row_id),
        updated_at,
        ingested_at,
    )


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
