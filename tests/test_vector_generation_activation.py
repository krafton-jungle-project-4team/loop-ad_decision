from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import threading
import uuid

import psycopg
import pytest
from psycopg import sql

from app.decision.repositories import PsycopgPostgresExecutor
from app.internal.user_behavior_vector_search_sync import (
    GENERATION_ACTIVATION_LOCK_NAMESPACE,
    UserBehaviorVectorSearchSyncRepository,
    VectorSearchGeneration,
    VectorSyncCursor,
)


class _UnusedClickHouse:
    def query(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("generation activation must not query ClickHouse")


class _ActivationPostgres:
    def __init__(self, generation: VectorSearchGeneration) -> None:
        self.generation = generation
        self.events: list[tuple[str, str, object]] = []

    def execute(self, query: str, params: object = ()) -> None:
        self.events.append(("execute", " ".join(query.split()), params))

    def fetchone(self, query: str, params: object = ()) -> dict[str, object]:
        compact = " ".join(query.split())
        self.events.append(("fetchone", compact, params))
        if "FOR UPDATE" in query:
            return _generation_row(self.generation)
        if "count(*) AS synced_user_count" in query:
            return {"synced_user_count": self.generation.expected_user_count}
        if "count(*) AS active_count" in query:
            return {
                "active_count": 1,
                "active_generation_id": self.generation.vector_generation_id,
            }
        raise AssertionError(f"unexpected activation query: {compact}")


def test_generation_activation_locks_rereads_and_verifies_in_order() -> None:
    generation = _generation("uvgen_a")
    postgres = _ActivationPostgres(generation)
    repository = UserBehaviorVectorSearchSyncRepository(
        clickhouse=_UnusedClickHouse(),
        postgres=postgres,
    )

    repository.activate_generation(generation=generation)

    operations = [event[0] for event in postgres.events]
    sql_values = [event[1] for event in postgres.events]
    assert operations == [
        "execute",
        "fetchone",
        "fetchone",
        "execute",
        "execute",
        "fetchone",
    ]
    assert "pg_advisory_xact_lock" in sql_values[0]
    assert postgres.events[0][2] == (
        GENERATION_ACTIVATION_LOCK_NAMESPACE,
        "project:hotel_behavior.v2",
    )
    assert "FOR UPDATE" in sql_values[1]
    assert "status = 'superseded'" in sql_values[3]
    assert "status = 'activated'" in sql_values[4]
    assert "count(*) AS active_count" in sql_values[5]


@pytest.fixture
def vector_generation_postgres_schema(
    loopad_test_postgres_dsn: str,
) -> tuple[str, str]:
    schema_name = f"test_vector_activation_{uuid.uuid4().hex}"
    admin = psycopg.connect(loopad_test_postgres_dsn, autocommit=True)
    try:
        admin.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
        )
        admin.execute(
            sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
        )
        admin.execute(
            """
            CREATE TABLE user_behavior_vector_search_generations (
                vector_generation_id text PRIMARY KEY,
                project_id text NOT NULL,
                vector_version text NOT NULL,
                manifest_hash text NOT NULL,
                window_start timestamptz NOT NULL,
                window_end timestamptz NOT NULL,
                source_revision_cutoff timestamptz NOT NULL,
                expected_user_count integer NOT NULL,
                synced_user_count integer NOT NULL,
                invalid_user_count integer NOT NULL,
                last_user_id text,
                status text NOT NULL,
                is_active boolean NOT NULL,
                activated_at timestamptz,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        admin.execute(
            """
            CREATE TABLE user_behavior_vector_search (
                vector_generation_id text NOT NULL,
                user_id text NOT NULL,
                PRIMARY KEY (vector_generation_id, user_id)
            )
            """
        )
        yield loopad_test_postgres_dsn, schema_name
    finally:
        admin.execute(
            sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
        )
        admin.close()


def test_two_connections_activate_exactly_one_generation(
    vector_generation_postgres_schema: tuple[str, str],
) -> None:
    dsn, schema_name = vector_generation_postgres_schema
    generations = (_generation("uvgen_a"), _generation("uvgen_b"))
    admin = psycopg.connect(dsn, autocommit=True)
    try:
        admin.execute(
            sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
        )
        for generation in generations:
            admin.execute(
                """
                INSERT INTO user_behavior_vector_search_generations (
                    vector_generation_id, project_id, vector_version,
                    manifest_hash, window_start, window_end,
                    source_revision_cutoff, expected_user_count,
                    synced_user_count, invalid_user_count, status, is_active
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 1, 1, 0,
                          'in_progress', false)
                """,
                (
                    generation.vector_generation_id,
                    generation.project_id,
                    generation.vector_version,
                    generation.manifest_hash,
                    generation.window_start,
                    generation.window_end,
                    generation.source_revision_cutoff,
                ),
            )
            admin.execute(
                """
                INSERT INTO user_behavior_vector_search (
                    vector_generation_id, user_id
                ) VALUES (%s, %s)
                """,
                (generation.vector_generation_id, "user_1"),
            )

        barrier = threading.Barrier(2)

        def activate(generation: VectorSearchGeneration) -> None:
            connection = psycopg.connect(dsn, autocommit=False)
            try:
                connection.execute(
                    sql.SQL("SET search_path TO {}").format(
                        sql.Identifier(schema_name)
                    )
                )
                repository = UserBehaviorVectorSearchSyncRepository(
                    clickhouse=_UnusedClickHouse(),
                    postgres=PsycopgPostgresExecutor(connection),
                )
                barrier.wait(timeout=5)
                repository.activate_generation(generation=generation)
                connection.commit()
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(activate, item) for item in generations]
            for future in futures:
                future.result(timeout=10)

        rows = admin.execute(
            """
            SELECT vector_generation_id, status, is_active
            FROM user_behavior_vector_search_generations
            ORDER BY vector_generation_id
            """
        ).fetchall()
        assert sum(1 for row in rows if row[2]) == 1
        assert {row[1] for row in rows} == {"activated", "superseded"}
    finally:
        admin.close()


def _generation(vector_generation_id: str) -> VectorSearchGeneration:
    now = datetime(2026, 7, 17, 0, 0, tzinfo=UTC)
    return VectorSearchGeneration(
        vector_generation_id=vector_generation_id,
        project_id="project",
        vector_version="hotel_behavior.v2",
        manifest_hash="a" * 64,
        window_start=now - timedelta(days=30),
        window_end=now,
        source_revision_cutoff=now + timedelta(seconds=1),
        expected_user_count=1,
        synced_user_count=1,
        invalid_user_count=0,
        cursor=VectorSyncCursor("user_1"),
        status="in_progress",
    )


def _generation_row(generation: VectorSearchGeneration) -> dict[str, object]:
    return {
        "vector_generation_id": generation.vector_generation_id,
        "project_id": generation.project_id,
        "vector_version": generation.vector_version,
        "manifest_hash": generation.manifest_hash,
        "window_start": generation.window_start,
        "window_end": generation.window_end,
        "source_revision_cutoff": generation.source_revision_cutoff,
        "expected_user_count": generation.expected_user_count,
        "synced_user_count": generation.synced_user_count,
        "invalid_user_count": generation.invalid_user_count,
        "last_user_id": generation.cursor.user_id,
        "status": generation.status,
    }
