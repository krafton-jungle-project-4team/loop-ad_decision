from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence

from psycopg.types.json import Jsonb

from app.decision.repositories import (
    PsycopgPostgresExecutor,
    SegmentAssignmentExecutionRepository,
    UserSegmentAssignmentRepository,
    UserSegmentAssignmentWrite,
)


RUN_ID = "prun_banner_001_loop_1"
EXECUTION_ID = "segment_assignment_execution_0123456789abcdef"
REQUEST_FINGERPRINT = "a" * 64
INPUT_FINGERPRINT = "b" * 64
SOURCE_CUTOFF_AT = datetime(2026, 7, 14, 1, 2, 3, 456789, tzinfo=UTC)
CREATED_AT = datetime(2026, 7, 14, 1, 2, 4, tzinfo=UTC)


def manifest(*, finalized: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": "segment_assignment_input_manifest.v1",
        "canonical_input": {
            "version": "segment_assignment_canonical_input.v1",
            "selection_version": "user_behavior_vector_revisions_argmax_v1",
            "source_cutoff_at": "2026-07-14T01:02:03.456789Z",
            "vector_row_id_stream_digest": "c" * 64,
        },
        "matcher": {
            "version": "segment_assignment_matcher_manifest.v1",
            "strategy": "exact_cosine",
        },
    }
    if finalized:
        value["result_summary"] = {
            "version": "segment_assignment_result_summary.v1",
            "assignment_count": 2,
            "newly_linked_count": 1,
            "reused_existing_count": 1,
        }
    return value


def execution_row(
    *,
    input_manifest_json: Mapping[str, Any] | str | None = None,
    matcher_strategy: str = "exact_cosine",
) -> dict[str, Any]:
    return {
        "segment_assignment_execution_id": EXECUTION_ID,
        "promotion_run_id": RUN_ID,
        "request_fingerprint": REQUEST_FINGERPRINT,
        "input_fingerprint": INPUT_FINGERPRINT,
        "matcher_strategy": matcher_strategy,
        "matcher_version": "exact_cosine_v1",
        "vector_version": "v1",
        "source_cutoff_at": SOURCE_CUTOFF_AT,
        "input_manifest_json": (
            input_manifest_json if input_manifest_json is not None else manifest()
        ),
        "created_at": CREATED_AT,
    }


@dataclass(frozen=True)
class DbCall:
    operation: str
    query: str
    params: Sequence[Any] | Mapping[str, Any]


class FakePostgresExecutor:
    def __init__(
        self,
        *,
        fetchone_result: Mapping[str, Any] | None = None,
        fetchall_result: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.calls: list[DbCall] = []

    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        self.calls.append(DbCall("fetchone", query, params))
        return self.fetchone_result

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        self.calls.append(DbCall("fetchall", query, params))
        return self.fetchall_result

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        self.calls.append(DbCall("execute", query, params))


class RecordingCursor:
    def __init__(self, row: Mapping[str, Any]) -> None:
        self.row = row
        self.query: str | None = None
        self.params: Sequence[Any] | Mapping[str, Any] | None = None

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any],
    ) -> None:
        self.query = query
        self.params = params

    def fetchone(self) -> Mapping[str, Any]:
        return self.row


class RecordingConnection:
    def __init__(self, row: Mapping[str, Any]) -> None:
        self.cursor_instance = RecordingCursor(row)

    def cursor(self, *, row_factory: Any) -> RecordingCursor:
        assert row_factory is not None
        return self.cursor_instance


def compact_sql(query: str) -> str:
    return " ".join(query.lower().split())


def assignment_write(
    *,
    run_id: str = RUN_ID,
    execution_id: str | None = EXECUTION_ID,
    user_id: str = "user_001",
) -> UserSegmentAssignmentWrite:
    return UserSegmentAssignmentWrite(
        project_id="hotel-client-a",
        promotion_run_id=run_id,
        user_id=user_id,
        segment_id="seg_family_trip",
        ad_experiment_id="adexp_family_trip_001",
        content_id="content_family_trip_001",
        content_option_id="option_a",
        similarity_score=Decimal("0.910000"),
        fallback=False,
        fallback_reason=None,
        assignment_source="decision_batch",
        assigned_at=datetime(2026, 7, 14, 1, 3, tzinfo=UTC),
        expires_at=datetime(2026, 7, 21, 1, 3, tzinfo=UTC),
        segment_assignment_execution_id=execution_id,
    )


def test_get_by_request_scopes_lookup_to_run_and_maps_json_readback() -> None:
    import json

    stored_manifest = manifest()
    db = FakePostgresExecutor(
        fetchone_result=execution_row(
            input_manifest_json=json.dumps(stored_manifest, separators=(",", ":"))
        )
    )

    execution = SegmentAssignmentExecutionRepository(db).get_by_request(
        promotion_run_id=RUN_ID,
        request_fingerprint=REQUEST_FINGERPRINT,
    )

    assert execution is not None
    assert execution["segment_assignment_execution_id"] == EXECUTION_ID
    assert execution["source_cutoff_at"] == SOURCE_CUTOFF_AT
    assert execution["input_manifest_json"] == stored_manifest
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from segment_assignment_executions" in sql
    assert "where promotion_run_id = %s and request_fingerprint = %s" in sql
    assert call.params == (RUN_ID, REQUEST_FINGERPRINT)


def test_insert_provisional_uses_jsonb_and_converges_on_run_request_conflict() -> None:
    provisional_manifest = manifest(finalized=False)
    connection = RecordingConnection(
        execution_row(input_manifest_json=provisional_manifest)
    )
    repository = SegmentAssignmentExecutionRepository(
        PsycopgPostgresExecutor(connection)
    )

    inserted = repository.insert_provisional(
        segment_assignment_execution_id=EXECUTION_ID,
        promotion_run_id=RUN_ID,
        request_fingerprint=REQUEST_FINGERPRINT,
        input_fingerprint=INPUT_FINGERPRINT,
        matcher_strategy="exact_cosine",
        matcher_version="exact_cosine_v1",
        vector_version="v1",
        source_cutoff_at=SOURCE_CUTOFF_AT,
        input_manifest_json=provisional_manifest,
    )

    assert inserted is not None
    cursor = connection.cursor_instance
    assert cursor.query is not None
    sql = compact_sql(cursor.query)
    assert "insert into segment_assignment_executions" in sql
    assert "on conflict (promotion_run_id, request_fingerprint) do nothing" in sql
    assert "do update" not in sql
    assert "returning segment_assignment_execution_id" in sql
    assert cursor.params is not None
    assert tuple(cursor.params[:8]) == (
        EXECUTION_ID,
        RUN_ID,
        REQUEST_FINGERPRINT,
        INPUT_FINGERPRINT,
        "exact_cosine",
        "exact_cosine_v1",
        "v1",
        SOURCE_CUTOFF_AT,
    )
    assert isinstance(cursor.params[8], Jsonb)
    assert cursor.params[8].obj == provisional_manifest


def test_insert_provisional_conflict_returns_none_without_mutating_winner() -> None:
    db = FakePostgresExecutor(fetchone_result=None)
    repository = SegmentAssignmentExecutionRepository(db)

    inserted = repository.insert_provisional(
        segment_assignment_execution_id=EXECUTION_ID,
        promotion_run_id=RUN_ID,
        request_fingerprint=REQUEST_FINGERPRINT,
        input_fingerprint=INPUT_FINGERPRINT,
        matcher_strategy="exact_cosine",
        matcher_version="exact_cosine_v1",
        vector_version="v1",
        source_cutoff_at=SOURCE_CUTOFF_AT,
        input_manifest_json=manifest(finalized=False),
    )

    assert inserted is None
    sql = compact_sql(db.calls[0].query)
    assert "on conflict (promotion_run_id, request_fingerprint) do nothing" in sql
    assert "do update" not in sql


def test_finalize_updates_bounded_result_fields_and_reads_back_final_row() -> None:
    final_manifest = manifest()
    db = FakePostgresExecutor(fetchone_result=execution_row())

    finalized = SegmentAssignmentExecutionRepository(db).finalize(
        segment_assignment_execution_id=EXECUTION_ID,
        promotion_run_id=RUN_ID,
        request_fingerprint=REQUEST_FINGERPRINT,
        input_fingerprint=INPUT_FINGERPRINT,
        matcher_strategy="exact_cosine",
        matcher_version="exact_cosine_v1",
        input_manifest_json=final_manifest,
    )

    assert finalized["input_manifest_json"] == final_manifest
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "update segment_assignment_executions" in sql
    assert "set input_fingerprint = %s" in sql
    assert "matcher_strategy = %s" in sql
    assert "matcher_version = %s" in sql
    assert "input_manifest_json = %s" in sql
    assert "source_cutoff_at =" not in sql
    assert "vector_version =" not in sql
    assert (
        "where segment_assignment_execution_id = %s "
        "and promotion_run_id = %s and request_fingerprint = %s"
    ) in sql
    assert call.params == (
        INPUT_FINGERPRINT,
        "exact_cosine",
        "exact_cosine_v1",
        final_manifest,
        EXECUTION_ID,
        RUN_ID,
        REQUEST_FINGERPRINT,
    )


def test_count_linked_assignments_uses_composite_run_execution_identity() -> None:
    db = FakePostgresExecutor(
        fetchone_result={"linked_assignment_count": 7}
    )

    count = SegmentAssignmentExecutionRepository(db).count_linked_assignments(
        promotion_run_id=RUN_ID,
        segment_assignment_execution_id=EXECUTION_ID,
    )

    assert count == 7
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from user_segment_assignments" in sql
    assert (
        "where promotion_run_id = %s and segment_assignment_execution_id = %s"
    ) in sql
    assert call.params == (RUN_ID, EXECUTION_ID)


def test_assignment_insert_persists_run_execution_pairs_and_never_updates_conflicts() -> None:
    other_run_id = "prun_banner_001_loop_2"
    other_execution_id = "segment_assignment_execution_fedcba9876543210"
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "user_id": "user_001",
                "segment_id": "seg_family_trip",
                "fallback": False,
                "fallback_reason": None,
                "similarity_score": Decimal("0.910000"),
                "segment_assignment_execution_id": EXECUTION_ID,
            }
        ]
    )

    inserted = UserSegmentAssignmentRepository(db).insert_many(
        [
            assignment_write(),
            assignment_write(
                run_id=other_run_id,
                execution_id=other_execution_id,
                user_id="user_002",
            ),
        ]
    )

    assert inserted[0].segment_assignment_execution_id == EXECUTION_ID
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "segment_assignment_execution_id" in sql
    assert (
        "promotion_run_id, user_id, segment_id, ad_experiment_id" in sql
    )
    assert "on conflict (promotion_run_id, user_id) do nothing" in sql
    assert "do update" not in sql
    assert "returning user_id, segment_id, fallback, fallback_reason, " in sql
    assert "similarity_score, segment_assignment_execution_id" in sql
    assert call.params[1] == [RUN_ID, other_run_id]
    assert call.params[13] == [EXECUTION_ID, other_execution_id]


def test_existing_legacy_null_execution_assignment_is_read_without_update() -> None:
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "user_id": "user_legacy",
                "segment_id": "seg_fallback",
                "fallback": True,
                "fallback_reason": "invalid_user_vector",
                "similarity_score": None,
                "segment_assignment_execution_id": None,
            }
        ]
    )

    records = UserSegmentAssignmentRepository(db).list_existing_assignments(
        promotion_run_id=RUN_ID,
        user_ids=["user_legacy"],
    )

    assert len(records) == 1
    assert records[0].segment_assignment_execution_id is None
    assert records[0].fallback is True
    call = db.calls[0]
    assert call.operation == "fetchall"
    sql = compact_sql(call.query)
    assert "select user_id, segment_id, fallback, fallback_reason" in sql
    assert "segment_assignment_execution_id" in sql
    assert "from user_segment_assignments" in sql
    assert "update" not in sql
    assert "insert" not in sql
    assert call.params == (RUN_ID, ["user_legacy"])
