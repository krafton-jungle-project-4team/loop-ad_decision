from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.assignment_service import (
    SegmentAssignmentRunNotFoundError,
    SegmentAssignmentValidationError,
)
from app.decision.router import get_segment_assignment_service
from app.decision.schemas import (
    SegmentAssignmentBuildRequest,
    SegmentAssignmentBuildResponse,
)
from app.main import create_app


DEFAULT_RUN_ROW = object()


def valid_env() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "15432",
        }
    )
    return values


def make_client(service: object | None = None) -> TestClient:
    app = create_app()
    if service is not None:
        app.dependency_overrides[get_segment_assignment_service] = lambda: service
    return TestClient(app)


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def test_segment_assignment_api_returns_conservative_response_shape() -> None:
    service = FakeAssignmentService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/segment-assignments/build",
        json={"user_ids": ["user_001"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "promotion_run_id": "prun_banner_001_loop_1",
        "matching_mode": "pgvector_hnsw_rerank",
        "vector_version": "v1",
        "ann_candidate_limit": 50,
        "ann_candidate_count": 1,
        "exact_reranked_pair_count": 1,
        "page_count": 1,
        "processed_user_count": 1,
        "assignment_count": 1,
        "run_assignment_count": 1,
        "run_has_fallback": False,
        "run_fallback_count": 0,
        "insert_conflict_count": 0,
        "segment_assignment_counts": {"seg_family_trip": 1},
        "batch_has_fallback": False,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "fallback_reason_counts": {
            "below_threshold": 0,
            "no_candidate": 0,
            "invalid_user_vector": 0,
        },
        "below_threshold_fallback_count": 0,
        "no_candidate_fallback_count": 0,
        "invalid_user_vector_fallback_count": 0,
        "similarity_score_buckets": {
            "not_available": 0,
            "0_00_to_0_50": 0,
            "0_50_to_0_65": 0,
            "0_65_to_0_80": 0,
            "0_80_to_0_90": 0,
            "gte_0_90": 1,
        },
        "ann_underfilled_user_count": 0,
        "exact_rescue_user_count": 0,
        "ann_applied": True,
        "ann_not_applied_reason": None,
        "skipped_existing_count": 0,
        "insufficient_segment_count": 0,
        "completion_scope": "current_request",
        "assignment_mode": "explicit_user_ids",
        "input_stability": "not_snapshotted",
        "status": "completed",
    }
    assert service.calls[0][0] == "prun_banner_001_loop_1"
    assert service.calls[0][1].user_ids == ["user_001"]


def test_segment_assignment_response_marks_insufficient_count_deprecated() -> None:
    property_schema = SegmentAssignmentBuildResponse.model_json_schema()[
        "properties"
    ]["insufficient_segment_count"]

    assert property_schema["deprecated"] is True
    assert property_schema["const"] == 0


def test_segment_assignment_response_documents_persisted_similarity_score_range() -> None:
    property_schema = SegmentAssignmentBuildResponse.model_json_schema()[
        "properties"
    ]["similarity_score_buckets"]

    assert "persisted similarity scores" in property_schema["description"]
    assert "[0, 1]" in property_schema["description"]


def test_segment_assignment_api_allows_empty_json_body_to_reach_service() -> None:
    service = FakeAssignmentService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/segment-assignments/build",
        json={},
    )

    assert response.status_code == 200
    assert service.calls[0][0] == "prun_banner_001_loop_1"
    assert service.calls[0][1].user_ids is None
    assert service.calls[0][1].eligible_user_limit is None


def test_segment_assignment_api_maps_service_errors() -> None:
    cases = [
        (SegmentAssignmentRunNotFoundError("missing run"), 404),
        (SegmentAssignmentValidationError("invalid assignment input"), 422),
    ]

    for exc, expected_status in cases:
        client = make_client(FakeAssignmentService(exc=exc))

        response = client.post(
            "/decision/v1/promotion-runs/prun_banner_001_loop_1/segment-assignments/build",
            json={"user_ids": ["user_001"]},
        )

        assert response.status_code == expected_status


def test_segment_assignment_api_wires_repositories_and_commits(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/segment-assignments/build",
        json={"user_ids": ["user_001"]},
    )

    assert response.status_code == 200
    assert response.json()["assignment_count"] == 1
    assert len(connections) == 1
    assert len(clickhouse_clients) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("from promotion_runs" in query for query in executed_sql)
    assert any("set_config('hnsw.ef_search'" in query for query in executed_sql)
    assert any("set_config('hnsw.iterative_scan'" in query for query in executed_sql)
    assert any("set_config('hnsw.max_scan_tuples'" in query for query in executed_sql)
    assert any("order by embedding <=>" in query for query in executed_sql)
    assert any("insert into user_segment_assignments" in query for query in executed_sql)


def test_segment_assignment_api_does_not_return_success_when_commit_fails(
    monkeypatch,
) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(commit_error=RuntimeError("commit failed"))
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/segment-assignments/build",
        json={"user_ids": ["user_001"]},
    )

    assert response.status_code == 500
    assert len(connections) == 1
    assert len(clickhouse_clients) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1


def test_segment_assignment_api_rolls_back_and_closes_on_failure(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(run_row=None)
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/missing_run/segment-assignments/build",
        json={"user_ids": ["user_001"]},
    )

    assert response.status_code == 404
    assert len(connections) == 1
    assert len(clickhouse_clients) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1


def test_segment_assignment_api_rolls_back_when_later_page_cursor_fails(
    monkeypatch,
) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.decision.assignment_service.ASSIGNMENT_PAGE_SIZE",
        1,
    )
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/segment-assignments/build",
        json={},
    )

    assert response.status_code == 422
    assert "cursor" in response.json()["detail"]
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert any(
        "insert into user_segment_assignments" in compact_sql(query)
        for query, _params in connection.executed
    )
    assert len(clickhouse_clients[0].calls) == 2
    assert clickhouse_clients[0].calls[1][1]["after_user_id"] == "user_001"


class FakeAssignmentService:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[tuple[str, SegmentAssignmentBuildRequest]] = []

    def build_assignments(
        self,
        *,
        promotion_run_id: str,
        request: SegmentAssignmentBuildRequest,
    ) -> SegmentAssignmentBuildResponse:
        self.calls.append((promotion_run_id, request))
        if self.exc is not None:
            raise self.exc
        return SegmentAssignmentBuildResponse(
            promotion_run_id=promotion_run_id,
            matching_mode="pgvector_hnsw_rerank",
            vector_version="v1",
            ann_candidate_limit=50,
            ann_candidate_count=1,
            exact_reranked_pair_count=1,
            page_count=1,
            processed_user_count=1,
            assignment_count=1,
            run_assignment_count=1,
            run_has_fallback=False,
            run_fallback_count=0,
            insert_conflict_count=0,
            segment_assignment_counts={"seg_family_trip": 1},
            batch_has_fallback=False,
            fallback_count=0,
            fallback_rate=0.0,
            fallback_reason_counts={
                "below_threshold": 0,
                "no_candidate": 0,
                "invalid_user_vector": 0,
            },
            below_threshold_fallback_count=0,
            no_candidate_fallback_count=0,
            invalid_user_vector_fallback_count=0,
            similarity_score_buckets={
                "not_available": 0,
                "0_00_to_0_50": 0,
                "0_50_to_0_65": 0,
                "0_65_to_0_80": 0,
                "0_80_to_0_90": 0,
                "gte_0_90": 1,
            },
            ann_underfilled_user_count=0,
            exact_rescue_user_count=0,
            ann_applied=True,
            ann_not_applied_reason=None,
            skipped_existing_count=0,
            insufficient_segment_count=0,
            completion_scope="current_request",
            assignment_mode="explicit_user_ids",
            input_stability="not_snapshotted",
            status="completed",
        )


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self._connection = connection
        self._last_query = ""

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: Any = None) -> None:
        self._last_query = query
        self._connection.executed.append((query, params))

    def fetchone(self) -> dict[str, object] | None:
        sql = compact_sql(self._last_query)
        if "from promotion_runs" in sql:
            return self._connection.run_row
        if "count(*) as assignment_count" in sql:
            return {"assignment_count": 1, "fallback_count": 0}
        return None

    def fetchall(self) -> list[dict[str, object]]:
        sql = compact_sql(self._last_query)
        if "insert into user_segment_assignments" in sql:
            return [
                {
                    "user_id": "user_001",
                    "segment_id": "seg_family_trip",
                    "fallback": False,
                    "fallback_reason": None,
                    "similarity_score": Decimal("1.000000"),
                }
            ]
        if "from ad_experiments" in sql:
            return [
                ad_experiment_row("seg_family_trip"),
                ad_experiment_row("seg_existing_all"),
            ]
        if "from segment_vectors" in sql and "order by embedding <=>" in sql:
            return [
                {
                    "query_user_id": "user_001",
                    "query_ordinal": 1,
                    "segment_vector_id": "segvec_family_v1",
                    "project_id": "hotel-client-a",
                    "promotion_id": "promo_banner_001",
                    "promotion_run_id": None,
                    "analysis_id": "analysis_banner_001",
                    "segment_id": "seg_family_trip",
                    "vector_dim": 64,
                    "vector_values": [1.0] + [0.0] * 63,
                    "vector_version": "v1",
                    "source": "decision_analysis",
                    "embedding": [1.0] + [0.0] * 63,
                }
            ]
        if "from segment_vectors" in sql:
            return [
                {
                    "segment_vector_id": "segvec_family_v1",
                    "project_id": "hotel-client-a",
                    "promotion_id": "promo_banner_001",
                    "promotion_run_id": None,
                    "analysis_id": "analysis_banner_001",
                    "segment_id": "seg_family_trip",
                    "vector_dim": 64,
                    "vector_values": [1.0] + [0.0] * 63,
                    "vector_version": "v1",
                    "source": "decision_analysis",
                    "embedding": [1.0] + [0.0] * 63,
                }
            ]
        if "from user_segment_assignments" in sql and "select user_id" in sql:
            return []
        if "from user_segment_assignments" in sql:
            return [
                {
                    "segment_id": "seg_family_trip",
                    "assigned_user_count": 1,
                }
            ]
        return []


class RecordingConnection:
    def __init__(
        self,
        run_row: dict[str, object] | None | object = DEFAULT_RUN_ROW,
        commit_error: Exception | None = None,
    ) -> None:
        self.run_row = promotion_run_row() if run_row is DEFAULT_RUN_ROW else run_row
        self.commit_error = commit_error
        self.executed: list[tuple[str, Any]] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self, **_kwargs: object) -> RecordingCursor:
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1
        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1


class RecordingClickHouseResult:
    result_rows = [
        (
            "hotel-client-a",
            "user_001",
            64,
            [1.0] + [0.0] * 63,
            "v1",
            "batch_profile",
        )
    ]


class RecordingClickHouseClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.close_count = 0

    def query(self, query: str, parameters: Any = None) -> RecordingClickHouseResult:
        self.calls.append((query, parameters))
        return RecordingClickHouseResult()

    def close(self) -> None:
        self.close_count += 1


def promotion_run_row() -> dict[str, object]:
    return {
        "promotion_run_id": "prun_banner_001_loop_1",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
        "status": "planned",
        "goal_snapshot_json": {"min_sample_size": 1},
        "segment_scope_json": ["seg_family_trip"],
        "segment_scope_fingerprint": "a" * 64,
    }


def ad_experiment_row(segment_id: str) -> dict[str, object]:
    return {
        "ad_experiment_id": f"adexp_{segment_id}",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "promotion_run_id": "prun_banner_001_loop_1",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "segment_id": segment_id,
        "segment_name": segment_id.replace("_", " "),
        "content_id": f"content_{segment_id}",
        "content_option_id": f"option_{segment_id}",
        "channel": "onsite_banner",
        "loop_count": 1,
        "status": "planned",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.030000"),
        "goal_basis": "all_segments",
    }
