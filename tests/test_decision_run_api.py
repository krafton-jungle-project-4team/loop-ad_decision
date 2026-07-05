from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi.testclient import TestClient
from psycopg import errors

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.router import get_promotion_run_service
from app.decision.schemas import (
    AdExperimentCreateResponse,
    Channel,
    PromotionRunStatus,
    RunCreateRequest,
    RunCreateResponse,
)
from app.decision.service import (
    PromotionNotFoundError,
    RunConflictError,
    RunValidationError,
)
from app.main import create_app


DEFAULT_ROW = object()


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
        app.dependency_overrides[get_promotion_run_service] = lambda: service
    return TestClient(app)


def test_run_api_returns_created_run_response_shape() -> None:
    service = FakeRunService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={"analysis_id": "analysis_banner_001", "loop_count": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["promotion_run_id"] == "prun_fake"
    assert payload["promotion_id"] == "promo_banner_001"
    assert payload["analysis_id"] == "analysis_banner_001"
    assert payload["generation_id"] == "generation_banner_001"
    assert payload["status"] == "planned"
    assert payload["goal_snapshot_json"]["goal_target_value"] == "0.030000"
    assert payload["ad_experiments"][0]["ad_experiment_id"] == "adexp_fake"
    assert service.calls[0][0] == "promo_banner_001"
    assert service.calls[0][1].loop_count == 2


def test_run_api_rejects_invalid_body() -> None:
    client = make_client(FakeRunService())

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={"unexpected": "field"},
    )

    assert response.status_code == 400
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_run_api_maps_service_errors() -> None:
    cases = [
        (PromotionNotFoundError("missing promotion"), 404),
        (RunValidationError("invalid run input"), 422),
        (RunConflictError("duplicate run"), 409),
        (errors.UniqueViolation("duplicate unique key"), 409),
    ]

    for exc, expected_status in cases:
        client = make_client(FakeRunService(exc=exc))

        response = client.post(
            "/decision/v1/promotions/promo_banner_001/runs",
            json={},
        )

        assert response.status_code == expected_status


def test_run_api_wires_repositories_and_commits(monkeypatch) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={},
    )

    assert response.status_code == 200
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into promotion_runs" in query for query in executed_sql)
    assert any("insert into ad_experiments" in query for query in executed_sql)


def test_run_api_rolls_back_when_service_validation_fails(monkeypatch) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(analysis_row=None)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={},
    )

    assert response.status_code == 422
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


def test_run_api_maps_unique_integrity_error_to_conflict_and_rolls_back(
    monkeypatch,
) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(raise_unique_on_insert=True)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={},
    )

    assert response.status_code == 409
    assert len(connections) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


class FakeRunService:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[tuple[str, RunCreateRequest]] = []

    def create_run(
        self,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        self.calls.append((promotion_id, request))
        if self.exc is not None:
            raise self.exc
        return RunCreateResponse(
            promotion_run_id="prun_fake",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id=promotion_id,
            analysis_id=request.analysis_id or "analysis_banner_001",
            generation_id=request.generation_id or "generation_banner_001",
            loop_count=request.loop_count,
            status=PromotionRunStatus.PLANNED,
            goal_snapshot_json={
                "source": "promotions",
                "goal_target_value": "0.030000",
            },
            ad_experiments=[
                AdExperimentCreateResponse(
                    ad_experiment_id="adexp_fake",
                    segment_id="seg_family_trip",
                    segment_name="Family hotel trip",
                    content_id="content_family_001",
                    content_option_id="family_option_a",
                    channel=Channel.ONSITE_BANNER,
                    loop_count=request.loop_count,
                    status="planned",
                )
            ],
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
        if (
            self._connection.raise_unique_on_insert
            and "insert into promotion_runs" in compact_sql(query)
        ):
            raise errors.UniqueViolation("duplicate run")

    def fetchone(self) -> dict[str, object] | None:
        sql = compact_sql(self._last_query)
        if "from promotions" in sql:
            return promotion_row()
        if "from promotion_analyses" in sql:
            return self._connection.analysis_row
        if "from generation_runs" in sql:
            return generation_row()
        if "from promotion_runs" in sql:
            return {"exists": 1} if self._connection.run_exists else None
        if "from ad_experiments" in sql:
            return None
        return None

    def fetchall(self) -> list[dict[str, object]]:
        sql = compact_sql(self._last_query)
        if "from promotion_target_segments" in sql:
            return [target_segment_row()]
        if "from content_candidates" in sql:
            return [content_candidate_row()]
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        analysis_row: dict[str, object] | None | object = DEFAULT_ROW,
        run_exists: bool = False,
        raise_unique_on_insert: bool = False,
    ) -> None:
        self.analysis_row = (
            analysis_row_ok() if analysis_row is DEFAULT_ROW else analysis_row
        )
        self.run_exists = run_exists
        self.raise_unique_on_insert = raise_unique_on_insert
        self.executed: list[tuple[str, Any]] = []
        self.row_factories: list[object] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self, *, row_factory: object = None) -> RecordingCursor:
        self.row_factories.append(row_factory)
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def promotion_row() -> dict[str, object]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "channel": "onsite_banner",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.030000"),
        "goal_basis": "all_segments",
        "min_sample_size": 1000,
        "max_loop_count": 3,
    }


def analysis_row_ok() -> dict[str, object]:
    return {
        "analysis_id": "analysis_banner_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "focus_segment_ids_json": None,
        "operator_instruction": None,
        "input_snapshot_json": {"promotion_id": "promo_banner_001"},
        "profile_summary_json": {"selected_segment_count": 1},
        "output_json": {"target_segment_count": 1},
        "status": "completed",
    }


def generation_row() -> dict[str, object]:
    return {
        "generation_id": "generation_banner_001",
        "analysis_id": "analysis_banner_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "content_option_count": 1,
        "operator_instruction": None,
        "input_json": {"analysis_id": "analysis_banner_001"},
        "output_json": {"content_count": 1},
        "generation_report_json": {"status": "completed"},
        "status": "completed",
    }


def target_segment_row() -> dict[str, object]:
    return {
        "analysis_id": "analysis_banner_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "segment_id": "seg_family_trip",
        "segment_name": "Family hotel trip",
        "segment_vector_id": "segvec_family_trip_v1",
        "rule_json": {"segment_id": "seg_family_trip"},
        "profile_json": {"segment_id": "seg_family_trip"},
        "content_brief_json": {"message_direction": "Highlight hotel benefits."},
        "data_evidence_json": {"event_count": 120},
        "estimated_size": 1200,
        "priority": "high",
        "status": "planned",
    }


def content_candidate_row() -> dict[str, object]:
    return {
        "content_id": "content_family_001",
        "content_option_id": "family_option_a",
        "generation_id": "generation_banner_001",
        "analysis_id": "analysis_banner_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "segment_id": "seg_family_trip",
        "channel": "onsite_banner",
        "status": "approved",
    }
