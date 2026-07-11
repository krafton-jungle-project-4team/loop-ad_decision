from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.evaluation_service import (
    AdExperimentEvaluationNotFoundError,
    AdExperimentEvaluationValidationError,
)
from app.decision.router import get_ad_experiment_evaluation_service
from app.decision.schemas import (
    AdExperimentEvaluateRequest,
    AdExperimentEvaluateResponse,
    EvaluationStrategySnapshot,
    GoalBasis,
    GoalMetric,
    PromotionEvaluationStatus,
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
        app.dependency_overrides[get_ad_experiment_evaluation_service] = (
            lambda: service
        )
    return TestClient(app)


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def test_ad_experiment_evaluation_api_returns_response_shape() -> None:
    service = FakeEvaluationService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/ad-experiments/adexp_family_trip_001/evaluate",
        json={},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["evaluation_id"] == "eval_adexp_family_trip_001"
    assert body["ad_experiment_id"] == "adexp_family_trip_001"
    assert body["promotion_run_id"] == "prun_banner_001_loop_1"
    assert body["promotion_id"] == "promo_banner_001"
    assert body["segment_id"] == "seg_family_trip"
    assert body["metric"] == GoalMetric.BOOKING_CONVERSION_RATE.value
    assert body["basis"] == GoalBasis.ALL_SEGMENTS.value
    assert body["status"] == PromotionEvaluationStatus.GOAL_NOT_MET.value
    assert body["target_gap"] == "-0.100000"
    assert body["status_reason"] == "target_not_met"
    assert body["strategy_snapshot"]["strategy_plan"] == {}
    assert body["next_loop_required"] is True
    assert body["feedback"] is None
    assert service.calls[0][0] == "adexp_family_trip_001"
    assert isinstance(service.calls[0][1], AdExperimentEvaluateRequest)


def test_ad_experiment_evaluation_api_rejects_extra_body() -> None:
    client = make_client(FakeEvaluationService())

    response = client.post(
        "/decision/v1/ad-experiments/adexp_family_trip_001/evaluate",
        json={"unexpected": True},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (AdExperimentEvaluationNotFoundError("missing experiment"), 404),
        (AdExperimentEvaluationValidationError("invalid evaluation input"), 422),
    ],
)
def test_ad_experiment_evaluation_api_maps_service_errors(
    exc: Exception,
    expected_status: int,
) -> None:
    client = make_client(FakeEvaluationService(exc=exc))

    response = client.post(
        "/decision/v1/ad-experiments/adexp_family_trip_001/evaluate",
        json={},
    )

    assert response.status_code == expected_status


def test_ad_experiment_evaluation_api_wires_repositories_and_commits(monkeypatch) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection()
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient(rows=[(2, 10)])
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
        "/decision/v1/ad-experiments/adexp_family_trip_001/evaluate",
        json={},
    )

    assert response.status_code == 200
    assert response.json()["status"] == PromotionEvaluationStatus.GOAL_NOT_MET.value
    assert len(connections) == 1
    assert len(clickhouse_clients) == 1
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("from ad_experiments" in query for query in executed_sql)
    assert any("from promotion_runs" in query for query in executed_sql)
    assert any("from content_candidates" in query for query in executed_sql)
    assert any("insert into promotion_evaluations" in query for query in executed_sql)
    assert any("update ad_experiments" in query for query in executed_sql)


@pytest.mark.parametrize(
    ("field_name", "mismatched_value"),
    [
        ("content_option_id", "option_b"),
        ("project_id", "other-project"),
        ("campaign_id", "other-campaign"),
        ("promotion_id", "other-promotion"),
        ("segment_id", "other-segment"),
        ("generation_id", "other-generation"),
    ],
)
def test_ad_experiment_evaluation_api_rejects_candidate_context_before_writes(
    monkeypatch,
    field_name: str,
    mismatched_value: str,
) -> None:
    candidate = {**default_content_candidate_row(), field_name: mismatched_value}
    connection = RecordingConnection(content_candidate_row=candidate)
    clickhouse_client = RecordingClickHouseClient(rows=[(2, 10)])
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/ad-experiments/adexp_family_trip_001/evaluate",
        json={},
    )

    assert response.status_code == 422
    assert field_name in response.json()["detail"]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert not any("insert into promotion_evaluations" in query for query in executed_sql)
    assert not any("update ad_experiments" in query for query in executed_sql)
    assert clickhouse_client.queries == []


def test_ad_experiment_evaluation_api_rolls_back_and_closes_on_failure(
    monkeypatch,
) -> None:
    connections: list[RecordingConnection] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(ad_experiment_row=None)
        connections.append(connection)
        return connection

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient(rows=[(0, 0)])
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
        "/decision/v1/ad-experiments/missing_adexp/evaluate",
        json={},
    )

    assert response.status_code == 404
    assert len(connections) == 1
    assert len(clickhouse_clients) == 1
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    assert clickhouse_clients[0].close_count == 1


class FakeEvaluationService:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[tuple[str, AdExperimentEvaluateRequest]] = []

    def evaluate(
        self,
        *,
        ad_experiment_id: str,
        request: AdExperimentEvaluateRequest,
    ) -> AdExperimentEvaluateResponse:
        self.calls.append((ad_experiment_id, request))
        if self.exc is not None:
            raise self.exc
        return AdExperimentEvaluateResponse(
            evaluation_id="eval_adexp_family_trip_001",
            ad_experiment_id=ad_experiment_id,
            promotion_run_id="prun_banner_001_loop_1",
            promotion_id="promo_banner_001",
            segment_id="seg_family_trip",
            metric=GoalMetric.BOOKING_CONVERSION_RATE,
            target_value=Decimal("0.300000"),
            actual_value=Decimal("0.200000"),
            target_gap=Decimal("-0.100000"),
            numerator_count=2,
            denominator_count=10,
            sample_size=10,
            basis=GoalBasis.ALL_SEGMENTS,
            status=PromotionEvaluationStatus.GOAL_NOT_MET,
            status_reason="target_not_met",
            strategy_snapshot=EvaluationStrategySnapshot(
                strategy_key="booking_confidence__family",
                strategy_plan={},
                evidence_refs=[],
                brief_fingerprint="sha256:brief",
                prompt_builder_version="dec-c2.v4",
                fallback_guidance_used=False,
            ),
            next_loop_required=True,
            feedback=None,
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
        if "from ad_experiments" in sql:
            return self._connection.ad_experiment_row
        if "from promotion_runs" in sql:
            return self._connection.promotion_run_row
        if "from content_candidates" in sql:
            return self._connection.content_candidate_row
        return None

    def fetchall(self) -> list[dict[str, object]]:
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        ad_experiment_row: dict[str, object] | None | object = DEFAULT_ROW,
        promotion_run_row: dict[str, object] | None | object = DEFAULT_ROW,
        content_candidate_row: dict[str, object] | None | object = DEFAULT_ROW,
    ) -> None:
        self.ad_experiment_row = (
            default_ad_experiment_row()
            if ad_experiment_row is DEFAULT_ROW
            else ad_experiment_row
        )
        self.promotion_run_row = (
            default_promotion_run_row()
            if promotion_run_row is DEFAULT_ROW
            else promotion_run_row
        )
        self.content_candidate_row = (
            default_content_candidate_row()
            if content_candidate_row is DEFAULT_ROW
            else content_candidate_row
        )
        self.executed: list[tuple[str, Any]] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self, *_args: object, **_kwargs: object) -> RecordingCursor:
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1


def default_ad_experiment_row() -> dict[str, object]:
    return {
        "ad_experiment_id": "adexp_family_trip_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "promotion_run_id": "prun_banner_001_loop_1",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "segment_id": "seg_family_trip",
        "segment_name": "Family hotel trip",
        "content_id": "content_family_trip_001",
        "content_option_id": "option_a",
        "channel": "onsite_banner",
        "loop_count": 1,
        "status": "running",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.300000"),
        "goal_basis": "all_segments",
    }


def default_promotion_run_row() -> dict[str, object]:
    return {
        "promotion_run_id": "prun_banner_001_loop_1",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
        "status": "running",
        "goal_snapshot_json": {
            "goal_target_value": "0.300000",
            "min_sample_size": 10,
        },
    }


def default_content_candidate_row() -> dict[str, object]:
    return {
        "content_id": "content_family_trip_001",
        "content_option_id": "option_a",
        "generation_id": "generation_banner_001",
        "analysis_id": "analysis_banner_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "segment_id": "seg_family_trip",
        "channel": "onsite_banner",
        "status": "approved",
        "metadata_json": {
            "strategy_key": "booking_confidence__family",
            "strategy_plan": {},
            "evidence_refs": [],
            "brief_fingerprint": "sha256:brief",
            "prompt_builder_version": "dec-c2.v4",
            "fallback_guidance_used": False,
        },
    }


class RecordingClickHouseResult:
    def __init__(self, rows: list[Any]) -> None:
        self.result_rows = rows


class RecordingClickHouseClient:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, Any]] = []
        self.close_count = 0

    def query(self, query: str, parameters: Any = None) -> RecordingClickHouseResult:
        self.queries.append((query, parameters))
        return RecordingClickHouseResult(self.rows)

    def close(self) -> None:
        self.close_count += 1
