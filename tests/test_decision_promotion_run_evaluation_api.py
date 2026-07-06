from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.evaluation_service import (
    PromotionRunEvaluationNotFoundError,
    PromotionRunEvaluationValidationError,
)
from app.decision.router import get_promotion_run_evaluation_service
from app.decision.schemas import (
    PromotionEvaluationStatus,
    PromotionRunAdExperimentResult,
    PromotionRunEvaluateRequest,
    PromotionRunEvaluateResponse,
    PromotionRunStatus,
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
        app.dependency_overrides[get_promotion_run_evaluation_service] = (
            lambda: service
        )
    return TestClient(app)


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def test_promotion_run_evaluation_api_returns_response_shape() -> None:
    service = FakePromotionRunEvaluationService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/evaluate",
        json={},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["promotion_run_id"] == "prun_banner_001_loop_1"
    assert body["promotion_id"] == "promo_banner_001"
    assert body["status"] == PromotionRunStatus.PARTIAL_GOAL_MET.value
    assert body["next_loop_required"] is True
    assert body["failed_segment_ids"] == ["seg_luxury"]
    assert body["failed_ad_experiment_ids"] == ["adexp_luxury_001"]
    assert body["ad_experiment_results"][0]["ad_experiment_id"] == (
        "adexp_family_trip_001"
    )
    assert isinstance(service.calls[0][1], PromotionRunEvaluateRequest)


def test_promotion_run_evaluation_api_rejects_extra_body() -> None:
    client = make_client(FakePromotionRunEvaluationService())

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/evaluate",
        json={"unexpected": True},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (PromotionRunEvaluationNotFoundError("missing run"), 404),
        (PromotionRunEvaluationValidationError("invalid aggregate input"), 422),
    ],
)
def test_promotion_run_evaluation_api_maps_service_errors(
    exc: Exception,
    expected_status: int,
) -> None:
    client = make_client(FakePromotionRunEvaluationService(exc=exc))

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/evaluate",
        json={},
    )

    assert response.status_code == expected_status


def test_promotion_run_evaluation_api_wires_repositories_and_commits(
    monkeypatch,
) -> None:
    connections: list[RecordingConnection] = []
    pools: list[RecordingPool] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_pool(_settings) -> RecordingPool:
        connection = RecordingConnection()
        connections.append(connection)
        pool = RecordingPool(connection)
        pools.append(pool)
        return pool

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.dependencies.create_postgres_pool",
        fake_create_postgres_pool,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/evaluate",
        json={},
    )

    assert response.status_code == 200
    assert response.json()["status"] == PromotionRunStatus.GOAL_MET.value
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 0
    assert pools[0].checkout_count == 1
    assert pools[0].return_count == 1
    assert clickhouse_clients[0].close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("from promotion_runs" in query for query in executed_sql)
    assert any("from ad_experiments" in query for query in executed_sql)
    assert any("from promotion_evaluations" in query for query in executed_sql)
    assert any("insert into promotion_evaluations" in query for query in executed_sql)
    assert any("update promotion_runs" in query for query in executed_sql)


def test_promotion_run_evaluation_api_rolls_back_and_closes_on_failure(
    monkeypatch,
) -> None:
    connections: list[RecordingConnection] = []
    pools: list[RecordingPool] = []
    clickhouse_clients: list[RecordingClickHouseClient] = []

    def fake_create_postgres_pool(_settings) -> RecordingPool:
        connection = RecordingConnection(promotion_run_row=None)
        connections.append(connection)
        pool = RecordingPool(connection)
        pools.append(pool)
        return pool

    def fake_create_clickhouse_client(_settings) -> RecordingClickHouseClient:
        client = RecordingClickHouseClient()
        clickhouse_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.dependencies.create_postgres_pool",
        fake_create_postgres_pool,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        fake_create_clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/missing/evaluate",
        json={},
    )

    assert response.status_code == 404
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 0
    assert pools[0].checkout_count == 1
    assert pools[0].return_count == 1
    assert clickhouse_clients[0].close_count == 1


class FakePromotionRunEvaluationService:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[tuple[str, PromotionRunEvaluateRequest]] = []

    def evaluate(
        self,
        *,
        promotion_run_id: str,
        request: PromotionRunEvaluateRequest,
    ) -> PromotionRunEvaluateResponse:
        self.calls.append((promotion_run_id, request))
        if self.exc is not None:
            raise self.exc
        return PromotionRunEvaluateResponse(
            promotion_run_id=promotion_run_id,
            promotion_id="promo_banner_001",
            status=PromotionRunStatus.PARTIAL_GOAL_MET,
            ad_experiment_results=[
                PromotionRunAdExperimentResult(
                    ad_experiment_id="adexp_family_trip_001",
                    segment_id="seg_family_trip",
                    actual_value=Decimal("0.400000"),
                    status=PromotionEvaluationStatus.GOAL_MET,
                ),
                PromotionRunAdExperimentResult(
                    ad_experiment_id="adexp_luxury_001",
                    segment_id="seg_luxury",
                    actual_value=Decimal("0.100000"),
                    status=PromotionEvaluationStatus.GOAL_NOT_MET,
                ),
            ],
            next_loop_required=True,
            failed_segment_ids=["seg_luxury"],
            failed_ad_experiment_ids=["adexp_luxury_001"],
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
            return self._connection.promotion_run_row
        return None

    def fetchall(self) -> list[dict[str, object]]:
        sql = compact_sql(self._last_query)
        if "from ad_experiments" in sql:
            return self._connection.ad_experiment_rows
        if "from promotion_evaluations" in sql:
            return self._connection.evaluation_rows
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        promotion_run_row: dict[str, object] | None | object = DEFAULT_ROW,
    ) -> None:
        self.promotion_run_row = (
            default_promotion_run_row()
            if promotion_run_row is DEFAULT_ROW
            else promotion_run_row
        )
        self.ad_experiment_rows = default_ad_experiment_rows()
        self.evaluation_rows = default_evaluation_rows()
        self.executed: list[tuple[str, Any]] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self, row_factory: Any = None) -> RecordingCursor:
        _ = row_factory
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1


class RecordingPool:
    def __init__(self, connection: RecordingConnection) -> None:
        self.connection_object = connection
        self.checkout_count = 0
        self.return_count = 0
        self.close_count = 0

    @contextmanager
    def connection(self) -> Iterator[RecordingConnection]:
        self.checkout_count += 1
        try:
            yield self.connection_object
        finally:
            self.return_count += 1

    def close(self) -> None:
        self.close_count += 1


class RecordingClickHouseClient:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def default_promotion_run_row() -> dict[str, object]:
    return {
        "promotion_run_id": "prun_banner_001_loop_1",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
        "status": PromotionRunStatus.RUNNING.value,
        "goal_snapshot_json": {
            "goal_metric": "booking_conversion_rate",
            "goal_target_value": "0.300000",
            "goal_basis": "all_segments",
            "min_sample_size": 10,
        },
    }


def default_ad_experiment_rows() -> list[dict[str, object]]:
    return [
        {
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
    ]


def default_evaluation_rows() -> list[dict[str, object]]:
    return [
        {
            "evaluation_id": "eval_adexp_family_trip_001",
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "promotion_run_id": "prun_banner_001_loop_1",
            "ad_experiment_id": "adexp_family_trip_001",
            "segment_id": "seg_family_trip",
            "content_id": "content_family_trip_001",
            "content_option_id": "option_a",
            "metric": "booking_conversion_rate",
            "target_value": Decimal("0.300000"),
            "actual_value": Decimal("0.400000"),
            "numerator_count": 4,
            "denominator_count": 10,
            "sample_size": 10,
            "basis": "all_segments",
            "status": "goal_met",
            "feedback": None,
            "next_loop_required": False,
            "result_json": {"status_reason": "target_met"},
        }
    ]
