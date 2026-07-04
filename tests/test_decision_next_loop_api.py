from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.next_loop_service import (
    NextLoopConflictError,
    NextLoopNotFoundError,
    NextLoopValidationError,
)
from app.decision.router import get_next_loop_service
from app.decision.schemas import (
    AdExperimentCreateResponse,
    AdExperimentStatus,
    Channel,
    NextLoopRequest,
    NextLoopResponse,
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
        app.dependency_overrides[get_next_loop_service] = lambda: service
    return TestClient(app)


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def test_next_loop_api_returns_response_shape() -> None:
    service = FakeNextLoopService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
            "operator_instruction": "Emphasize breakfast benefits.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["previous_promotion_run_id"] == "prun_banner_001_loop_1"
    assert body["next_promotion_run_id"] == "prun_banner_001_loop_2"
    assert body["promotion_id"] == "promo_banner_001"
    assert body["loop_count"] == 2
    assert body["next_analysis_id"] == "analysis_next_001"
    assert body["next_generation_id"] == "generation_next_001"
    assert body["next_ad_experiments"][0]["segment_id"] == "seg_luxury"
    assert "status" not in body
    assert isinstance(service.calls[0][1], NextLoopRequest)
    assert service.calls[0][1].operator_instruction == "Emphasize breakfast benefits."


def test_next_loop_api_rejects_extra_body() -> None:
    client = make_client(FakeNextLoopService())

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": [],
            "failed_ad_experiment_ids": [],
            "unexpected": True,
        },
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (NextLoopNotFoundError("missing run"), 404),
        (NextLoopValidationError("invalid next-loop input"), 422),
        (NextLoopConflictError("next run exists"), 409),
    ],
)
def test_next_loop_api_maps_service_errors(
    exc: Exception,
    expected_status: int,
) -> None:
    client = make_client(FakeNextLoopService(exc=exc))

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
        },
    )

    assert response.status_code == expected_status


def test_next_loop_api_wires_repositories_and_commits_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": [],
            "failed_ad_experiment_ids": [],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["next_promotion_run_id"] is None
    assert body["next_analysis_id"] is None
    assert body["next_generation_id"] is None
    assert body["next_ad_experiments"] == []
    assert "status" not in body
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("from promotion_runs" in query for query in executed_sql)


def test_next_loop_api_rolls_back_and_closes_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(promotion_run_row=None)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/missing/next-loop",
        json={
            "failed_segment_ids": [],
            "failed_ad_experiment_ids": [],
        },
    )

    assert response.status_code == 404
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


def test_next_loop_api_keeps_unavailable_production_adapter_until_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
        },
    )

    assert response.status_code == 422
    assert "integration is not configured" in response.json()["detail"]
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert not any("insert into promotion_runs" in query for query in executed_sql)
    assert not any("update promotion_runs" in query for query in executed_sql)


class FakeNextLoopService:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[tuple[str, NextLoopRequest]] = []

    def create_next_loop(
        self,
        *,
        promotion_run_id: str,
        request: NextLoopRequest,
    ) -> NextLoopResponse:
        self.calls.append((promotion_run_id, request))
        if self.exc is not None:
            raise self.exc
        return NextLoopResponse(
            previous_promotion_run_id=promotion_run_id,
            next_promotion_run_id="prun_banner_001_loop_2",
            promotion_id="promo_banner_001",
            loop_count=2,
            next_analysis_id="analysis_next_001",
            next_generation_id="generation_next_001",
            next_ad_experiments=[
                AdExperimentCreateResponse(
                    ad_experiment_id="adexp_luxury_002",
                    segment_id="seg_luxury",
                    segment_name="Luxury hotel users",
                    content_id="content_luxury_002",
                    content_option_id="option_luxury_002",
                    channel=Channel.ONSITE_BANNER,
                    loop_count=2,
                    status=AdExperimentStatus.PLANNED,
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

    def fetchone(self) -> dict[str, object] | None:
        sql = compact_sql(self._last_query)
        if "from promotion_runs" in sql and "where promotion_run_id = %s" in sql:
            return self._connection.promotion_run_row
        if "from promotion_runs" in sql and "where promotion_id = %s" in sql:
            return None
        if "from promotions" in sql:
            return promotion_row()
        return None

    def fetchall(self) -> list[dict[str, object]]:
        sql = compact_sql(self._last_query)
        if "from ad_experiments" in sql:
            return [ad_experiment_row()]
        if "from promotion_evaluations" in sql:
            return [promotion_evaluation_row()]
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


def default_promotion_run_row() -> dict[str, object]:
    return {
        "promotion_run_id": "prun_banner_001_loop_1",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
        "status": PromotionRunStatus.PARTIAL_GOAL_MET.value,
        "goal_snapshot_json": {
            "goal_metric": "booking_conversion_rate",
            "goal_target_value": "0.300000",
            "goal_basis": "all_segments",
            "min_sample_size": 10,
        },
    }


def promotion_row() -> dict[str, object]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "channel": "onsite_banner",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.300000"),
        "goal_basis": "all_segments",
        "min_sample_size": 10,
        "max_loop_count": 3,
    }


def ad_experiment_row() -> dict[str, object]:
    return {
        "ad_experiment_id": "adexp_luxury_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "promotion_run_id": "prun_banner_001_loop_1",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "segment_id": "seg_luxury",
        "segment_name": "Luxury hotel users",
        "content_id": "content_luxury_001",
        "content_option_id": "option_luxury_001",
        "channel": "onsite_banner",
        "loop_count": 1,
        "status": "goal_not_met",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.300000"),
        "goal_basis": "all_segments",
    }


def promotion_evaluation_row() -> dict[str, object]:
    return {
        "evaluation_id": "eval_luxury_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "promotion_run_id": "prun_banner_001_loop_1",
        "ad_experiment_id": "adexp_luxury_001",
        "segment_id": "seg_luxury",
        "content_id": "content_luxury_001",
        "content_option_id": "option_luxury_001",
        "metric": "booking_conversion_rate",
        "target_value": Decimal("0.300000"),
        "actual_value": Decimal("0.100000"),
        "numerator_count": 1,
        "denominator_count": 10,
        "sample_size": 10,
        "basis": "all_segments",
        "status": "goal_not_met",
        "feedback": None,
        "next_loop_required": True,
        "result_json": {"status_reason": "goal_not_met"},
    }
