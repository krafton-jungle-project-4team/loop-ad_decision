from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import os
import threading
import time
from typing import Any
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import errors
from psycopg import sql

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.next_loop_service import (
    NextLoopConflictError,
    NextLoopNotFoundError,
    NextLoopValidationError,
)
from app.decision.repositories import (
    NextLoopPreparationRepository,
    NextLoopPreparationWrite,
    PsycopgPostgresExecutor,
)
from app.decision.router import (
    SerializedNextLoopPreparationRepository,
    get_next_loop_service,
)
from app.decision.schemas import (
    AdExperimentCreateResponse,
    AdExperimentStatus,
    Channel,
    ContentApprovalMode,
    NextLoopPreparationStatus,
    NextLoopRequest,
    NextLoopResponse,
    PromotionRunStatus,
)
from app.main import create_app
from app.generation.repositories import ContentCandidateRepository


DEFAULT_ROW = object()


class _FakeDiag:
    def __init__(self, constraint_name: str) -> None:
        self.constraint_name = constraint_name


class UniqueViolationWithConstraint(errors.UniqueViolation):
    def __init__(self, message: str, constraint_name: str) -> None:
        super().__init__(message)
        self._constraint_name = constraint_name

    @property
    def diag(self) -> _FakeDiag:
        return _FakeDiag(self._constraint_name)


def valid_env() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED": "true",
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
    assert [experiment["segment_id"] for experiment in body["next_ad_experiments"]] == [
        "seg_luxury"
    ]
    assert set(body) == {
        "previous_promotion_run_id",
        "next_promotion_run_id",
        "promotion_id",
        "loop_count",
        "segment_ids",
        "next_analysis_id",
        "next_generation_id",
        "next_ad_experiments",
    }
    assert isinstance(service.calls[0][1], NextLoopRequest)
    assert service.calls[0][1].operator_instruction == "Emphasize breakfast benefits."
    assert (
        service.calls[0][1].content_approval_mode
        is ContentApprovalMode.AUTOMATIC
    )


def test_next_loop_api_parses_manual_mode_without_changing_automatic_response() -> None:
    service = FakeNextLoopService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
            "content_approval_mode": "manual",
        },
    )

    assert response.status_code == 200
    assert (
        service.calls[0][1].content_approval_mode is ContentApprovalMode.MANUAL
    )
    assert set(response.json()) == {
        "previous_promotion_run_id",
        "next_promotion_run_id",
        "promotion_id",
        "loop_count",
        "segment_ids",
        "next_analysis_id",
        "next_generation_id",
        "next_ad_experiments",
    }


def test_next_loop_response_serializes_explicit_manual_contract_fields() -> None:
    response = NextLoopResponse(
        status=NextLoopPreparationStatus.AWAITING_CONTENT_APPROVAL,
        content_approval_required=True,
        next_loop_preparation_id="nlprep_banner_001_loop_2_attempt_1",
        previous_promotion_run_id="prun_banner_001_loop_1",
        next_promotion_run_id=None,
        promotion_id="promo_banner_001",
        loop_count=2,
        segment_ids=["seg_luxury"],
        next_analysis_id="analysis_next_001",
        next_generation_id="generation_next_001",
        pending_content_ids=[],
        next_ad_experiments=[],
    )

    assert response.model_dump(mode="json") == {
        "status": "awaiting_content_approval",
        "content_approval_required": True,
        "next_loop_preparation_id": "nlprep_banner_001_loop_2_attempt_1",
        "previous_promotion_run_id": "prun_banner_001_loop_1",
        "next_promotion_run_id": None,
        "promotion_id": "promo_banner_001",
        "loop_count": 2,
        "segment_ids": ["seg_luxury"],
        "next_analysis_id": "analysis_next_001",
        "next_generation_id": "generation_next_001",
        "pending_content_ids": [],
        "next_ad_experiments": [],
    }


def test_next_loop_response_keeps_explicit_manual_default_values() -> None:
    response = NextLoopResponse(
        status=NextLoopPreparationStatus.ACTIVATED,
        content_approval_required=False,
        next_loop_preparation_id=None,
        previous_promotion_run_id="prun_banner_001_loop_1",
        next_promotion_run_id="prun_banner_001_loop_2",
        promotion_id="promo_banner_001",
        loop_count=2,
        segment_ids=["seg_luxury"],
        next_analysis_id="analysis_next_001",
        next_generation_id="generation_next_001",
        pending_content_ids=[],
        next_ad_experiments=[],
    )

    serialized = response.model_dump(mode="json")

    assert serialized["status"] == "activated"
    assert serialized["content_approval_required"] is False
    assert serialized["next_loop_preparation_id"] is None
    assert serialized["pending_content_ids"] == []


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
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
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
    assert {
        "status",
        "content_approval_required",
        "next_loop_preparation_id",
        "pending_content_ids",
    }.isdisjoint(body)
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("from promotion_runs" in query for query in executed_sql)
    assert not any("next_loop_preparations" in query for query in executed_sql)


def test_manual_next_loop_flag_off_rejects_before_any_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = RecordingConnection()
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
            "content_approval_mode": "manual",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "manual next-loop preparation is disabled"
    assert connection.executed == []
    assert connection.commit_count == 0
    assert connection.rollback_count == 1


def test_manual_next_loop_api_persists_preparation_and_returns_pending_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = RecordingConnection()
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    app = create_app(settings=load_settings(valid_env()))
    app.state.manual_next_loop_enabled = True
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury", "seg_luxury"],
            "failed_ad_experiment_ids": [
                "adexp_luxury_001",
                "adexp_luxury_001",
            ],
            "content_approval_mode": "manual",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "awaiting_content_approval"
    assert body["content_approval_required"] is True
    assert body["next_promotion_run_id"] is None
    assert body["next_ad_experiments"] == []
    assert body["next_generation_id"].endswith(
        "_loop_2_1d7b63967183_attempt_1"
    )
    assert len(body["pending_content_ids"]) == 3
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into next_loop_preparations" in query for query in executed_sql)
    assert any("pg_advisory_xact_lock" in query for query in executed_sql)
    assert not any("insert into promotion_runs" in query for query in executed_sql)
    assert not any("insert into ad_experiments" in query for query in executed_sql)
    evaluation_sql = next(
        query for query in executed_sql if "from promotion_evaluations" in query
    )
    assert "created_at desc, evaluation_id desc" in evaluation_sql


def test_manual_next_loop_api_reuses_canonical_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = RecordingConnection()
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    app = create_app(settings=load_settings(valid_env()))
    app.state.manual_next_loop_enabled = True
    client = TestClient(app)
    payload = {
        "failed_segment_ids": ["seg_luxury"],
        "failed_ad_experiment_ids": ["adexp_luxury_001"],
        "content_approval_mode": "manual",
    }

    first = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json=payload,
    )
    second = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json=payload,
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json() == first.json()
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert sum("insert into next_loop_preparations" in query for query in executed_sql) == 1
    assert sum("insert into generation_runs" in query for query in executed_sql) == 1
    assert sum("insert into content_candidates" in query for query in executed_sql) == 3
    assert sum("pg_advisory_xact_lock" in query for query in executed_sql) == 2
    preparation_lock_indices = [
        index
        for index, query in enumerate(executed_sql)
        if "from next_loop_preparations" in query and "for update" in query
    ]
    candidate_lock_indices = [
        index
        for index, query in enumerate(executed_sql)
        if "from content_candidates" in query and "for update" in query
    ]
    assert preparation_lock_indices
    assert candidate_lock_indices
    assert preparation_lock_indices[-1] < candidate_lock_indices[-1]
    candidate_lock_sql = executed_sql[candidate_lock_indices[-1]]
    assert "order by segment_id, content_option_id, content_id" in candidate_lock_sql
    assert connection.commit_count == 2
    assert connection.rollback_count == 0


def test_manual_next_loop_api_rolls_back_generation_on_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = RecordingConnection(
        raise_unique_on_content_candidate_insert=True,
    )
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    app = create_app(settings=load_settings(valid_env()))
    app.state.manual_next_loop_enabled = True
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
            "content_approval_mode": "manual",
        },
    )

    assert response.status_code == 409
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.rolled_back_inserts == ["generation_runs"]
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert not any("insert into next_loop_preparations" in query for query in executed_sql)
    assert not any("insert into promotion_runs" in query for query in executed_sql)
    assert not any("insert into ad_experiments" in query for query in executed_sql)


def test_manual_next_loop_api_regenerates_exhausted_candidates_as_attempt_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = manual_regeneration_connection()
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    app = create_app(settings=load_settings(valid_env()))
    app.state.manual_next_loop_enabled = True
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
            "content_approval_mode": "manual",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["next_analysis_id"] == "analysis_banner_001_loop_2_1d7b63967183"
    assert body["next_generation_id"].endswith(
        "_loop_2_1d7b63967183_attempt_2"
    )
    assert len(body["pending_content_ids"]) == 3
    assert all("attempt_2" in content_id for content_id in body["pending_content_ids"])
    assert [row["status"] for row in connection.rejected_preparation_rows] == [
        "rejected"
    ]
    assert connection.active_preparation_row is not None
    assert connection.active_preparation_row["attempt_no"] == 2
    assert connection.active_preparation_row["status"] == (
        "awaiting_content_approval"
    )
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    preparation_lock = next(
        index
        for index, query in enumerate(executed_sql)
        if "from next_loop_preparations" in query and "for update" in query
    )
    candidate_lock = next(
        index
        for index, query in enumerate(executed_sql)
        if "from content_candidates" in query and "for update" in query
    )
    reject_update = next(
        index
        for index, query in enumerate(executed_sql)
        if "update next_loop_preparations" in query
        and "status = 'rejected'" in query
    )
    assert preparation_lock < candidate_lock < reject_update
    assert not any("insert into promotion_analyses" in query for query in executed_sql)
    assert sum("insert into generation_runs" in query for query in executed_sql) == 1
    assert sum("insert into content_candidates" in query for query in executed_sql) == 3
    assert sum("insert into next_loop_preparations" in query for query in executed_sql) == 1
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_manual_next_loop_api_rolls_back_rejection_and_attempt_two_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = manual_regeneration_connection(
        raise_unique_on_content_candidate_insert=True,
    )
    old_candidate_ids = {
        str(candidate["content_id"])
        for candidate in connection.content_candidate_rows
    }
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    app = create_app(settings=load_settings(valid_env()))
    app.state.manual_next_loop_enabled = True
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
            "content_approval_mode": "manual",
        },
    )

    assert response.status_code == 409
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.active_preparation_row is not None
    assert connection.active_preparation_row["attempt_no"] == 1
    assert connection.active_preparation_row["status"] == (
        "awaiting_content_approval"
    )
    assert {
        str(candidate["content_id"])
        for candidate in connection.content_candidate_rows
    } == old_candidate_ids
    assert connection.rolled_back_inserts == ["generation_runs"]


def test_next_loop_service_wires_s3_artifact_publisher_outside_test_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings({**valid_env(), "LOOPAD_ENV": "dev"})
    artifact_publisher = object()

    monkeypatch.setattr(
        "app.decision.router.get_settings",
        lambda _request: settings,
    )
    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        lambda _settings: RecordingConnection(),
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    monkeypatch.setattr(
        "app.decision.router.build_external_content_generator",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        "app.decision.router.build_s3_creative_artifact_publisher",
        lambda _settings: artifact_publisher,
    )

    service_iterator = get_next_loop_service(object())
    service = next(service_iterator)
    try:
        generation_gateway = service._generation_gateway
        generation_service = generation_gateway._generation_service
        assert generation_service._artifact_publisher is artifact_publisher
    finally:
        service_iterator.close()


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
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
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


def test_next_loop_api_maps_segment_vector_data_unavailable_to_validation_error(
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
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(user_vector_rows=[]),
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
    assert response.json()["detail"] == "segment vector data unavailable"
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


def test_next_loop_api_flag_off_rejects_before_lifecycle_writes(
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
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
    )
    env = valid_env()
    env["LOOPAD_PARTIAL_PROMOTION_RUN_SCOPE_ENABLED"] = "false"
    app = create_app(settings=load_settings(env))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": ["seg_luxury"],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
        },
    )

    assert response.status_code == 409
    connection = connections[0]
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert not any(query.startswith("insert ") for query in executed_sql)
    assert not any(query.startswith("update ") for query in executed_sql)
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


def test_next_loop_api_wires_focus_analysis_generation_and_creates_next_run(
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
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
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

    assert response.status_code == 200
    body = response.json()
    assert body["previous_promotion_run_id"] == "prun_banner_001_loop_1"
    assert body["next_promotion_run_id"].startswith("prun_promo_banner_001_loop_2")
    assert body["loop_count"] == 2
    assert body["next_analysis_id"] == "analysis_banner_001_loop_2_1d7b63967183"
    assert body["next_generation_id"] == (
        "generation_banner_001_loop_2_1d7b63967183"
    )
    assert [
        experiment["segment_id"] for experiment in body["next_ad_experiments"]
    ] == ["seg_luxury", "seg_existing_all"]
    assert "status" not in body
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into promotion_analyses" in query for query in executed_sql)
    assert any("insert into promotion_target_segments" in query for query in executed_sql)
    assert any("insert into generation_runs" in query for query in executed_sql)
    assert any("insert into content_candidates" in query for query in executed_sql)
    assert any("insert into promotion_runs" in query for query in executed_sql)
    assert any("insert into ad_experiments" in query for query in executed_sql)
    assert not any("update promotion_runs" in query for query in executed_sql)
    content_insert_params = next(
        params
        for query, params in connection.executed
        if "insert into content_candidates" in compact_sql(query)
    )
    assert content_insert_params["status"] == "approved"


def test_next_loop_api_reuses_stored_ai_segment_without_recommending_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_segment_id = (
        "seg_ai_raw_promo_banner_001_"
        "target_destination_affinity_membership"
    )
    connections: list[RecordingConnection] = []
    clickhouse_client = FakeClickHouseClient(fail_on_recommendation_query=True)

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(segment_id=stored_segment_id)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: clickhouse_client,
    )
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app)

    response = client.post(
        "/decision/v1/promotion-runs/prun_banner_001_loop_1/next-loop",
        json={
            "failed_segment_ids": [stored_segment_id],
            "failed_ad_experiment_ids": ["adexp_luxury_001"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["next_analysis_id"] == "analysis_banner_001_loop_2_1d7b63967183"
    assert body["next_generation_id"] == (
        "generation_banner_001_loop_2_1d7b63967183"
    )
    assert [
        experiment["segment_id"] for experiment in body["next_ad_experiments"]
    ] == [stored_segment_id, "seg_existing_all"]
    assert not any("from raw_events" in query for query in clickhouse_client.queries)
    connection = connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into promotion_analyses" in query for query in executed_sql)
    assert any("insert into promotion_target_segments" in query for query in executed_sql)
    assert not any(
        "insert into promotion_segment_suggestions" in query
        for query in executed_sql
    )
    assert any("insert into generation_runs" in query for query in executed_sql)
    assert any("insert into promotion_runs" in query for query in executed_sql)
    assert any("insert into ad_experiments" in query for query in executed_sql)


def test_next_loop_api_maps_approved_content_unique_violation_to_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            raise_unique_on_content_candidate_insert=True,
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
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

    assert response.status_code == 409
    assert response.json()["detail"] == "next-loop output already exists"
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into generation_runs" in query for query in executed_sql)
    assert any("insert into content_candidates" in query for query in executed_sql)
    assert not any("insert into promotion_runs" in query for query in executed_sql)


def test_next_loop_api_maps_approved_content_constraint_to_specific_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            raise_unique_on_content_candidate_insert=True,
            content_candidate_unique_constraint_name=(
                "uq_content_candidates_one_approved_per_segment"
            ),
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
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

    assert response.status_code == 409
    assert response.json()["detail"] == "approved content already exists for segment"
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


def test_next_loop_api_rolls_back_approved_content_when_run_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            raise_unique_on_promotion_run_insert=True,
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(
        "app.decision.router.create_clickhouse_client",
        lambda _settings: FakeClickHouseClient(),
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

    assert response.status_code == 409
    assert response.json()["detail"] == "next-loop output already exists"
    connection = connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into content_candidates" in query for query in executed_sql)
    assert any("insert into promotion_runs" in query for query in executed_sql)
    assert not any("insert into ad_experiments" in query for query in executed_sql)
    assert connection.committed_inserts == []
    assert connection.pending_inserts == []
    assert connection.rolled_back_inserts == [
        "generation_runs",
        "content_candidates",
    ]


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
            segment_ids=["seg_luxury"],
            next_analysis_id="analysis_next_001",
            next_generation_id="generation_next_001",
            next_ad_experiments=[
                AdExperimentCreateResponse(
                    ad_experiment_id="adexp_luxury_loop_2",
                    segment_id="seg_luxury",
                    segment_name="Luxury hotel users",
                    content_id="content_luxury_next",
                    content_option_id="option_luxury_next",
                    channel=Channel.ONSITE_BANNER,
                    loop_count=2,
                    status=AdExperimentStatus.PLANNED,
                    is_fallback=False,
                )
            ],
        )


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self._connection = connection
        self._last_query = ""
        self._last_params: Any = None

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: Any = None) -> None:
        self._last_query = query
        self._last_params = params
        self._connection.executed.append((query, params))
        sql = compact_sql(query)
        if (
            self._connection.raise_unique_on_content_candidate_insert
            and "insert into content_candidates" in sql
        ):
            if self._connection.content_candidate_unique_constraint_name is not None:
                raise UniqueViolationWithConstraint(
                    "duplicate approved content candidate",
                    self._connection.content_candidate_unique_constraint_name,
                )
            raise errors.UniqueViolation("duplicate approved content candidate")
        if (
            self._connection.raise_unique_on_promotion_run_insert
            and "insert into promotion_runs" in sql
        ):
            raise errors.UniqueViolation("duplicate promotion run")
        table_name = _inserted_table_name(sql)
        if table_name is not None:
            self._connection.pending_inserts.append(table_name)
        if "insert into content_candidates" in sql:
            self._connection.content_candidate_rows.append(
                content_candidate_insert_row(params)
            )

    def fetchone(self) -> dict[str, object] | None:
        sql = compact_sql(self._last_query)
        if "insert into promotion_runs" in sql:
            return {"promotion_run_id": "prun_inserted"}
        if "insert into generation_runs" in sql:
            return generation_run_insert_row(self._last_params)
        if "insert into content_candidates" in sql:
            return content_candidate_insert_row(self._last_params)
        if "insert into next_loop_preparations" in sql:
            row = next_loop_preparation_insert_row(self._last_params)
            self._connection.active_preparation_row = row
            return row
        if "update next_loop_preparations" in sql and "status = 'rejected'" in sql:
            active = self._connection.active_preparation_row
            if active is None or active["status"] != "awaiting_content_approval":
                return None
            self._connection.preparation_before_rejection = dict(active)
            rejected = {
                **active,
                "status": "rejected",
                "updated_at": datetime.now(UTC),
            }
            self._connection.rejected_preparation_rows.append(rejected)
            self._connection.active_preparation_row = None
            return rejected
        if "select coalesce(max(attempt_no), 0) + 1" in sql:
            return {"next_attempt_no": self._connection.next_attempt_no}
        if "from next_loop_preparations" in sql:
            return self._connection.active_preparation_row
        if "from promotion_analyses" in sql:
            return promotion_analysis_row(self._connection.segment_id)
        if "from generation_runs" in sql:
            return generation_run_row(
                self._last_params,
                segment_id=self._connection.segment_id,
            )
        if "from promotion_runs" in sql and "where promotion_run_id = %s" in sql:
            return self._connection.promotion_run_row
        if "from promotion_runs" in sql and "where promotion_id = %s" in sql:
            return None
        if "from segment_vectors" in sql:
            return None
        if "from promotions" in sql:
            return promotion_row(sql)
        return None

    def fetchall(self) -> list[dict[str, object]]:
        sql = compact_sql(self._last_query)
        if "from ad_experiments" in sql:
            return [ad_experiment_row(self._connection.segment_id)]
        if "from promotion_evaluations" in sql:
            return [promotion_evaluation_row(self._connection.segment_id)]
        if "from segment_definitions" in sql:
            return [segment_definition_row(self._connection.segment_id)]
        if "from promotion_target_segments" in sql:
            if "project_id" in sql:
                return [decision_target_segment_row(self._connection.segment_id)]
            return [target_segment_row(self._connection.segment_id)]
        if "from content_candidates" in sql:
            if self._connection.content_candidate_rows:
                generation_id = None
                if isinstance(self._last_params, dict):
                    generation_id = self._last_params.get("generation_id")
                candidate_rows = [
                    candidate
                    for candidate in self._connection.content_candidate_rows
                    if generation_id is None
                    or candidate["generation_id"] == generation_id
                ]
                if "status in ('approved', 'active')" in sql:
                    return [
                        decision_content_candidate_row(candidate)
                        for candidate in candidate_rows
                        if candidate["status"] in {"approved", "active"}
                    ]
                return candidate_rows
            return [approved_content_candidate_row(self._connection.segment_id)]
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        promotion_run_row: dict[str, object] | None | object = DEFAULT_ROW,
        raise_unique_on_content_candidate_insert: bool = False,
        content_candidate_unique_constraint_name: str | None = None,
        raise_unique_on_promotion_run_insert: bool = False,
        segment_id: str = "seg_luxury",
        active_preparation_row: dict[str, object] | None = None,
        next_attempt_no: int = 1,
    ) -> None:
        self.segment_id = segment_id
        self.promotion_run_row = (
            default_promotion_run_row(segment_id)
            if promotion_run_row is DEFAULT_ROW
            else promotion_run_row
        )
        self.executed: list[tuple[str, Any]] = []
        self.raise_unique_on_content_candidate_insert = (
            raise_unique_on_content_candidate_insert
        )
        self.content_candidate_unique_constraint_name = (
            content_candidate_unique_constraint_name
        )
        self.raise_unique_on_promotion_run_insert = raise_unique_on_promotion_run_insert
        self.segment_id = segment_id
        self.active_preparation_row = active_preparation_row
        self.next_attempt_no = next_attempt_no
        self.content_candidate_rows: list[dict[str, object]] = []
        self.rejected_preparation_rows: list[dict[str, object]] = []
        self.preparation_before_rejection: dict[str, object] | None = None
        self.committed_content_candidate_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0
        self.pending_inserts: list[str] = []
        self.committed_inserts: list[str] = []
        self.rolled_back_inserts: list[str] = []

    def cursor(self, row_factory: Any = None) -> RecordingCursor:
        _ = row_factory
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1
        self.committed_inserts.extend(self.pending_inserts)
        self.pending_inserts.clear()
        self.committed_content_candidate_count = len(self.content_candidate_rows)
        self.preparation_before_rejection = None

    def rollback(self) -> None:
        self.rollback_count += 1
        self.rolled_back_inserts.extend(self.pending_inserts)
        self.pending_inserts.clear()
        del self.content_candidate_rows[self.committed_content_candidate_count :]
        if self.preparation_before_rejection is not None:
            self.active_preparation_row = self.preparation_before_rejection
            self.preparation_before_rejection = None

    def close(self) -> None:
        self.close_count += 1


def _inserted_table_name(sql: str) -> str | None:
    for table_name in (
        "generation_runs",
        "content_candidates",
        "next_loop_preparations",
        "promotion_runs",
        "ad_experiments",
    ):
        if f"insert into {table_name}" in sql:
            return table_name
    return None


def default_promotion_run_row(segment_id: str = "seg_luxury") -> dict[str, object]:
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
        "segment_scope_json": [segment_id],
        "segment_scope_fingerprint": "a" * 64,
    }


def promotion_row(sql: str) -> dict[str, object]:
    base = {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "channel": "onsite_banner",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.300000"),
        "goal_basis": "all_segments",
    }
    if "max_loop_count" in sql:
        return {**base, "min_sample_size": 10, "max_loop_count": 3}
    if "min_sample_size" in sql:
        return {
            **base,
            "min_sample_size": 10,
            "landing_url": "https://demo-stay.example.com/summer",
            "message_brief": "Drive summer hotel booking.",
        }
    return {
        **base,
        "landing_url": "https://demo-stay.example.com/summer",
        "message_brief": "Drive summer hotel booking.",
    }


def ad_experiment_row(segment_id: str = "seg_luxury") -> dict[str, object]:
    return {
        "ad_experiment_id": "adexp_luxury_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "promotion_run_id": "prun_banner_001_loop_1",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "segment_id": segment_id,
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


def promotion_evaluation_row(segment_id: str = "seg_luxury") -> dict[str, object]:
    return {
        "evaluation_id": "eval_luxury_001",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "promotion_run_id": "prun_banner_001_loop_1",
        "ad_experiment_id": "adexp_luxury_001",
        "segment_id": segment_id,
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


def segment_definition_row(segment_id: str = "seg_luxury") -> dict[str, object]:
    ai_suggested = segment_id.startswith("seg_ai_")
    return {
        "segment_id": segment_id,
        "project_id": "hotel-client-a",
        "segment_name": "Stored AI hotel audience",
        "source": "ai_suggested" if ai_suggested else "system_default",
        "query_preview_id": None,
        "natural_language_query": "luxury hotel users",
        "generated_sql": None,
        "rule_json": {
            "segment_id": segment_id,
            "candidate_user_ids": ["user_luxury_001", "user_luxury_002"],
        },
        "profile_json": {"primary_segment": segment_id},
        "sample_size": 1200,
        "total_eligible_user_count": 50000,
        "sample_ratio": Decimal("0.024000"),
        "status": "active",
    }


def target_segment_row(segment_id: str = "seg_luxury") -> dict[str, object]:
    return {
        "analysis_id": "analysis_banner_001_loop_2_1d7b63967183",
        "promotion_id": "promo_banner_001",
        "segment_id": segment_id,
        "segment_name": "Luxury hotel users",
        "content_brief_json": {
            "message_direction": "Use a hotel booking message tailored to this segment.",
            "keywords": ["hotel booking", "seasonal stay", "booking benefit"],
        },
        "data_evidence_json": {
            "source": "system_default",
            "sample_size": 1200,
            "sample_ratio": "0.024000",
        },
        "segment_vector_id": f"segvec_{segment_id}_v1",
        "estimated_size": 1200,
        "priority": "high",
    }


def decision_target_segment_row(
    segment_id: str = "seg_luxury",
) -> dict[str, object]:
    return {
        **target_segment_row(segment_id),
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "rule_json": {
            "segment_id": segment_id,
            "candidate_user_ids": ["user_luxury_001", "user_luxury_002"],
        },
        "profile_json": {"primary_segment": segment_id},
        "status": "active",
    }


def promotion_analysis_row(segment_id: str = "seg_luxury") -> dict[str, object]:
    return {
        "analysis_id": "analysis_banner_001_loop_2_1d7b63967183",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "focus_segment_ids_json": [segment_id],
        "operator_instruction": None,
        "input_snapshot_json": {},
        "profile_summary_json": {},
        "output_json": {},
        "status": "completed",
    }


def generation_run_row(
    params: Any,
    *,
    segment_id: str = "seg_luxury",
) -> dict[str, object]:
    generation_id = (
        params or ("generation_banner_001_loop_2_1d7b63967183",)
    )[0]
    return {
        "generation_id": generation_id,
        "analysis_id": "analysis_banner_001_loop_2_1d7b63967183",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "content_option_count": 3,
        "operator_instruction": None,
        "input_json": {
            "target_segment_ids": [segment_id],
            "next_loop": {
                "loop_count": 2,
                "source_promotion_run_id": "prun_banner_001_loop_1",
                "source_generation_id": "generation_banner_001",
                "focus_segment_ids": [segment_id],
                "attempt_no": 1,
            }
        },
        "output_json": {},
        "generation_report_json": {},
        "status": "completed",
    }


def approved_content_candidate_row(
    segment_id: str = "seg_luxury",
) -> dict[str, object]:
    return {
        "content_id": "content_banner_luxury_hotel_users_001",
        "content_option_id": "banner_luxury_hotel_users_option_001",
        "generation_id": "generation_banner_001_loop_2_1d7b63967183",
        "analysis_id": "analysis_banner_001_loop_2_1d7b63967183",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "segment_id": segment_id,
        "channel": "onsite_banner",
        "status": "approved",
    }


def generation_run_insert_row(params: Any) -> dict[str, object]:
    values = dict(params or {})
    return {
        "generation_id": values["generation_id"],
        "analysis_id": values["analysis_id"],
        "project_id": values["project_id"],
        "campaign_id": values["campaign_id"],
        "promotion_id": values["promotion_id"],
        "content_option_count": values["content_option_count"],
        "operator_instruction": values["operator_instruction"],
        "input_json": values["input_json"],
        "output_json": values["output_json"],
        "generation_report_json": values["generation_report_json"],
        "status": values["status"],
        "created_at": None,
        "updated_at": None,
    }


def content_candidate_insert_row(params: Any) -> dict[str, object]:
    values = dict(params or {})
    return {
        "content_id": values["content_id"],
        "content_option_id": values["content_option_id"],
        "generation_id": values["generation_id"],
        "analysis_id": values["analysis_id"],
        "project_id": values["project_id"],
        "campaign_id": values["campaign_id"],
        "promotion_id": values["promotion_id"],
        "segment_id": values["segment_id"],
        "channel": values["channel"],
        "subject": values["subject"],
        "preheader": values["preheader"],
        "title": values["title"],
        "body": values["body"],
        "cta": values["cta"],
        "message": values["message"],
        "image_prompt": values["image_prompt"],
        "image_url": values["image_url"],
        "landing_url": values["landing_url"],
        "generation_prompt": values["generation_prompt"],
        "reason_summary": values["reason_summary"],
        "data_evidence_json": values["data_evidence_json"],
        "message_strategy": values["message_strategy"],
        "metadata_json": values["metadata_json"],
        "status": values["status"],
        "created_at": None,
        "updated_at": None,
    }


def decision_content_candidate_row(
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        field_name: candidate[field_name]
        for field_name in (
            "content_id",
            "content_option_id",
            "generation_id",
            "analysis_id",
            "project_id",
            "campaign_id",
            "promotion_id",
            "segment_id",
            "channel",
            "status",
        )
    }


def manual_regeneration_connection(
    *,
    raise_unique_on_content_candidate_insert: bool = False,
) -> RecordingConnection:
    generation_id = "generation_banner_001_loop_2_1d7b63967183_attempt_1"
    analysis_id = "analysis_banner_001_loop_2_1d7b63967183"
    connection = RecordingConnection(
        active_preparation_row=manual_preparation_row(
            generation_id=generation_id,
            analysis_id=analysis_id,
        ),
        next_attempt_no=2,
        raise_unique_on_content_candidate_insert=(
            raise_unique_on_content_candidate_insert
        ),
    )
    connection.content_candidate_rows = [
        manual_candidate_row(
            generation_id=generation_id,
            analysis_id=analysis_id,
            option_index=option_index,
            status="rejected",
        )
        for option_index in range(1, 4)
    ]
    connection.committed_content_candidate_count = len(
        connection.content_candidate_rows
    )
    return connection


def manual_preparation_row(
    *,
    generation_id: str,
    analysis_id: str,
) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "next_loop_preparation_id": "nlprep_existing_attempt_1",
        "source_promotion_run_id": "prun_banner_001_loop_1",
        "analysis_id": analysis_id,
        "generation_id": generation_id,
        "attempt_no": 1,
        "failed_segment_ids_json": ["seg_luxury"],
        "failed_ad_experiment_ids_json": ["adexp_luxury_001"],
        "source_evaluation_ids_json": ["eval_luxury_001"],
        "status": "awaiting_content_approval",
        "activated_promotion_run_id": None,
        "created_at": now,
        "updated_at": now,
    }


def manual_candidate_row(
    *,
    generation_id: str,
    analysis_id: str,
    option_index: int,
    status: str,
) -> dict[str, object]:
    generation_slug = generation_id.removeprefix("generation_banner_001_")
    content_slug = f"luxury_{generation_slug}"
    return {
        "content_id": f"content_banner_{content_slug}_{option_index:03d}",
        "content_option_id": (
            f"banner_{content_slug}_option_{option_index:03d}"
        ),
        "generation_id": generation_id,
        "analysis_id": analysis_id,
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "segment_id": "seg_luxury",
        "channel": "onsite_banner",
        "subject": None,
        "preheader": None,
        "title": "Summer hotel stay",
        "body": "Book your summer hotel stay.",
        "cta": "Book now",
        "message": None,
        "image_prompt": "A summer hotel",
        "image_url": None,
        "landing_url": "https://demo-stay.example.com/summer",
        "generation_prompt": "Generate a hotel banner.",
        "reason_summary": "Uses the failed hotel segment.",
        "data_evidence_json": {},
        "message_strategy": "Highlight hotel booking benefits.",
        "metadata_json": {},
        "status": status,
        "created_at": None,
        "updated_at": None,
    }


def next_loop_preparation_insert_row(params: Any) -> dict[str, object]:
    values = tuple(params or ())
    now = datetime.now(UTC)

    def json_value(value: object) -> object:
        return getattr(value, "obj", value)

    return {
        "next_loop_preparation_id": values[0],
        "source_promotion_run_id": values[1],
        "analysis_id": values[2],
        "generation_id": values[3],
        "attempt_no": values[4],
        "failed_segment_ids_json": json_value(values[5]),
        "failed_ad_experiment_ids_json": json_value(values[6]),
        "source_evaluation_ids_json": json_value(values[7]),
        "status": "awaiting_content_approval",
        "activated_promotion_run_id": None,
        "created_at": now,
        "updated_at": now,
    }


class FakeClickHouseResult:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.result_rows = rows or []

    def named_results(self) -> list[dict[str, object]]:
        return self.result_rows


class FakeClickHouseClient:
    def __init__(
        self,
        *,
        user_vector_rows: list[dict[str, object]] | None = None,
        fail_on_recommendation_query: bool = False,
    ) -> None:
        self.close_count = 0
        self.queries: list[str] = []
        self.fail_on_recommendation_query = fail_on_recommendation_query
        self.user_vector_rows = (
            user_behavior_vector_rows()
            if user_vector_rows is None
            else user_vector_rows
        )

    def query(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> FakeClickHouseResult:
        sql = compact_sql(query)
        self.queries.append(sql)
        _ = parameters
        if self.fail_on_recommendation_query and "from raw_events" in sql:
            raise AssertionError(
                "next-loop analysis must not run segment recommendation queries"
            )
        if "from user_behavior_vectors" in sql and "user_id in" in sql:
            return FakeClickHouseResult(self.user_vector_rows)
        return FakeClickHouseResult()

    def close(self) -> None:
        self.close_count += 1


def user_behavior_vector_rows() -> list[dict[str, object]]:
    return [
        {
            "project_id": "hotel-client-a",
            "user_id": "user_luxury_001",
            "vector_dim": 64,
            "vector_values": [1.0, *([0.0] * 63)],
            "vector_version": "v1",
            "source": "batch_profile",
        }
    ]


@pytest.fixture
def next_loop_postgres_schema() -> tuple[str, str]:
    dsn = os.getenv("LOOPAD_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("LOOPAD_TEST_POSTGRES_DSN is required for PostgreSQL locking tests")
    schema_name = f"test_next_loop_{uuid.uuid4().hex}"
    admin = psycopg.connect(dsn, autocommit=True)
    try:
        admin.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
        )
        admin.execute(
            sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
        )
        admin.execute(
            """
            CREATE TABLE next_loop_preparations (
                next_loop_preparation_id text PRIMARY KEY,
                source_promotion_run_id text NOT NULL,
                analysis_id text NOT NULL,
                generation_id text NOT NULL,
                attempt_no integer NOT NULL CHECK (attempt_no >= 1),
                failed_segment_ids_json jsonb NOT NULL,
                failed_ad_experiment_ids_json jsonb NOT NULL,
                source_evaluation_ids_json jsonb NOT NULL,
                status text NOT NULL CHECK (
                    status IN (
                        'awaiting_content_approval',
                        'rejected',
                        'activated'
                    )
                ),
                activated_promotion_run_id text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (source_promotion_run_id, attempt_no)
            )
            """
        )
        admin.execute(
            """
            CREATE UNIQUE INDEX uq_next_loop_preparations_active_source
            ON next_loop_preparations (source_promotion_run_id)
            WHERE status = 'awaiting_content_approval'
            """
        )
        admin.execute(
            """
            CREATE TABLE content_candidates (
                content_id text PRIMARY KEY,
                content_option_id text NOT NULL,
                generation_id text NOT NULL,
                analysis_id text NOT NULL,
                project_id text NOT NULL,
                campaign_id text NOT NULL,
                promotion_id text NOT NULL,
                segment_id text NOT NULL,
                channel text NOT NULL,
                subject text,
                preheader text,
                title text,
                body text,
                cta text,
                message text,
                image_prompt text,
                image_url text,
                landing_url text,
                generation_prompt text,
                reason_summary text,
                data_evidence_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                message_strategy text,
                metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                status text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (generation_id, segment_id, content_option_id)
            )
            """
        )
        admin.execute(
            """
            INSERT INTO next_loop_preparations (
                next_loop_preparation_id,
                source_promotion_run_id,
                analysis_id,
                generation_id,
                attempt_no,
                failed_segment_ids_json,
                failed_ad_experiment_ids_json,
                source_evaluation_ids_json,
                status
            )
            VALUES (
                'prep_attempt_1',
                'run_source_1',
                'analysis_next_1',
                'generation_attempt_1',
                1,
                '["seg_luxury"]'::jsonb,
                '["adexp_luxury_1"]'::jsonb,
                '["eval_luxury_1"]'::jsonb,
                'awaiting_content_approval'
            )
            """
        )
        for option_index in range(1, 3):
            _insert_postgres_candidate(
                admin,
                generation_id="generation_attempt_1",
                option_index=option_index,
                status="rejected",
            )
        yield dsn, schema_name
    finally:
        admin.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(schema_name)
            )
        )
        admin.close()


def test_postgres_dashboard_approve_lock_blocks_regenerate_until_commit(
    next_loop_postgres_schema: tuple[str, str],
) -> None:
    dsn, schema_name = next_loop_postgres_schema
    dashboard_connection = _open_postgres_test_connection(dsn, schema_name)
    decision_connection = _open_postgres_test_connection(dsn, schema_name)
    completed = threading.Event()
    result: dict[str, object] = {}
    failures: list[BaseException] = []
    try:
        dashboard_preparations = NextLoopPreparationRepository(
            PsycopgPostgresExecutor(dashboard_connection)
        )
        dashboard_candidates = ContentCandidateRepository(dashboard_connection)
        assert dashboard_preparations.get_active_by_source_run("run_source_1")
        assert dashboard_candidates.list_by_generation_for_update(
            "generation_attempt_1"
        )
        dashboard_connection.execute(
            """
            UPDATE content_candidates
            SET status = 'approved'
            WHERE content_id = 'content_generation_attempt_1_1'
            """
        )

        def regenerate_reader() -> None:
            try:
                preparations = SerializedNextLoopPreparationRepository(
                    PsycopgPostgresExecutor(decision_connection)
                )
                candidates = ContentCandidateRepository(decision_connection)
                active = preparations.get_active_by_source_run("run_source_1")
                result["active"] = active
                result["candidates"] = candidates.list_by_generation_for_update(
                    "generation_attempt_1"
                )
                decision_connection.commit()
            except BaseException as exc:  # pragma: no cover - diagnostic path
                failures.append(exc)
                decision_connection.rollback()
            finally:
                completed.set()

        worker = threading.Thread(target=regenerate_reader)
        worker.start()
        assert not completed.wait(0.25)
        dashboard_connection.commit()
        assert completed.wait(5)
        worker.join(timeout=1)

        assert failures == []
        assert result["active"] is not None
        statuses = {
            str(candidate["status"])
            for candidate in result["candidates"]  # type: ignore[union-attr]
        }
        assert statuses == {"approved", "rejected"}
    finally:
        dashboard_connection.rollback()
        decision_connection.rollback()
        dashboard_connection.close()
        decision_connection.close()


def test_postgres_regenerate_lock_blocks_dashboard_reject_until_commit(
    next_loop_postgres_schema: tuple[str, str],
) -> None:
    dsn, schema_name = next_loop_postgres_schema
    decision_connection = _open_postgres_test_connection(dsn, schema_name)
    dashboard_connection = _open_postgres_test_connection(dsn, schema_name)
    completed = threading.Event()
    result: dict[str, object] = {}
    failures: list[BaseException] = []
    try:
        preparations = SerializedNextLoopPreparationRepository(
            PsycopgPostgresExecutor(decision_connection)
        )
        candidates = ContentCandidateRepository(decision_connection)
        active = preparations.get_active_by_source_run("run_source_1")
        assert active is not None
        assert candidates.list_by_generation_for_update(active.generation_id)

        def dashboard_reject_reader() -> None:
            try:
                row = dashboard_connection.execute(
                    """
                    SELECT status
                    FROM next_loop_preparations
                    WHERE next_loop_preparation_id = 'prep_attempt_1'
                    FOR UPDATE
                    """
                ).fetchone()
                result["status"] = None if row is None else row[0]
                dashboard_connection.commit()
            except BaseException as exc:  # pragma: no cover - diagnostic path
                failures.append(exc)
                dashboard_connection.rollback()
            finally:
                completed.set()

        worker = threading.Thread(target=dashboard_reject_reader)
        worker.start()
        assert not completed.wait(0.25)
        rejected = preparations.mark_rejected(active.next_loop_preparation_id)
        assert rejected is not None
        assert rejected.status == "rejected"
        decision_connection.commit()
        assert completed.wait(5)
        worker.join(timeout=1)

        assert failures == []
        assert result["status"] == "rejected"
    finally:
        decision_connection.rollback()
        dashboard_connection.rollback()
        decision_connection.close()
        dashboard_connection.close()


def test_postgres_concurrent_regenerate_creates_only_one_attempt_two(
    next_loop_postgres_schema: tuple[str, str],
) -> None:
    dsn, schema_name = next_loop_postgres_schema
    barrier = threading.Barrier(2)
    results: list[str] = []
    failures: list[BaseException] = []

    def regenerate() -> None:
        connection = _open_postgres_test_connection(dsn, schema_name)
        try:
            preparations = SerializedNextLoopPreparationRepository(
                PsycopgPostgresExecutor(connection)
            )
            candidates = ContentCandidateRepository(connection)
            barrier.wait(timeout=5)
            active = preparations.get_active_by_source_run("run_source_1")
            assert active is not None
            rows = candidates.list_by_generation_for_update(active.generation_id)
            if active.attempt_no == 2:
                assert {str(row["status"]) for row in rows} == {"draft"}
                results.append("reused")
                connection.commit()
                return

            assert active.attempt_no == 1
            assert {str(row["status"]) for row in rows} == {"rejected"}
            rejected = preparations.mark_rejected(active.next_loop_preparation_id)
            assert rejected is not None
            assert preparations.get_next_attempt_no("run_source_1") == 2
            time.sleep(0.1)
            preparations.insert(
                NextLoopPreparationWrite(
                    next_loop_preparation_id="prep_attempt_2",
                    source_promotion_run_id="run_source_1",
                    analysis_id="analysis_next_1",
                    generation_id="generation_attempt_2",
                    attempt_no=2,
                    failed_segment_ids_json=("seg_luxury",),
                    failed_ad_experiment_ids_json=("adexp_luxury_1",),
                    source_evaluation_ids_json=("eval_luxury_1",),
                )
            )
            _insert_postgres_candidate(
                connection,
                generation_id="generation_attempt_2",
                option_index=1,
                status="draft",
            )
            results.append("created")
            connection.commit()
        except BaseException as exc:  # pragma: no cover - diagnostic path
            failures.append(exc)
            connection.rollback()
        finally:
            connection.close()

    workers = [threading.Thread(target=regenerate) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=8)

    assert all(not worker.is_alive() for worker in workers)
    assert failures == []
    assert sorted(results) == ["created", "reused"]
    observer = _open_postgres_test_connection(dsn, schema_name)
    try:
        rows = observer.execute(
            """
            SELECT attempt_no, status
            FROM next_loop_preparations
            WHERE source_promotion_run_id = 'run_source_1'
            ORDER BY attempt_no
            """
        ).fetchall()
        assert rows == [(1, "rejected"), (2, "awaiting_content_approval")]
    finally:
        observer.rollback()
        observer.close()


def test_postgres_regeneration_failure_rolls_back_rejection_and_attempt_two(
    next_loop_postgres_schema: tuple[str, str],
) -> None:
    dsn, schema_name = next_loop_postgres_schema
    connection = _open_postgres_test_connection(dsn, schema_name)
    try:
        preparations = SerializedNextLoopPreparationRepository(
            PsycopgPostgresExecutor(connection)
        )
        candidates = ContentCandidateRepository(connection)
        active = preparations.get_active_by_source_run("run_source_1")
        assert active is not None
        assert candidates.list_by_generation_for_update(active.generation_id)
        assert preparations.mark_rejected(active.next_loop_preparation_id)
        preparations.insert(
            NextLoopPreparationWrite(
                next_loop_preparation_id="prep_attempt_2",
                source_promotion_run_id="run_source_1",
                analysis_id="analysis_next_1",
                generation_id="generation_attempt_2",
                attempt_no=2,
                failed_segment_ids_json=("seg_luxury",),
                failed_ad_experiment_ids_json=("adexp_luxury_1",),
                source_evaluation_ids_json=("eval_luxury_1",),
            )
        )
        _insert_postgres_candidate(
            connection,
            generation_id="generation_attempt_2",
            option_index=1,
            status="draft",
        )
        connection.rollback()
    finally:
        connection.close()

    observer = _open_postgres_test_connection(dsn, schema_name)
    try:
        preparations = observer.execute(
            """
            SELECT attempt_no, status
            FROM next_loop_preparations
            WHERE source_promotion_run_id = 'run_source_1'
            ORDER BY attempt_no
            """
        ).fetchall()
        attempt_two_candidates = observer.execute(
            """
            SELECT count(*)
            FROM content_candidates
            WHERE generation_id = 'generation_attempt_2'
            """
        ).fetchone()
        assert preparations == [(1, "awaiting_content_approval")]
        assert attempt_two_candidates == (0,)
    finally:
        observer.rollback()
        observer.close()


def _open_postgres_test_connection(
    dsn: str,
    schema_name: str,
) -> psycopg.Connection[Any]:
    connection = psycopg.connect(dsn, autocommit=False)
    connection.execute(
        sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
    )
    connection.commit()
    return connection


def _insert_postgres_candidate(
    connection: Any,
    *,
    generation_id: str,
    option_index: int,
    status: str,
) -> None:
    connection.execute(
        """
        INSERT INTO content_candidates (
            content_id,
            content_option_id,
            generation_id,
            analysis_id,
            project_id,
            campaign_id,
            promotion_id,
            segment_id,
            channel,
            title,
            body,
            cta,
            landing_url,
            generation_prompt,
            reason_summary,
            data_evidence_json,
            message_strategy,
            metadata_json,
            status
        )
        VALUES (
            %s, %s, %s,
            'analysis_next_1',
            'hotel-client-a',
            'campaign-summer',
            'promotion-banner',
            'seg_luxury',
            'onsite_banner',
            'Summer hotel stay',
            'Book a summer hotel stay.',
            'Book now',
            'https://hotel.example.com/summer',
            'Generate a hotel banner.',
            'Uses the failed hotel segment.',
            '{}'::jsonb,
            'Highlight hotel booking benefits.',
            '{}'::jsonb,
            %s
        )
        """,
        (
            f"content_{generation_id}_{option_index}",
            f"option_{generation_id}_{option_index}",
            generation_id,
            status,
        ),
    )
