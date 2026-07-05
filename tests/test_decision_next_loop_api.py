from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from psycopg import errors

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
    assert body["next_analysis_id"] == "analysis_banner_001_loop_2"
    assert body["next_generation_id"] == "generation_banner_001_loop_2"
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

    def fetchone(self) -> dict[str, object] | None:
        sql = compact_sql(self._last_query)
        if "insert into generation_runs" in sql:
            return generation_run_insert_row(self._last_params)
        if "insert into content_candidates" in sql:
            return content_candidate_insert_row(self._last_params)
        if "from promotion_analyses" in sql:
            return promotion_analysis_row()
        if "from generation_runs" in sql:
            return generation_run_row(self._last_params)
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
            return [ad_experiment_row()]
        if "from promotion_evaluations" in sql:
            return [promotion_evaluation_row()]
        if "from segment_definitions" in sql:
            return [segment_definition_row()]
        if "from promotion_target_segments" in sql:
            if "project_id" in sql:
                return [decision_target_segment_row()]
            return [target_segment_row()]
        if "from content_candidates" in sql:
            return [approved_content_candidate_row()]
        return []


class RecordingConnection:
    def __init__(
        self,
        *,
        promotion_run_row: dict[str, object] | None | object = DEFAULT_ROW,
        raise_unique_on_content_candidate_insert: bool = False,
        content_candidate_unique_constraint_name: str | None = None,
        raise_unique_on_promotion_run_insert: bool = False,
    ) -> None:
        self.promotion_run_row = (
            default_promotion_run_row()
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

    def rollback(self) -> None:
        self.rollback_count += 1
        self.rolled_back_inserts.extend(self.pending_inserts)
        self.pending_inserts.clear()

    def close(self) -> None:
        self.close_count += 1


def _inserted_table_name(sql: str) -> str | None:
    for table_name in (
        "generation_runs",
        "content_candidates",
        "promotion_runs",
        "ad_experiments",
    ):
        if f"insert into {table_name}" in sql:
            return table_name
    return None


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


def segment_definition_row() -> dict[str, object]:
    return {
        "segment_id": "seg_luxury",
        "project_id": "hotel-client-a",
        "segment_name": "Luxury hotel users",
        "source": "system_default",
        "query_preview_id": None,
        "natural_language_query": "luxury hotel users",
        "generated_sql": None,
        "rule_json": {"segment_id": "seg_luxury"},
        "profile_json": {"primary_segment": "seg_luxury"},
        "sample_size": 1200,
        "total_eligible_user_count": 50000,
        "sample_ratio": Decimal("0.024000"),
        "status": "active",
    }


def target_segment_row() -> dict[str, object]:
    return {
        "analysis_id": "analysis_banner_001_loop_2",
        "promotion_id": "promo_banner_001",
        "segment_id": "seg_luxury",
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
        "segment_vector_id": "segvec_seg_luxury_v1",
        "estimated_size": 1200,
        "priority": "high",
    }


def decision_target_segment_row() -> dict[str, object]:
    return {
        **target_segment_row(),
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "rule_json": {"segment_id": "seg_luxury"},
        "profile_json": {"primary_segment": "seg_luxury"},
        "status": "active",
    }


def promotion_analysis_row() -> dict[str, object]:
    return {
        "analysis_id": "analysis_banner_001_loop_2",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "focus_segment_ids_json": ["seg_luxury"],
        "operator_instruction": None,
        "input_snapshot_json": {},
        "profile_summary_json": {},
        "output_json": {},
        "status": "completed",
    }


def generation_run_row(params: Any) -> dict[str, object]:
    generation_id = (params or ("generation_banner_001_loop_2",))[0]
    return {
        "generation_id": generation_id,
        "analysis_id": "analysis_banner_001_loop_2",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "content_option_count": 1,
        "operator_instruction": None,
        "input_json": {},
        "output_json": {},
        "generation_report_json": {},
        "status": "completed",
    }


def approved_content_candidate_row() -> dict[str, object]:
    return {
        "content_id": "content_banner_luxury_hotel_users_001",
        "content_option_id": "banner_luxury_hotel_users_option_001",
        "generation_id": "generation_banner_001_loop_2",
        "analysis_id": "analysis_banner_001_loop_2",
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "segment_id": "seg_luxury",
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


class FakeClickHouseResult:
    result_rows: list[dict[str, object]] = []

    def named_results(self) -> list[dict[str, object]]:
        return []


class FakeClickHouseClient:
    close_count = 0

    def query(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> FakeClickHouseResult:
        _ = query
        _ = parameters
        return FakeClickHouseResult()

    def close(self) -> None:
        self.close_count += 1
