from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from psycopg import errors

from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.repositories import PromotionRunWrite
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
    PromotionRunService,
    RunConflictError,
    RunValidationError,
    normalize_explicit_segment_ids,
)
from app.main import create_app
from tests.test_decision_run_service import (
    activation_candidates,
    activation_request,
    content_candidate_record,
    make_preparation_activation_service,
    make_service,
)


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
    assert payload["segment_ids"] == ["seg_family_trip"]
    assert payload["ad_experiments"][0]["ad_experiment_id"] == "adexp_fake"
    assert payload["ad_experiments"][0]["is_fallback"] is False
    assert service.calls[0][0] == "promo_banner_001"
    assert service.calls[0][1].loop_count == 2
    assert service.calls[0][1].next_loop_preparation_id is None


def test_run_api_accepts_and_forwards_segment_ids() -> None:
    service = FakeRunService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={
            "analysis_id": "analysis_banner_001",
            "generation_id": "generation_banner_001",
            "segment_ids": ["seg_mobile_user"],
        },
    )

    assert response.status_code == 200
    assert service.calls[0][1].segment_ids == ["seg_mobile_user"]


def test_run_api_accepts_nullable_next_loop_preparation_id() -> None:
    service = FakeRunService()
    client = make_client(service)

    legacy_response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={"next_loop_preparation_id": None},
    )
    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={
            "analysis_id": "analysis_banner_002",
            "generation_id": "generation_banner_002_attempt_1",
            "loop_count": 2,
            "next_loop_preparation_id": "nlprep_banner_001_loop_2_attempt_1",
        },
    )

    assert legacy_response.status_code == 200
    assert service.calls[0][1].next_loop_preparation_id is None
    assert response.status_code == 200
    assert (
        service.calls[1][1].next_loop_preparation_id
        == "nlprep_banner_001_loop_2_attempt_1"
    )


def test_run_api_rejects_invalid_body() -> None:
    client = make_client(FakeRunService())

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={"unexpected": "field"},
    )

    assert response.status_code == 400
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_run_api_rejects_blank_next_loop_preparation_id() -> None:
    client = make_client(FakeRunService())

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={"next_loop_preparation_id": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"][0]["type"] == "string_too_short"


def test_run_api_normalizes_duplicate_segment_ids_and_rejects_empty_or_blank() -> None:
    service = NormalizingFakeRunService()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={"segment_ids": [" seg_b ", "seg_a", "seg_a"]},
    )

    assert response.status_code == 200
    assert service.calls[0][1].segment_ids == ["seg_a", "seg_b"]

    for segment_ids in ([], ["   "], ["seg_family_trip", "   "]):
        response = client.post(
            "/decision/v1/promotions/promo_banner_001/runs",
            json={"segment_ids": segment_ids},
        )

        assert response.status_code == 422


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


def test_run_api_accepts_explicit_scope_without_runtime_gate(
    monkeypatch,
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
        "/decision/v1/promotions/promo_banner_001/runs",
        json={"segment_ids": ["seg_family_trip"]},
    )

    assert response.status_code == 200
    connection = connections[0]
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into promotion_runs" in query for query in executed_sql)
    assert any("insert into ad_experiments" in query for query in executed_sql)
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1


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


def test_run_api_rejects_preparation_activation_when_manual_switch_is_off(
    monkeypatch,
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
        "/decision/v1/promotions/promo_banner_001/runs",
        json={
            "analysis_id": "analysis_banner_002",
            "generation_id": "generation_banner_002",
            "loop_count": 2,
            "next_loop_preparation_id": "prep_banner_002",
        },
    )

    assert response.status_code == 409
    connection = connections[0]
    assert connection.executed == []
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1


def test_run_api_activates_dashboard_approval_fixture_with_lineage_and_retry() -> None:
    candidates = activation_candidates() + [
        content_candidate_record(
            analysis_id="analysis_banner_loop_2",
            generation_id="generation_banner_loop_2",
            segment_id="seg_family_trip",
            content_id="content_family_draft",
            status="draft",
        ),
        content_candidate_record(
            analysis_id="analysis_banner_loop_2",
            generation_id="generation_banner_loop_2",
            segment_id="seg_mobile_user",
            content_id="content_mobile_rejected",
            status="rejected",
        ),
    ]
    service, repos = make_preparation_activation_service(candidates=candidates)
    client = make_client(service)
    payload = activation_request().model_dump(mode="json", exclude_none=True)

    first = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json=payload,
    )
    retry = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json=payload,
    )

    assert first.status_code == 200, first.text
    assert retry.status_code == 200, retry.text
    assert retry.json() == first.json()
    assert first.json()["segment_ids"] == ["seg_family_trip", "seg_mobile_user"]
    assert [
        experiment["is_fallback"] for experiment in first.json()["ad_experiments"]
    ] == [False, False, True]
    lineage = {
        experiment.segment_id: (
            experiment.parent_ad_experiment_id,
            experiment.source_evaluation_id,
        )
        for experiment in repos.ad_experiments.inserted_batches[0]
    }
    assert lineage == {
        "seg_family_trip": ("adexp_source_family", "eval_source_family"),
        "seg_mobile_user": ("adexp_source_mobile", "eval_source_mobile"),
        FALLBACK_SEGMENT_ID: (None, None),
    }
    assert len(repos.runs.inserted) == 1
    assert len(repos.ad_experiments.inserted_batches) == 1
    assert repos.preparations.activated_calls == [
        ("prep_loop_2", first.json()["promotion_run_id"])
    ]


@pytest.mark.parametrize(
    "case",
    ["zero", "partial", "duplicate", "cross-generation"],
)
def test_run_api_rejects_incomplete_or_cross_generation_dashboard_approval(
    case: str,
) -> None:
    candidates = activation_candidates()
    if case == "zero":
        candidates = []
    elif case == "partial":
        candidates = candidates[:1]
    elif case == "duplicate":
        candidates.append(
            content_candidate_record(
                analysis_id="analysis_banner_loop_2",
                generation_id="generation_banner_loop_2",
                segment_id="seg_family_trip",
                content_id="content_family_duplicate",
            )
        )
    else:
        candidates = [
            replace(candidate, generation_id="generation_other")
            for candidate in candidates
        ]
    service, repos = make_preparation_activation_service(candidates=candidates)
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json=activation_request().model_dump(mode="json", exclude_none=True),
    )

    assert response.status_code == 422
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []
    assert repos.preparations.activated_calls == []


def test_run_api_keeps_legacy_single_candidate_and_nullable_lineage() -> None:
    service, repos = make_service()
    client = make_client(service)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={},
    )

    assert response.status_code == 200, response.text
    assert response.json()["segment_ids"] == ["seg_family_trip"]
    experiments = repos.ad_experiments.inserted_batches[0]
    assert [experiment.segment_id for experiment in experiments] == [
        "seg_family_trip",
        FALLBACK_SEGMENT_ID,
    ]
    assert all(
        experiment.parent_ad_experiment_id is None
        and experiment.source_evaluation_id is None
        for experiment in experiments
    )
    assert repos.preparations.activated_calls == []


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
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into promotion_runs" in query for query in executed_sql)
    assert not any("insert into ad_experiments" in query for query in executed_sql)


@pytest.mark.parametrize("fail_on_mark_activation", [False, True])
def test_run_api_keeps_run_insert_and_preparation_activation_in_one_transaction(
    monkeypatch,
    fail_on_mark_activation: bool,
) -> None:
    connections: list[RecordingConnection] = []

    def fake_create_postgres_connection(_settings) -> RecordingConnection:
        connection = RecordingConnection(
            raise_on_mark_activation=fail_on_mark_activation,
        )
        connections.append(connection)
        return connection

    def write_run_then_activate(
        self: PromotionRunService,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        run = PromotionRunWrite(
            promotion_run_id="prun_transaction_loop_2",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id=promotion_id,
            analysis_id="analysis_banner_002",
            generation_id="generation_banner_002",
            loop_count=2,
            status=PromotionRunStatus.PLANNED.value,
            goal_snapshot_json={},
            segment_scope_json=("seg_family_trip",),
            segment_scope_fingerprint="a" * 64,
        )
        assert self._promotion_run_repository.insert_if_absent(run)
        self._next_loop_preparation_repository.mark_activated(
            next_loop_preparation_id="prep_banner_002",
            activated_promotion_run_id=run.promotion_run_id,
        )
        return FakeRunService().create_run(
            promotion_id=promotion_id,
            request=request,
        )

    monkeypatch.setattr(
        "app.decision.router.create_postgres_connection",
        fake_create_postgres_connection,
    )
    monkeypatch.setattr(PromotionRunService, "create_run", write_run_then_activate)
    app = create_app(settings=load_settings(valid_env()))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/runs",
        json={},
    )

    connection = connections[0]
    executed_sql = [compact_sql(query) for query, _params in connection.executed]
    assert any("insert into promotion_runs" in query for query in executed_sql)
    assert any(
        "update next_loop_preparations" in query
        and "set status = 'activated'" in query
        for query in executed_sql
    )
    if fail_on_mark_activation:
        assert response.status_code == 500
        assert connection.commit_count == 0
        assert connection.rollback_count == 1
    else:
        assert response.status_code == 200
        assert connection.commit_count == 1
        assert connection.rollback_count == 0
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
            segment_ids=list(request.segment_ids or ["seg_family_trip"]),
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
                    is_fallback=False,
                )
            ],
        )


class NormalizingFakeRunService(FakeRunService):
    def create_run(
        self,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        normalized = normalize_explicit_segment_ids(request.segment_ids)
        normalized_request = request.model_copy(
            update={
                "segment_ids": list(normalized) if normalized is not None else None
            }
        )
        return super().create_run(
            promotion_id=promotion_id,
            request=normalized_request,
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
            self._connection.raise_on_mark_activation
            and "update next_loop_preparations" in compact_sql(query)
            and "set status = 'activated'" in compact_sql(query)
        ):
            raise RuntimeError("mark_activated failed after run insert")
        if (
            self._connection.raise_unique_on_insert
            and "insert into promotion_runs" in compact_sql(query)
        ):
            raise errors.UniqueViolation("duplicate run")

    def fetchone(self) -> dict[str, object] | None:
        sql = compact_sql(self._last_query)
        if "insert into promotion_runs" in sql:
            return {"promotion_run_id": "prun_inserted"}
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
        raise_on_mark_activation: bool = False,
    ) -> None:
        self.analysis_row = (
            analysis_row_ok() if analysis_row is DEFAULT_ROW else analysis_row
        )
        self.run_exists = run_exists
        self.raise_unique_on_insert = raise_unique_on_insert
        self.raise_on_mark_activation = raise_on_mark_activation
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
        "input_json": {
            "analysis_id": "analysis_banner_001",
            "target_segment_ids": ["seg_family_trip"],
        },
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
