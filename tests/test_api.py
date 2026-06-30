from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.jobs.decision_job import DecisionRunHandle, DecisionRunRequest


def settings() -> Settings:
    return Settings(
        env="dev",
        service_id="decision-api",
        port=8080,
        internal_api_key="internal-secret",
        aurora_host="aurora.local",
        aurora_port=5432,
        aurora_database="loopad",
        aurora_username="app",
        aurora_password="password",
        clickhouse_url="http://clickhouse.local:8123",
        clickhouse_database="loopad",
        clickhouse_username="app",
        clickhouse_password="password",
        data_storage_bucket="example-data-storage-bucket",
        genai_assets_base_prefix="genai/",
        openai_api_key="test-openai-key",
        legacy_admin_token="legacy-secret",
    )


class FakeJobService:
    def __init__(self) -> None:
        self.started_requests: list[DecisionRunRequest] = []
        self.executed_run_ids: list[int] = []

    def start_run(self, request: DecisionRunRequest) -> DecisionRunHandle:
        self.started_requests.append(request)
        return DecisionRunHandle(
            run_id=123,
            project_id=1,
            project_key=request.project_key,
            analysis_date=request.analysis_date,
            status="running",
        )

    def execute_run(self, run_id: int) -> object:
        self.executed_run_ids.append(run_id)
        return None


def test_health_returns_ok_for_decision_api() -> None:
    client = TestClient(create_app(settings=settings()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "decision-api",
        "env": "dev",
    }


def test_internal_daily_decision_rejects_invalid_internal_key() -> None:
    fake_service = FakeJobService()
    client = TestClient(
        create_app(settings=settings(), job_service_factory=lambda: fake_service)
    )

    response = client.post(
        "/internal/jobs/daily-decision/run",
        headers={"X-Loop-Ad-Internal-Key": "wrong"},
        json={
            "project_key": "demo-shop",
            "analysis_date": "2021-01-04",
            "mode": "demo",
            "force": True,
        },
    )

    assert response.status_code == 401
    assert fake_service.started_requests == []


def test_internal_daily_decision_uses_loopad_internal_key_and_manual_api_run_type() -> None:
    fake_service = FakeJobService()
    client = TestClient(
        create_app(settings=settings(), job_service_factory=lambda: fake_service)
    )

    response = client.post(
        "/internal/jobs/daily-decision/run",
        headers={"X-Loop-Ad-Internal-Key": "internal-secret"},
        json={
            "project_key": "demo-shop",
            "analysis_date": "2021-01-04",
            "mode": "demo",
            "force": True,
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "run_id": 123,
        "project_key": "demo-shop",
        "analysis_date": "2021-01-04",
        "status": "running",
        "message": "daily decision job started",
    }
    assert fake_service.started_requests == [
        DecisionRunRequest(
            project_key="demo-shop",
            analysis_date=date(2021, 1, 4),
            mode="demo",
            force=True,
            run_type="manual_api",
            trigger_source="api",
            requested_by="testclient",
        )
    ]
    assert fake_service.executed_run_ids == [123]


def test_internal_daily_decision_accepts_legacy_admin_token_when_configured() -> None:
    fake_service = FakeJobService()
    client = TestClient(
        create_app(settings=settings(), job_service_factory=lambda: fake_service)
    )

    response = client.post(
        "/internal/jobs/daily-decision/run",
        headers={"X-Admin-Token": "legacy-secret"},
        json={"project_key": "demo-shop", "analysis_date": "2021-01-04"},
    )

    assert response.status_code == 202
    assert fake_service.started_requests[0].run_type == "manual_api"
