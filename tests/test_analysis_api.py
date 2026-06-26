from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.analysis.router import get_analysis_repository
from app.main import app
from app.persistence.job_statuses import ANALYSIS_JOB_STATUS_QUEUED


KST = ZoneInfo("Asia/Seoul")


class FakeAnalysisJobRepository:
    def __init__(self) -> None:
        self.jobs: dict[int, SimpleNamespace] = {}
        self.committed = False
        self.rolled_back = False
        self.created_request_json: dict[str, object] | None = None

    def create_analysis_job(
        self,
        project_id: str,
        request_json: dict[str, object],
        status: str = ANALYSIS_JOB_STATUS_QUEUED,
    ) -> SimpleNamespace:
        now = datetime(2026, 6, 26, 12, 0, tzinfo=KST)
        job = SimpleNamespace(
            id=1,
            project_id=project_id,
            status=status,
            request_json=request_json,
            recommendation_result_id=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            started_at=None,
            finished_at=None,
        )
        self.jobs[job.id] = job
        self.created_request_json = request_json
        return job

    def get_analysis_job(self, job_id: int) -> SimpleNamespace | None:
        return self.jobs.get(job_id)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_post_analysis_funnel_recommend_rejects_invalid_top_n() -> None:
    response = TestClient(app).post(
        "/analysis/funnel/recommend",
        json={
            "project_id": "loopad-demo-shop",
            "window_start": "2026-06-24T17:00:00+09:00",
            "window_end": "2026-06-24T18:00:00+09:00",
            "top_n": 0,
        },
    )

    assert response.status_code == 400


def test_post_analysis_funnel_recommend_jobs_creates_queued_job() -> None:
    fake_repository = FakeAnalysisJobRepository()
    app.dependency_overrides[get_analysis_repository] = lambda: fake_repository

    try:
        response = TestClient(app).post(
            "/analysis/funnel/recommend/jobs",
            json={
                "project_id": "loopad-demo-shop",
                "window_start": "2026-06-24T17:00:00+09:00",
                "window_end": "2026-06-24T18:00:00+09:00",
                "top_n": 5,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "job_id": 1,
        "status": "queued",
        "recommendation_result_id": None,
        "polling_url": "/analysis/jobs/1",
    }
    assert fake_repository.committed is True
    assert fake_repository.rolled_back is False
    assert fake_repository.created_request_json is not None
    assert fake_repository.created_request_json["project_id"] == "loopad-demo-shop"


def test_get_analysis_job_returns_status() -> None:
    fake_repository = FakeAnalysisJobRepository()
    job = fake_repository.create_analysis_job(
        project_id="loopad-demo-shop",
        request_json={},
        status="running",
    )
    job.started_at = datetime(2026, 6, 26, 12, 1, tzinfo=KST)
    app.dependency_overrides[get_analysis_repository] = lambda: fake_repository

    try:
        response = TestClient(app).get("/analysis/jobs/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == 1
    assert payload["status"] == "running"
    assert payload["recommendation_result_id"] is None
    assert payload["error_message"] is None
    assert payload["created_at"] == "2026-06-26T12:00:00+09:00"
    assert payload["started_at"] == "2026-06-26T12:01:00+09:00"
    assert payload["finished_at"] is None


def test_get_analysis_job_returns_404_for_missing_job() -> None:
    fake_repository = FakeAnalysisJobRepository()
    app.dependency_overrides[get_analysis_repository] = lambda: fake_repository

    try:
        response = TestClient(app).get("/analysis/jobs/404")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
