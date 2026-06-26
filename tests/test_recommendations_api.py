from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.main import app
from app.recommendations.router import get_recommendation_repository

KST = ZoneInfo("Asia/Seoul")
WINDOW_START = datetime(2026, 6, 24, 17, 0, tzinfo=KST)
WINDOW_END = datetime(2026, 6, 24, 18, 0, tzinfo=KST)


class FakeRecommendationApiRepository:
    def __init__(self) -> None:
        self.results = [
            recommendation_result(1, "pending_review"),
            recommendation_result(2, "rejected"),
        ]
        self.active_mappings = [
            SimpleNamespace(
                id=10,
                project_id="loopad-demo-shop",
                segment_json={"channel": "kakao"},
                segment_hash="hash",
                action_id="recommend_alternative_product",
                action_type="PRODUCT",
                execution_hint_json={"slot": "product_detail"},
                experiment_id=20,
                recommendation_result_id=1,
                source="manual_approval",
                status="active",
            )
        ]

    def list_recommendation_results(
        self,
        *,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SimpleNamespace]:
        values = self.results
        if project_id is not None:
            values = [result for result in values if result.project_id == project_id]
        if status is not None:
            values = [result for result in values if result.status == status]
        return values[:limit]

    def get_recommendation_result(self, recommendation_result_id: int) -> SimpleNamespace | None:
        for result in self.results:
            if result.id == recommendation_result_id:
                return result
        return None

    def list_active_segment_ad_mappings(self, project_id: str) -> list[SimpleNamespace]:
        return [
            mapping
            for mapping in self.active_mappings
            if mapping.project_id == project_id and mapping.status == "active"
        ]


def recommendation_result(result_id: int, status: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=result_id,
        project_id="loopad-demo-shop",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        baseline_start=None,
        baseline_end=None,
        segment_json={"channel": "kakao"},
        segment_hash="hash",
        status=status,
        anomaly_json={},
        root_causes_json={},
        recommendations_json={"recommendations": []},
        policy_decision_json={"actions": []},
        created_at=WINDOW_START,
        updated_at=WINDOW_START,
    )


def test_get_recommendations_filters_by_status() -> None:
    fake_repository = FakeRecommendationApiRepository()
    app.dependency_overrides[get_recommendation_repository] = lambda: fake_repository
    try:
        response = TestClient(app).get(
            "/recommendations",
            params={"project_id": "loopad-demo-shop", "status": "pending_review"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == 1
    assert body[0]["status"] == "pending_review"


def test_get_recommendation_detail_returns_404_when_missing() -> None:
    fake_repository = FakeRecommendationApiRepository()
    app.dependency_overrides[get_recommendation_repository] = lambda: fake_repository
    try:
        response = TestClient(app).get("/recommendations/999")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_get_active_ad_mappings_returns_active_project_mappings() -> None:
    fake_repository = FakeRecommendationApiRepository()
    app.dependency_overrides[get_recommendation_repository] = lambda: fake_repository
    try:
        response = TestClient(app).get(
            "/ad-mappings/active",
            params={"project_id": "loopad-demo-shop"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["mapping_id"] == 10
    assert body[0]["status"] == "active"
