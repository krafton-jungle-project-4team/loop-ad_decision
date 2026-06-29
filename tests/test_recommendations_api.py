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
            recommendation_result(1, "pending_actions"),
            recommendation_result(2, "dismissed"),
        ]
        self.actions = [recommendation_action()]
        self.active_mappings = [
            SimpleNamespace(
                id=10,
                project_id="loopad-demo-shop",
                segment_json={"channel": "kakao"},
                segment_hash="hash",
                action_id="recommend_alternative_product",
                action_type="PRODUCT",
                execution_hint_json={"slot": "product_detail", "serving_weight": 0.7},
                experiment_id=20,
                recommendation_result_id=1,
                recommendation_action_id=100,
                bandit_policy_id=30,
                bandit_arm_id=40,
                bandit_decision_id=50,
                campaign_id=60,
                creative_id=70,
                coupon_id=80,
                source="manual_approval",
                status="active",
                creative=SimpleNamespace(
                    image_url="https://cdn.example/ad.png",
                    title="대체 상품",
                    message="지금 확인하세요",
                    landing_url="https://shop.example/products/1",
                ),
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

    def list_recommendation_actions(
        self,
        *,
        recommendation_result_id: int,
    ) -> list[SimpleNamespace]:
        return [
            action
            for action in self.actions
            if action.recommendation_result_id == recommendation_result_id
        ]

    def list_active_segment_ad_mappings_with_creatives(self, project_id: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(mapping=mapping, creative=getattr(mapping, "creative", None))
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


def recommendation_action() -> SimpleNamespace:
    return SimpleNamespace(
        id=100,
        project_id="loopad-demo-shop",
        recommendation_result_id=1,
        action_id="recommend_alternative_product",
        action_type="PRODUCT",
        title="대체 상품 추천",
        description="품절 상품 대신 대체 상품을 추천합니다.",
        target_step=None,
        priority_score=0.8,
        expected_impact="전환율 개선",
        rationale="테스트 추천",
        triggered_by_json=["cause-1"],
        execution_hint_json={"slot": "product_detail"},
        experiment_json={"primary_metric": "view_to_purchase_rate"},
        policy_status=None,
        policy_reasons_json=[],
        policy_decision_json={},
        selected_by_strategy="policy",
        bandit_policy_id=None,
        bandit_arm_id=None,
        sampled_value=None,
        status="pending_review",
        auto_executed_at=None,
        reviewed_by=None,
        reviewed_at=None,
        review_reason=None,
        created_at=WINDOW_START,
        updated_at=WINDOW_START,
    )


def test_get_recommendations_filters_by_status() -> None:
    fake_repository = FakeRecommendationApiRepository()
    app.dependency_overrides[get_recommendation_repository] = lambda: fake_repository
    try:
        response = TestClient(app).get(
            "/recommendations",
            params={"project_id": "loopad-demo-shop", "status": "pending_actions"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == 1
    assert body[0]["status"] == "pending_actions"
    assert body[0]["actions"][0]["id"] == 100


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
    assert body[0]["recommendation_action_id"] == 100
    assert body[0]["content_url"] == "https://cdn.example/ad.png"
    assert body[0]["serving_weight"] == 0.7
