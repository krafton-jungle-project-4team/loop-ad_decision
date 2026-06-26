from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.automation.router import get_automation_policy_repository
from app.main import app


def policy_object(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": 1,
        "project_id": "loopad-demo-shop",
        "enabled": True,
        "auto_execute_enabled": True,
        "allowed_action_ids": [],
        "allowed_action_types": ["PRODUCT"],
        "blocked_action_ids": [],
        "max_experiment_traffic_ratio": 0.2,
        "min_priority_score": 0.5,
        "max_discount_rate": 0.1,
        "max_daily_coupon_budget": 1000000.0,
        "max_message_per_user_per_day": 1,
        "stop_loss_relative_drop": 0.05,
        "created_at": datetime(2026, 6, 26, 10, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        "updated_at": datetime(2026, 6, 26, 10, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakePolicyRepository:
    def __init__(self, existing_policy: SimpleNamespace | None = None) -> None:
        self.existing_policy = existing_policy
        self.upsert_values: dict[str, object] | None = None
        self.committed = False
        self.rolled_back = False

    def get_automation_policy(self, project_id: str) -> SimpleNamespace | None:
        if self.existing_policy is None:
            return None
        if self.existing_policy.project_id != project_id:
            return None
        return self.existing_policy

    def upsert_automation_policy(
        self,
        project_id: str,
        values: dict[str, object],
    ) -> SimpleNamespace:
        self.upsert_values = values
        self.existing_policy = policy_object(project_id=project_id, **values)
        return self.existing_policy

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_get_automation_policy_returns_existing_policy() -> None:
    fake_repository = FakePolicyRepository(policy_object())
    app.dependency_overrides[get_automation_policy_repository] = lambda: fake_repository
    try:
        response = TestClient(app).get("/automation-policies/loopad-demo-shop")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "loopad-demo-shop"
    assert body["enabled"] is True
    assert body["allowed_action_types"] == ["PRODUCT"]


def test_get_automation_policy_returns_404_when_missing() -> None:
    fake_repository = FakePolicyRepository()
    app.dependency_overrides[get_automation_policy_repository] = lambda: fake_repository
    try:
        response = TestClient(app).get("/automation-policies/loopad-demo-shop")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_put_automation_policy_upserts_and_commits() -> None:
    fake_repository = FakePolicyRepository()
    app.dependency_overrides[get_automation_policy_repository] = lambda: fake_repository
    try:
        response = TestClient(app).put(
            "/automation-policies/loopad-demo-shop",
            json={
                "enabled": True,
                "auto_execute_enabled": True,
                "allowed_action_ids": ["recommend_alternative_product"],
                "allowed_action_types": ["PRODUCT", "MESSAGE"],
                "blocked_action_ids": ["limited_time_coupon"],
                "max_experiment_traffic_ratio": 0.2,
                "min_priority_score": 0.6,
                "max_discount_rate": 0.1,
                "max_daily_coupon_budget": 1000000,
                "max_message_per_user_per_day": 1,
                "stop_loss_relative_drop": 0.05,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "loopad-demo-shop"
    assert body["allowed_action_ids"] == ["recommend_alternative_product"]
    assert body["blocked_action_ids"] == ["limited_time_coupon"]
    assert fake_repository.upsert_values is not None
    assert fake_repository.upsert_values["min_priority_score"] == 0.6
    assert fake_repository.committed is True
    assert fake_repository.rolled_back is False


def test_put_automation_policy_rejects_invalid_ratio() -> None:
    response = TestClient(app).put(
        "/automation-policies/loopad-demo-shop",
        json={"max_experiment_traffic_ratio": 1.2},
    )

    assert response.status_code == 400
