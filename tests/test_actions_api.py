import pytest
from fastapi.testclient import TestClient

from app.main import app


def action_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "project_id": "loopad-demo-shop",
        "window_start": "2026-06-24T17:00:00+09:00",
        "window_end": "2026-06-24T18:00:00+09:00",
        "segment": {"channel": "kakao", "age_group": "30s"},
        "causes": [
            {
                "cause_id": "cause-1",
                "cause_type": "LOW_VIEW_TO_CART_RATE",
                "description": "상품 상세에서 장바구니 전환이 낮습니다.",
                "affected_step": "product_view_to_add_to_cart",
                "severity": 0.8,
                "confidence": 0.7,
                "evidence": [],
                "attributes": {"metric": "view_to_cart_rate"},
            }
        ],
        "top_n": 5,
    }
    values.update(overrides)
    return values


def root_cause_candidate_payload() -> dict[str, object]:
    return {
        "rank": 1,
        "cause_type": "channel_specific_drop",
        "dimension": "channel",
        "value": "kakao",
        "metric": "view_to_purchase_rate",
        "funnel_step": "product_view_to_purchase",
        "severity": "critical",
        "current_value": 0.10,
        "baseline_value": 0.30,
        "drop_point": 0.20,
        "relative_drop": 0.67,
        "current_denominator": 100,
        "baseline_denominator": 120,
        "support_share": 0.40,
        "excess_lost_sessions": 20.0,
        "score": 2.4,
        "message": "channel=kakao 전환율이 하락했습니다.",
    }


def test_post_actions_recommend_returns_recommendations() -> None:
    response = TestClient(app).post("/actions/recommend", json=action_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "loopad-demo-shop"
    assert body["segment"]["channel"] == "kakao"
    assert "recommendations" in body
    assert body["recommendations"][0]["action_id"]


def test_post_actions_recommend_accepts_root_cause_candidate_payload() -> None:
    response = TestClient(app).post(
        "/actions/recommend",
        json=action_payload(causes=[root_cause_candidate_payload()]),
    )

    assert response.status_code == 200
    body = response.json()
    ids = [action["action_id"] for action in body["recommendations"]]
    assert "adjust_landing_page" in ids


def test_post_actions_recommend_rejects_naive_datetime() -> None:
    response = TestClient(app).post(
        "/actions/recommend",
        json=action_payload(window_start="2026-06-24T17:00:00"),
    )

    assert response.status_code == 400


def test_post_actions_recommend_rejects_window_start_after_window_end() -> None:
    response = TestClient(app).post(
        "/actions/recommend",
        json=action_payload(
            window_start="2026-06-24T18:00:00+09:00",
            window_end="2026-06-24T17:00:00+09:00",
        ),
    )

    assert response.status_code == 400


@pytest.mark.parametrize("top_n", [0, 21])
def test_post_actions_recommend_rejects_invalid_top_n(top_n: int) -> None:
    response = TestClient(app).post(
        "/actions/recommend",
        json=action_payload(top_n=top_n),
    )

    assert response.status_code == 400
