from fastapi.testclient import TestClient

from app.main import create_app


FORBIDDEN_PUBLIC_KEYS = {"creative_id", "variant_id", "experiment_id"}


def test_generation_api_returns_v1_6_final_names() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 3,
            "operator_instruction": None,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["generation_id"] == "generation_banner_001"
    assert payload["promotion_id"] == "promo_banner_001"
    assert payload["status"] == "completed"
    assert len(payload["content_candidates"]) == 3
    assert "content_candidates" in payload
    assert_no_forbidden_public_keys(payload)

    first_candidate = payload["content_candidates"][0]
    assert first_candidate["content_id"] == "content_banner_repeat_hotel_001"
    assert first_candidate["content_option_id"] == "banner_repeat_hotel_option_001"
    assert first_candidate["segment_id"] == "seg_repeat_hotel_no_booking"
    assert first_candidate["channel"] == "onsite_banner"
    assert first_candidate["status"] == "draft"
    assert first_candidate["title"]
    assert first_candidate["body"]
    assert first_candidate["cta"]
    assert first_candidate["image_prompt"]
    assert first_candidate["landing_url"] == "https://demo-stay.example.com/summer"


def test_generation_api_rejects_path_body_promotion_mismatch() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_other_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": None,
        },
    )

    assert response.status_code == 400
    assert "promotion_id" in response.json()["detail"]


def test_generation_api_rejects_non_positive_content_option_count() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 0,
            "operator_instruction": None,
        },
    )

    assert response.status_code == 400


def test_health_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "decision-api"}


def assert_no_forbidden_public_keys(value) -> None:
    if isinstance(value, dict):
        assert not (set(value) & FORBIDDEN_PUBLIC_KEYS)
        for item in value.values():
            assert_no_forbidden_public_keys(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_forbidden_public_keys(item)
