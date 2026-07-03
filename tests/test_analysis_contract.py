from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi.testclient import TestClient

from app.analysis.schemas import AnalysisStatus, Channel, GoalBasis, GoalMetric
from app.main import create_app


client = TestClient(create_app())

FORBIDDEN_PUBLIC_TERMS = (
    "recommendation",
    "anomaly",
    "root_cause",
    "arm_id",
    "bandit",
    "thompson",
    "experiment_id",
    "variant_id",
    "creative_id",
    "product",
    "cart",
    "purchase",
)


def analysis_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "focus_segment_ids": None,
        "operator_instruction": None,
    }
    payload.update(overrides)
    return payload


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        strings: list[str] = []
        for key, child in value.items():
            strings.append(str(key))
            strings.extend(collect_strings(child))
        return strings

    if isinstance(value, list):
        strings = []
        for child in value:
            strings.extend(collect_strings(child))
        return strings

    if isinstance(value, str):
        return [value]

    return []


def enum_values(enum_type: type[Enum]) -> set[str]:
    return {member.value for member in enum_type}


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "decision-api"}


def test_analysis_returns_v1_6_contract_shape() -> None:
    response = client.post(
        "/decision/v1/promotions/promo_banner_001/analysis",
        json=analysis_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == "analysis_promo_banner_001"
    assert body["promotion_id"] == "promo_banner_001"
    assert body["status"] == "completed"
    assert body["target_segments"] == [
        {
            "segment_id": "seg_repeat_hotel_no_booking",
            "segment_name": "Repeat hotel viewers without booking",
            "segment_vector_id": "segvec_repeat_hotel_no_booking_v1",
            "estimated_size": 1342,
            "content_brief": {
                "message_direction": (
                    "Emphasize free cancellation, same-day availability, "
                    "and breakfast benefits."
                ),
                "keywords": [
                    "free cancellation",
                    "same-day availability",
                    "breakfast included",
                ],
            },
        }
    ]


def test_analysis_requires_mandatory_fields() -> None:
    response = client.post(
        "/decision/v1/promotions/promo_banner_001/analysis",
        json={"project_id": "hotel-client-a"},
    )

    assert response.status_code == 422


def test_analysis_rejects_promotion_id_mismatch() -> None:
    response = client.post(
        "/decision/v1/promotions/promo_banner_001/analysis",
        json=analysis_payload(promotion_id="promo_email_001"),
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "path promotion_id must match request promotion_id"
    }


def test_analysis_response_does_not_expose_forbidden_terms() -> None:
    response = client.post(
        "/decision/v1/promotions/promo_banner_001/analysis",
        json=analysis_payload(),
    )

    assert response.status_code == 200
    response_text = " ".join(collect_strings(response.json())).lower()
    for term in FORBIDDEN_PUBLIC_TERMS:
        assert term not in response_text


def test_v1_6_enum_values() -> None:
    assert enum_values(Channel) == {"email", "sms", "onsite_banner"}
    assert enum_values(GoalMetric) == {
        "inflow_rate",
        "booking_conversion_rate",
        "funnel_step_rate",
    }
    assert enum_values(GoalBasis) == {"promotion_average", "all_segments"}
    assert enum_values(AnalysisStatus) == {
        "requested",
        "running",
        "completed",
        "failed",
    }
