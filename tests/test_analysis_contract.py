from __future__ import annotations

from enum import Enum
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.analysis.schemas import AnalysisStatus, Channel, GoalBasis, GoalMetric
from app.config import REQUIRED_ENV_NAMES, SettingsError, load_settings
from app.main import create_app


def valid_env() -> dict[str, str]:
    values = {name: f"value-for-{name.lower()}" for name in REQUIRED_ENV_NAMES}
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "15432",
            "LOOPAD_OPENAI_CONTENT_MODEL": "gpt-test",
        }
    )
    return values


client = TestClient(create_app(settings=load_settings(valid_env())))

FORBIDDEN_PUBLIC_TERMS = tuple(
    "".join(parts)
    for parts in (
        ("recom", "mendation"),
        ("ano", "maly"),
        ("root", "_cause"),
        ("arm", "_id"),
        ("ban", "dit"),
        ("thomp", "son"),
        ("experiment", "_id"),
        ("variant", "_id"),
        ("creative", "_id"),
        ("pro", "duct"),
        ("ca", "rt"),
        ("pur", "chase"),
    )
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
    assert response.json() == {
        "status": "ok",
        "service": "decision-api",
        "env": "test",
    }


def test_server_port_comes_from_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "8080")

    settings = load_settings(valid_env() | {"PORT": "8080"})

    assert settings.port == 8080


def test_server_port_has_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)

    env = valid_env()
    env.pop("PORT")
    with pytest.raises(SettingsError, match="PORT"):
        load_settings(env)


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
