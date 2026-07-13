from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.analysis.repositories import PromotionAnalysisWrite, PromotionTargetSegmentWrite
from app.analysis.router import get_analysis_service
from app.analysis.schemas import AnalysisStatus
from app.analysis.service import PromotionAnalysisResult
from app.config import REQUIRED_ENV_NAMES, load_settings
from app.decision.router import _manual_next_loop_enabled
from app.decision.schemas import (
    AdExperimentCreateResponse,
    NextLoopPreparationStatus,
    NextLoopRequest,
    NextLoopResponse,
    RunCreateRequest,
    RunCreateResponse,
)
from app.generation.artifacts import render_banner_html
from app.generation.router import get_generation_service
from app.generation.service import GenerationService
from app.main import create_app


README_PATH = Path(__file__).resolve().parents[1] / "README.md"
RUN_RESPONSE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "contracts"
    / "decision-promotion-run-response.v1.json"
)


TERM_ARM_ID = "arm" + "_id"
TERM_BANDIT = "ban" + "dit"
TERM_THOMPSON_SAMPLING = "Thompson" + " " + "Sampling"
TERM_ANOMALY = "ano" + "maly"
TERM_ROOT_CAUSE = "root" + "_cause"
TERM_RECOMMENDATION_API = "recommendation" + " API"
TERM_RECOMMENDATIONS_PATH = "/recomm" + "endations"
TERM_RECOMMENDATION_RESULT = "recommendation" + "_result"
TERM_PLAIN_EXPERIMENT_ID = "experiment" + "_id"
TERM_VARIANT_ID = "variant" + "_id"
TERM_PRODUCT = "pro" + "duct"
TERM_CART = "ca" + "rt"
TERM_PURCHASE = "pur" + "chase"

FORBIDDEN_PUBLIC_PATTERNS = {
    TERM_ARM_ID: re.compile(rf"\b{re.escape(TERM_ARM_ID)}\b", re.IGNORECASE),
    TERM_BANDIT: re.compile(rf"\b{re.escape(TERM_BANDIT)}\b", re.IGNORECASE),
    TERM_THOMPSON_SAMPLING: re.compile(r"\bthompson\s+sampling\b", re.IGNORECASE),
    TERM_ANOMALY: re.compile(rf"\b{re.escape(TERM_ANOMALY)}\b", re.IGNORECASE),
    TERM_ROOT_CAUSE: re.compile(rf"\b{re.escape(TERM_ROOT_CAUSE)}\b", re.IGNORECASE),
    TERM_RECOMMENDATION_API: re.compile(r"\brecommendation\s+api\b", re.IGNORECASE),
    TERM_RECOMMENDATIONS_PATH: re.compile(
        rf"{re.escape(TERM_RECOMMENDATIONS_PATH)}\b", re.IGNORECASE
    ),
    TERM_RECOMMENDATION_RESULT: re.compile(
        rf"\b{re.escape(TERM_RECOMMENDATION_RESULT)}\b", re.IGNORECASE
    ),
    TERM_PLAIN_EXPERIMENT_ID: re.compile(
        rf"(?<!ad_)\b{re.escape(TERM_PLAIN_EXPERIMENT_ID)}\b", re.IGNORECASE
    ),
    TERM_VARIANT_ID: re.compile(rf"\b{re.escape(TERM_VARIANT_ID)}\b", re.IGNORECASE),
}

FORBIDDEN_SHOPPING_PATTERNS = {
    TERM_PRODUCT: re.compile(rf"\b{re.escape(TERM_PRODUCT)}\b", re.IGNORECASE),
    TERM_CART: re.compile(rf"\b{re.escape(TERM_CART)}\b", re.IGNORECASE),
    TERM_PURCHASE: re.compile(rf"\b{re.escape(TERM_PURCHASE)}\b", re.IGNORECASE),
}


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


def make_client() -> TestClient:
    app = create_app(settings=load_settings(valid_env()))
    app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService()
    app.dependency_overrides[get_generation_service] = lambda: GenerationService()
    return TestClient(app)


def test_public_outputs_do_not_expose_forbidden_terms() -> None:
    payload_text = public_payload_text(make_client())

    for label, pattern in FORBIDDEN_PUBLIC_PATTERNS.items():
        assert not pattern.search(payload_text), f"forbidden public term: {label}"


def test_ad_experiment_id_is_allowed_but_plain_experiment_id_is_forbidden() -> None:
    allowed_payload = json.dumps({"ad_experiment_id": "adexp_luxury_001"})
    forbidden_payload = json.dumps({TERM_PLAIN_EXPERIMENT_ID: "legacy_exp_001"})
    pattern = FORBIDDEN_PUBLIC_PATTERNS[TERM_PLAIN_EXPERIMENT_ID]

    assert not pattern.search(allowed_payload)
    assert pattern.search(forbidden_payload)


def test_recommendation_word_is_not_banned_unless_exposed_as_public_api_object() -> None:
    benign_payload = "The model can make an internal recommendation-like choice."
    forbidden_payloads = (
        f"Public {TERM_RECOMMENDATION_API}",
        f"/decision/v1{TERM_RECOMMENDATIONS_PATH}",
        json.dumps({TERM_RECOMMENDATION_RESULT: {}}),
    )

    assert not FORBIDDEN_PUBLIC_PATTERNS[TERM_RECOMMENDATION_API].search(benign_payload)
    for payload in forbidden_payloads:
        assert any(
            pattern.search(payload)
            for label, pattern in FORBIDDEN_PUBLIC_PATTERNS.items()
            if label
            in {
                TERM_RECOMMENDATION_API,
                TERM_RECOMMENDATIONS_PATH,
                TERM_RECOMMENDATION_RESULT,
            }
        )


def test_public_outputs_do_not_use_shopping_terms() -> None:
    payload_text = public_payload_text(make_client())

    for label, pattern in FORBIDDEN_SHOPPING_PATTERNS.items():
        assert not pattern.search(payload_text), f"shopping term leaked: {label}"


def test_manual_next_loop_contract_does_not_expose_forbidden_public_terms() -> None:
    payload_text = json.dumps(
        NextLoopResponse(
            status=NextLoopPreparationStatus.AWAITING_CONTENT_APPROVAL,
            content_approval_required=True,
            next_loop_preparation_id="nlprep_banner_001_loop_2_attempt_1",
            previous_promotion_run_id="prun_banner_001_loop_1",
            next_promotion_run_id=None,
            promotion_id="promo_banner_001",
            loop_count=2,
            segment_ids=["seg_luxury"],
            next_analysis_id="analysis_banner_002",
            next_generation_id="generation_banner_002_attempt_1",
            pending_content_ids=["content_banner_002_option_1"],
            next_ad_experiments=[],
        ).model_dump(mode="json"),
        ensure_ascii=False,
    )

    for label, pattern in {
        **FORBIDDEN_PUBLIC_PATTERNS,
        **FORBIDDEN_SHOPPING_PATTERNS,
    }.items():
        assert not pattern.search(payload_text), f"forbidden public term: {label}"


def test_manual_next_loop_fields_are_additive_in_public_schemas() -> None:
    request_schema = NextLoopRequest.model_json_schema()
    run_request_schema = RunCreateRequest.model_json_schema()
    openapi_schema = make_client().get("/openapi.json").json()
    response_schema = openapi_schema["components"]["schemas"]["NextLoopResponse"]
    preparation_status_schema = openapi_schema["components"]["schemas"][
        "NextLoopPreparationStatus"
    ]

    assert request_schema["properties"]["content_approval_mode"]["default"] == (
        "automatic"
    )
    assert set(request_schema["$defs"]["ContentApprovalMode"]["enum"]) == {
        "automatic",
        "manual",
    }
    assert "next_loop_preparation_id" in run_request_schema["properties"]
    assert {
        "status",
        "content_approval_required",
        "next_loop_preparation_id",
        "pending_content_ids",
    } <= response_schema["properties"].keys()
    assert {
        "status",
        "content_approval_required",
        "next_loop_preparation_id",
        "pending_content_ids",
    }.isdisjoint(response_schema.get("required", []))
    assert set(preparation_status_schema["enum"]) == {
        "awaiting_content_approval",
        "activated",
        "rejected",
    }


def test_automatic_next_loop_serialization_keeps_legacy_wire_shape() -> None:
    response = NextLoopResponse(
        previous_promotion_run_id="prun_banner_001_loop_1",
        next_promotion_run_id="prun_banner_001_loop_2",
        promotion_id="promo_banner_001",
        loop_count=2,
        segment_ids=["seg_luxury"],
        next_analysis_id="analysis_banner_002",
        next_generation_id="generation_banner_002",
        next_ad_experiments=[],
    )

    assert set(response.model_dump(mode="json")) == {
        "previous_promotion_run_id",
        "next_promotion_run_id",
        "promotion_id",
        "loop_count",
        "segment_ids",
        "next_analysis_id",
        "next_generation_id",
        "next_ad_experiments",
    }


def test_manual_next_loop_switch_is_off_until_explicitly_enabled() -> None:
    app = create_app(settings=load_settings(valid_env()))
    request = SimpleNamespace(app=app)

    assert _manual_next_loop_enabled(request) is False

    app.state.manual_next_loop_enabled = True
    assert _manual_next_loop_enabled(request) is True


def test_decision_routes_exclude_dashboard_reads_and_hot_paths() -> None:
    app = create_app(settings=load_settings(valid_env()))
    routes = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
    }

    assert not {
        path
        for method, path in routes
        if method == "GET" and path.startswith("/decision/v1")
    }
    assert ("POST", "/decision/v1/segments/query-preview") not in routes
    assert ("POST", "/decision/v1/segments") not in routes
    assert not {path for _method, path in routes if "/chatkit/" in path}
    assert (
        "POST",
        "/decision/v1/promotion-runs/{promotion_run_id}/segment-match",
    ) not in routes
    assert (
        "GET",
        "/decision/v1/promotion-runs/{promotion_run_id}/active-contents",
    ) not in routes


def test_run_response_exposes_segment_scope_and_fallback_marker() -> None:
    run_schema = RunCreateResponse.model_json_schema()
    experiment_schema = AdExperimentCreateResponse.model_json_schema()
    next_loop_schema = NextLoopResponse.model_json_schema()

    assert "segment_ids" in run_schema["required"]
    assert "is_fallback" in experiment_schema["required"]
    assert "segment_ids" in next_loop_schema["required"]


def test_dashboard_run_response_fixture_matches_public_contract() -> None:
    response = RunCreateResponse.model_validate_json(
        RUN_RESPONSE_FIXTURE_PATH.read_text(encoding="utf-8")
    )
    non_fallback_segment_ids = {
        experiment.segment_id
        for experiment in response.ad_experiments
        if not experiment.is_fallback
    }
    fallback_experiments = [
        experiment
        for experiment in response.ad_experiments
        if experiment.is_fallback
    ]

    assert response.segment_ids == ["segment-a", "segment-b"]
    assert non_fallback_segment_ids == set(response.segment_ids)
    assert len(fallback_experiments) == 1
    assert fallback_experiments[0].segment_id == "seg_existing_all"


def test_banner_artifact_html_uses_hotel_booking_language_not_shopping_terms() -> None:
    artifact_text = render_banner_html(
        {
            "title": "이번 주말 호텔 특가",
            "body": "환불 가능한 객실과 숙박 혜택을 지금 비교해보세요.",
            "cta": "호텔 특가 보기",
        }
    ).lower()

    assert "호텔" in artifact_text
    assert "특가" in artifact_text
    for label, pattern in FORBIDDEN_SHOPPING_PATTERNS.items():
        assert not pattern.search(artifact_text), f"shopping term leaked: {label}"


def test_readme_documents_decision_boundary_and_dashboard_serving_path() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "lifecycle write API" in readme
    assert "Decision hot path" in readme
    assert "active_ad_serving_assignments" in readme
    assert "Data Source Contract" in readme
    assert "does not provide active_ad_serving_assignments" in readme
    assert "B6 next-loop" in readme
    assert "follow-up integration PR" in readme


class FakeAnalysisService:
    def recommend_segments(self, request: Any) -> PromotionAnalysisResult:
        return PromotionAnalysisResult(
            analysis=PromotionAnalysisWrite(
                analysis_id=f"analysis_{request.promotion_id}",
                project_id=request.project_id,
                campaign_id=request.campaign_id,
                promotion_id=request.promotion_id,
                status=AnalysisStatus.COMPLETED.value,
                focus_segment_ids_json=None,
                operator_instruction=request.operator_instruction,
                input_snapshot_json={},
                profile_summary_json={},
                output_json={},
            ),
            target_segments=[
                PromotionTargetSegmentWrite(
                    analysis_id=f"analysis_{request.promotion_id}",
                    project_id=request.project_id,
                    campaign_id=request.campaign_id,
                    promotion_id=request.promotion_id,
                    segment_id="seg_repeat_hotel_no_booking",
                    segment_name="Repeat hotel viewers without booking",
                    rule_json={},
                    profile_json={},
                    content_brief_json={
                        "schema_version": "content_brief.v2",
                        "readiness": {
                            "level": "fallback_only",
                            "available_sections": ["fallback_guidance"],
                            "missing_sections": [
                                "primary_signals",
                                "score_components",
                            ],
                        },
                        "fallback_guidance": {
                            "message_direction": (
                                "호텔 예약 가능성과 조식 특가 혜택을 강조한다."
                            ),
                            "keywords": [
                                "호텔 예약",
                                "당일 예약 가능",
                                "조식 특가",
                            ],
                            "source": "legacy_segment_content_hints",
                        },
                    },
                    data_evidence_json={},
                    segment_vector_id="segvec_repeat_hotel_no_booking_v1",
                    estimated_size=1342,
                    priority="high",
                    status="planned",
                )
            ],
        )


def public_payload_text(client: TestClient) -> str:
    payloads: list[Any] = []
    health_response = client.get("/health")
    assert health_response.status_code == 200
    payloads.append(health_response.json())

    analysis_response = client.post(
        "/decision/v1/promotions/promo_banner_001/segment-suggestions/recommend",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "operator_instruction": None,
        },
    )
    assert analysis_response.status_code == 200
    payloads.append(analysis_response.json())

    generation_response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "analysis_id": "analysis_banner_001",
            "content_option_count": 1,
            "operator_instruction": None,
        },
    )
    assert generation_response.status_code == 200
    payloads.append(generation_response.json())

    return json.dumps(payloads, ensure_ascii=False).lower()
