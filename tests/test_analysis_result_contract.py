from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Sequence

from fastapi.testclient import TestClient

from app.analysis.repositories import (
    BookingTrainingRecord,
    HotelMarketingProfileRecord,
    PromotionAnalysisWrite,
    PromotionRecord,
    PromotionSegmentSuggestionWrite,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
)
from app.analysis.router import get_analysis_service
from app.analysis.schemas import AnalysisRequest
from app.analysis.service import PromotionAnalysisService
from app.analysis.vector_service import (
    SegmentVectorBuildRequest,
    SegmentVectorBuildResult,
)
from app.config import REQUIRED_ENV_NAMES, load_settings
from app.main import create_app


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


@dataclass
class SavedAnalysis:
    analysis: PromotionAnalysisWrite | None = None
    target_segments: list[PromotionTargetSegmentWrite] | None = None
    segment_suggestions: list[PromotionSegmentSuggestionWrite] | None = None


class FakePromotionRepository:
    def __init__(self, promotion: PromotionRecord) -> None:
        self.promotion = promotion

    def get_for_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionRecord | None:
        if (
            project_id,
            campaign_id,
            promotion_id,
        ) != (
            self.promotion.project_id,
            self.promotion.campaign_id,
            self.promotion.promotion_id,
        ):
            return None
        return self.promotion


class FakeSegmentDefinitionRepository:
    def __init__(self, segments: list[SegmentDefinitionRecord]) -> None:
        self.segments = segments

    def list_active(
        self,
        *,
        project_id: str,
        campaign_id: str | None = None,
        promotion_id: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        del campaign_id, promotion_id
        return [
            segment
            for segment in self.segments
            if segment.project_id == project_id and segment.status == "active"
        ]


class FakeHotelProfileRepository:
    def __init__(self, profiles: list[HotelMarketingProfileRecord]) -> None:
        self.profiles = profiles

    def list_marketing_profiles(
        self,
        *,
        project_id: str,
    ) -> list[HotelMarketingProfileRecord]:
        return [profile for profile in self.profiles if profile.project_id == project_id]

    def summarize_user_ids(
        self,
        *,
        project_id: str,
        profile_name: str,
        user_ids: Sequence[str],
    ) -> HotelMarketingProfileRecord | None:
        del user_ids
        for profile in self.profiles:
            if profile.project_id == project_id and profile.profile_name == profile_name:
                return profile
        return None

    def list_booking_training_records(
        self,
        *,
        limit: int = 500,
    ) -> list[BookingTrainingRecord]:
        del limit
        return []


class FakePromotionAnalysisRepository:
    def __init__(self) -> None:
        self.saved = SavedAnalysis()
        self.events: list[str] = []

    def save_analysis(self, analysis: PromotionAnalysisWrite) -> None:
        self.saved.analysis = analysis
        self.events.append("analysis")

    def save_target_segments(
        self,
        target_segments: Sequence[PromotionTargetSegmentWrite],
    ) -> None:
        self.saved.target_segments = list(target_segments)
        self.events.append("target_segments")

    def save_segment_suggestions(
        self,
        suggestions: Sequence[PromotionSegmentSuggestionWrite],
    ) -> None:
        self.saved.segment_suggestions = list(suggestions)
        self.events.append("segment_suggestions")


class FakeSegmentVectorService:
    def __init__(self) -> None:
        self.calls: list[SegmentVectorBuildRequest] = []

    def prepare_segment_vector(
        self,
        request: SegmentVectorBuildRequest,
    ) -> SegmentVectorBuildResult:
        self.calls.append(request)
        return SegmentVectorBuildResult(
            segment_id=request.segment_id,
            segment_vector_id=f"segvec_{request.segment_id}_v1",
            vector_values=[1.0, *([0.0] * 63)],
            source="decision_analysis",
        )


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


def make_analysis_service() -> PromotionAnalysisService:
    return PromotionAnalysisService(
        promotion_repository=FakePromotionRepository(promotion_record()),
        segment_definition_repository=FakeSegmentDefinitionRepository(
            default_segments()
        ),
        hotel_profile_repository=FakeHotelProfileRepository(
            [
                profile_record("seg_repeat_hotel_no_booking", event_count=5000),
                profile_record("seg_family_trip", event_count=3000),
            ]
        ),
        promotion_analysis_repository=FakePromotionAnalysisRepository(),
        segment_vector_service=FakeSegmentVectorService(),
    )


def make_client() -> TestClient:
    app = create_app(settings=load_settings(valid_env()))
    app.dependency_overrides[get_analysis_service] = make_analysis_service
    return TestClient(app)


def analysis_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "operator_instruction": None,
    }
    payload.update(overrides)
    return payload


def promotion_record() -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel="onsite_banner",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.030000"),
        goal_basis="all_segments",
        min_sample_size=1000,
        landing_url="https://demo-stay.example.com/hotel-detail",
        message_brief="Drive hotel booking conversion for summer stays.",
    )


def segment_record(
    segment_id: str,
    *,
    source: str = "system_default",
    sample_size: int = 2000,
    sample_ratio: Decimal = Decimal("0.020000"),
    rule_json: Mapping[str, Any] | None = None,
) -> SegmentDefinitionRecord:
    return SegmentDefinitionRecord(
        segment_id=segment_id,
        project_id="hotel-client-a",
        segment_name=segment_id.replace("_", " ").title(),
        source=source,
        query_preview_id=None,
        natural_language_query=f"{segment_id} hotel booking audience",
        generated_sql=None,
        rule_json=rule_json or {"primary_segment": segment_id},
        profile_json={"primary_segment": segment_id},
        sample_size=sample_size,
        total_eligible_user_count=74200,
        sample_ratio=sample_ratio,
        status="active",
    )


def profile_record(
    profile_name: str,
    *,
    event_count: int,
) -> HotelMarketingProfileRecord:
    return HotelMarketingProfileRecord(
        project_id="hotel-client-a",
        profile_name=profile_name,
        profile_json={
            "event_count": event_count,
            "booking_count": 120,
            "mobile_ratio": 0.65,
            "package_ratio": 0.25,
            "avg_stay_nights": 2.4,
            "avg_days_until_checkin": 14.2,
        },
    )


def default_segments() -> list[SegmentDefinitionRecord]:
    return [
        segment_record("seg_family_trip", sample_size=2400),
        segment_record(
            "seg_mobile_user",
            sample_size=2200,
            rule_json={
                "primary_segment": "seg_mobile_user",
                "candidate_user_ids": ["user_001", "user_002"],
            },
        ),
        segment_record("seg_repeat_hotel_no_booking", sample_size=1342),
        segment_record("seg_near_checkin", sample_size=1800),
        segment_record("seg_existing_all", sample_size=5000),
    ]


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
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

    if isinstance(value, Enum):
        return [value.value]

    if isinstance(value, str):
        return [value]

    return []


def assert_no_forbidden_public_terms(payload: Any) -> None:
    response_text = " ".join(collect_strings(payload)).lower()
    for term in FORBIDDEN_PUBLIC_TERMS:
        assert term not in response_text


def test_analysis_api_response_snapshot_for_dashboard_contract() -> None:
    response = make_client().post(
        "/decision/v1/promotions/promo_banner_001/segment-suggestions/recommend",
        json=analysis_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert re.fullmatch(
        r"analysis_promo_banner_001_run_[0-9a-f]{8}",
        body["analysis_id"],
    )
    assert body == {
        "analysis_id": body["analysis_id"],
        "promotion_id": "promo_banner_001",
        "status": "completed",
        "target_segments": [
            {
                "segment_id": "seg_family_trip",
                "segment_name": "Seg Family Trip",
                "segment_vector_id": "segvec_seg_family_trip_v1",
                "estimated_size": 2400,
                "content_brief": {
                    "message_direction": (
                        "Highlight family rooms, breakfast, and flexible cancellation."
                    ),
                    "keywords": [
                        "family room",
                        "breakfast included",
                        "flexible cancellation",
                    ],
                },
            },
            {
                "segment_id": "seg_mobile_user",
                "segment_name": "Seg Mobile User",
                "segment_vector_id": "segvec_seg_mobile_user_v1",
                "estimated_size": 2200,
                "content_brief": {
                    "message_direction": (
                        "Reduce steps and emphasize mobile-friendly booking."
                    ),
                    "keywords": [
                        "mobile booking",
                        "quick checkout",
                        "easy reservation",
                    ],
                },
            },
            {
                "segment_id": "seg_repeat_hotel_no_booking",
                "segment_name": "Seg Repeat Hotel No Booking",
                "segment_vector_id": "segvec_seg_repeat_hotel_no_booking_v1",
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
            },
            {
                "segment_id": "seg_near_checkin",
                "segment_name": "Seg Near Checkin",
                "segment_vector_id": "segvec_seg_near_checkin_v1",
                "estimated_size": 1800,
                "content_brief": {
                    "message_direction": (
                        "Emphasize near check-in availability and low-friction booking."
                    ),
                    "keywords": [
                        "near check-in",
                        "same-day availability",
                        "free cancellation",
                    ],
                },
            },
        ],
    }
    assert_no_forbidden_public_terms(response.json())


def test_analysis_api_rejects_focus_segment_ids_contract() -> None:
    response = make_client().post(
        "/decision/v1/promotions/promo_banner_001/segment-suggestions/recommend",
        json=analysis_payload(focus_segment_ids=["seg_near_checkin"]),
    )

    assert response.status_code == 422


def test_analysis_service_persists_dashboard_db_contract() -> None:
    analysis_repository = FakePromotionAnalysisRepository()
    vector_service = FakeSegmentVectorService()
    service = PromotionAnalysisService(
        promotion_repository=FakePromotionRepository(promotion_record()),
        segment_definition_repository=FakeSegmentDefinitionRepository(
            default_segments()
        ),
        hotel_profile_repository=FakeHotelProfileRepository(
            [
                profile_record("seg_repeat_hotel_no_booking", event_count=5000),
                profile_record("seg_family_trip", event_count=3000),
            ]
        ),
        promotion_analysis_repository=analysis_repository,
        segment_vector_service=vector_service,
    )

    result = service.analyze(
        AnalysisRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            operator_instruction=None,
        )
    )

    saved_analysis = analysis_repository.saved.analysis
    saved_suggestions = analysis_repository.saved.segment_suggestions
    assert saved_analysis is not None
    assert saved_suggestions is not None
    assert analysis_repository.events == ["analysis", "segment_suggestions"]
    assert saved_analysis == result.analysis
    assert saved_suggestions == result.segment_suggestions

    expected_segment_ids = [
        "seg_family_trip",
        "seg_mobile_user",
        "seg_repeat_hotel_no_booking",
        "seg_near_checkin",
    ]
    saved_segments = result.target_segments
    assert [segment.segment_id for segment in saved_suggestions] == expected_segment_ids
    assert [segment.segment_id for segment in saved_segments] == expected_segment_ids
    assert len(saved_suggestions) == saved_analysis.output_json["target_segment_count"]
    expected_signal_chips = {
        "seg_family_trip": ["가족 여행 관심"],
        "seg_mobile_user": ["모바일 이용"],
        "seg_repeat_hotel_no_booking": ["반복 조회"],
        "seg_near_checkin": ["임박 예약 관심"],
    }
    assert saved_analysis.output_json == {
        "selected_segment_ids": expected_segment_ids,
        "target_segment_count": 4,
    }
    assert saved_analysis.profile_summary_json == {
        "total_eligible_users": 74200,
        "candidate_segment_count": 5,
        "selected_segment_count": 4,
        "selection_mode": "default",
        "reason": (
            "Selected hotel audience segments by channel, goal metric, "
            "and active segment definitions."
        ),
    }
    assert saved_analysis.focus_segment_ids_json is None
    assert set(saved_analysis.input_snapshot_json) == {
        "promotion",
        "available_segment_definitions",
        "focus_segment_ids",
        "operator_instruction",
    }
    assert saved_analysis.input_snapshot_json["focus_segment_ids"] is None
    assert saved_analysis.input_snapshot_json["promotion"] == {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "channel": "onsite_banner",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": "0.030000",
        "goal_basis": "all_segments",
        "min_sample_size": 1000,
        "landing_url": "https://demo-stay.example.com/hotel-detail",
        "message_brief": "Drive hotel booking conversion for summer stays.",
    }
    assert all(
        segment["segment_id"].startswith("seg_")
        for segment in saved_analysis.input_snapshot_json[
            "available_segment_definitions"
        ]
    )
    assert re.fullmatch(
        r"analysis_promo_banner_001_run_[0-9a-f]{8}",
        saved_analysis.analysis_id,
    )

    for rank, segment in enumerate(saved_segments):
        assert segment.analysis_id == saved_analysis.analysis_id
        assert segment.project_id == "hotel-client-a"
        assert segment.campaign_id == "camp_summer_2026"
        assert segment.promotion_id == "promo_banner_001"
        assert segment.segment_vector_id == f"segvec_{segment.segment_id}_v1"
        assert segment.estimated_size >= 0
        assert segment.status == "planned"
        assert segment.priority == ("high" if rank < 2 else "medium")
        content_brief = segment.content_brief_json
        assert content_brief["schema_version"] == "content_brief.v2"
        assert "message_direction" not in content_brief
        assert "keywords" not in content_brief
        assert set(content_brief["readiness"]["missing_sections"]) == {
            "primary_signals",
            "score_components",
        }
        assert content_brief["readiness"]["level"] == "fallback_only"
        assert content_brief["segment_snapshot"] == {
            "segment_id": segment.segment_id,
            "segment_name": segment.segment_name,
            "segment_source": "system_default",
            "estimated_size": segment.estimated_size,
            "segment_vector_id": f"segvec_{segment.segment_id}_v1",
        }
        assert content_brief["promotion_context"]["goal_metric"] == (
            "booking_conversion_rate"
        )
        assert isinstance(content_brief["fallback_guidance"]["keywords"], list)
        assert (
            content_brief["fallback_guidance"]["source"]
            == "legacy_segment_content_hints"
        )
        assert segment.data_evidence_json["source"] == "system_default"
        assert segment.data_evidence_json["sample_size"] == segment.estimated_size
        assert "sample_ratio" in segment.data_evidence_json
        assert "total_eligible_user_count" in segment.data_evidence_json
        assert segment.profile_json["primary_segment"] == segment.segment_id

    for rank, suggestion in enumerate(saved_suggestions):
        assert suggestion.analysis_id == saved_analysis.analysis_id
        assert suggestion.project_id == "hotel-client-a"
        assert suggestion.campaign_id == "camp_summer_2026"
        assert suggestion.promotion_id == "promo_banner_001"
        assert suggestion.segment_id == expected_segment_ids[rank]
        assert suggestion.suggested_rank == rank + 1
        assert suggestion.suggestion_source == "ai_ranked_existing"
        assert suggestion.status == "suggested"
        assert suggestion.metadata_json["segment_name"] == saved_segments[rank].segment_name
        assert suggestion.metadata_json["segment_vector_id"] == (
            f"segvec_{suggestion.segment_id}_v1"
        )
        assert suggestion.reason_json["primary_signals"]
        display_copy = suggestion.metadata_json["display_copy"]
        assert set(display_copy) == {
            "title",
            "audience_summary",
            "signal_chips",
            "reason",
            "action_hint",
        }
        assert display_copy["signal_chips"] == expected_signal_chips[
            suggestion.segment_id
        ]
        assert display_copy["audience_summary"].startswith("분석 대상 74200명 중 ")
        assert display_copy["reason"] == (
            "예약 전환 목표에 가까운 행동 패턴을 보인 고객군입니다."
        )

    family_segment = saved_segments[0]
    repeat_hotel_segment = saved_segments[2]
    assert "hotel_profile" not in family_segment.profile_json
    assert family_segment.content_brief_json["hotel_profile"] == {
        "event_count": 3000,
        "booking_count": 120,
        "mobile_ratio": 0.65,
        "package_ratio": 0.25,
        "avg_stay_nights": 2.4,
        "avg_days_until_checkin": 14.2,
    }
    assert repeat_hotel_segment.data_evidence_json["hotel_profile"] == {
        "event_count": 5000,
        "booking_count": 120,
        "mobile_ratio": 0.65,
        "package_ratio": 0.25,
        "avg_stay_nights": 2.4,
        "avg_days_until_checkin": 14.2,
    }

    assert [call.segment_id for call in vector_service.calls] == expected_segment_ids
    assert vector_service.calls[1].candidate_user_ids == [
        "user_001",
        "user_002",
    ]
    assert_no_forbidden_public_terms(saved_analysis.input_snapshot_json)
    assert_no_forbidden_public_terms(saved_analysis.profile_summary_json)
    assert_no_forbidden_public_terms(saved_analysis.output_json)
    assert_no_forbidden_public_terms(
        [
            {
                "segment_id": segment.segment_id,
                "segment_name": segment.segment_name,
                "rule_json": segment.rule_json,
                "profile_json": segment.profile_json,
                "content_brief_json": segment.content_brief_json,
                "data_evidence_json": segment.data_evidence_json,
                "segment_vector_id": segment.segment_vector_id,
                "priority": segment.priority,
                "status": segment.status,
            }
            for segment in saved_segments
        ]
    )
    assert_no_forbidden_public_terms(
        [
            {
                "suggestion_source": suggestion.suggestion_source,
                "score_json": suggestion.score_json,
                "reason_json": suggestion.reason_json,
                "metadata_json": suggestion.metadata_json,
            }
            for suggestion in saved_suggestions
        ]
    )
