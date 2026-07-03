from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Mapping, Sequence

import pytest

from app.analysis.repositories import (
    HotelMarketingProfileRecord,
    PromotionAnalysisWrite,
    PromotionRecord,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
)
from app.analysis.schemas import AnalysisRequest
from app.analysis.service import (
    PromotionAnalysisService,
    PromotionNotFoundError,
    SegmentSelectionError,
)
from app.analysis.vector_service import (
    SegmentVectorBuildRequest,
    SegmentVectorBuildResult,
)


@dataclass
class SavedAnalysis:
    analysis: PromotionAnalysisWrite | None = None
    target_segments: list[PromotionTargetSegmentWrite] | None = None


class FakePromotionRepository:
    def __init__(self, promotion: PromotionRecord | None) -> None:
        self.promotion = promotion
        self.calls: list[Mapping[str, str]] = []

    def get_for_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionRecord | None:
        self.calls.append(
            {
                "project_id": project_id,
                "campaign_id": campaign_id,
                "promotion_id": promotion_id,
            }
        )
        return self.promotion


class FakeSegmentDefinitionRepository:
    def __init__(self, segments: list[SegmentDefinitionRecord]) -> None:
        self.segments = segments
        self.calls: list[Mapping[str, Any]] = []
        self.saved_ai_suggested: list[SegmentDefinitionRecord] = []

    def list_active(
        self,
        *,
        project_id: str,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        self.calls.append({"project_id": project_id, "sources": sources})
        return self.segments

    def save_ai_suggested(
        self,
        segments: Sequence[SegmentDefinitionRecord],
    ) -> None:
        self.saved_ai_suggested.extend(segments)


class FakeHotelProfileRepository:
    def __init__(self, profiles: list[HotelMarketingProfileRecord]) -> None:
        self.profiles = profiles
        self.calls: list[Mapping[str, str]] = []

    def list_marketing_profiles(
        self,
        *,
        project_id: str,
    ) -> list[HotelMarketingProfileRecord]:
        self.calls.append({"project_id": project_id})
        return self.profiles


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
            source="fixture",
        )


class FakeSegmentSuggester:
    def __init__(self, segments: list[SegmentDefinitionRecord]) -> None:
        self.segments = segments
        self.calls: list[PromotionRecord] = []

    def suggest_segments(
        self,
        *,
        promotion: PromotionRecord,
    ) -> list[SegmentDefinitionRecord]:
        self.calls.append(promotion)
        return self.segments


def promotion_record(
    *,
    channel: str = "onsite_banner",
    goal_metric: str = "booking_conversion_rate",
    min_sample_size: int = 1000,
) -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id=f"promo_{channel}_001",
        channel=channel,
        goal_metric=goal_metric,
        goal_target_value=Decimal("0.030000"),
        goal_basis="all_segments",
        min_sample_size=min_sample_size,
        landing_url="https://demo-stay.example.com/summer",
        message_brief="Drive summer hotel booking.",
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
        natural_language_query=f"{segment_id} hotel audience",
        generated_sql=None,
        rule_json=rule_json or {"segment_id": segment_id},
        profile_json={"primary_segment": segment_id},
        sample_size=sample_size,
        total_eligible_user_count=74200,
        sample_ratio=sample_ratio,
        status="active",
    )


def profile_record(
    profile_name: str,
    *,
    event_count: int = 2000,
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


def analysis_request(
    *,
    promotion_id: str,
    operator_instruction: str | None = None,
) -> AnalysisRequest:
    return AnalysisRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id=promotion_id,
        operator_instruction=operator_instruction,
    )


def build_service(
    *,
    promotion: PromotionRecord | None,
    segments: list[SegmentDefinitionRecord],
    profiles: list[HotelMarketingProfileRecord] | None = None,
    segment_vector_service: FakeSegmentVectorService | None = None,
    segment_suggester: FakeSegmentSuggester | None = None,
) -> tuple[
    PromotionAnalysisService,
    FakePromotionAnalysisRepository,
    FakeSegmentDefinitionRepository,
]:
    analysis_repository = FakePromotionAnalysisRepository()
    segment_definition_repository = FakeSegmentDefinitionRepository(segments)
    service = PromotionAnalysisService(
        promotion_repository=FakePromotionRepository(promotion),
        segment_definition_repository=segment_definition_repository,
        hotel_profile_repository=FakeHotelProfileRepository(profiles or []),
        promotion_analysis_repository=analysis_repository,
        segment_vector_service=segment_vector_service,
        segment_suggester=segment_suggester,
    )
    return service, analysis_repository, segment_definition_repository


def default_segments() -> list[SegmentDefinitionRecord]:
    return [
        segment_record("seg_mobile_user"),
        segment_record("seg_family_trip"),
        segment_record("seg_near_checkin"),
        segment_record("seg_existing_all"),
        segment_record("seg_repeat_hotel_no_booking"),
        segment_record("seg_long_stay"),
    ]


def segment_ids(
    target_segments: Sequence[PromotionTargetSegmentWrite],
) -> list[str]:
    return [segment.segment_id for segment in target_segments]


def test_service_analyzes_email_promotion_and_persists_four_segments() -> None:
    promotion = promotion_record(channel="email")
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        profiles=[profile_record("seg_mobile_user", event_count=5000)],
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert result.analysis.status == "completed"
    assert segment_ids(result.target_segments) == [
        "seg_mobile_user",
        "seg_family_trip",
        "seg_near_checkin",
        "seg_existing_all",
    ]
    assert analysis_repository.saved.analysis == result.analysis
    assert analysis_repository.saved.target_segments == result.target_segments
    assert result.analysis.profile_summary_json["selected_segment_count"] == 4
    assert result.target_segments[0].profile_json["hotel_profile"]["event_count"] == 5000


def test_service_prioritizes_related_custom_segment_for_onsite_banner() -> None:
    promotion = promotion_record(channel="onsite_banner")
    segments = [
        segment_record(
            "seg_repeat_hotel_no_booking",
            source="custom_chatkit",
            sample_size=1342,
        ),
        segment_record("seg_family_trip"),
        segment_record("seg_mobile_user"),
        segment_record("seg_near_checkin"),
        segment_record("seg_existing_all"),
    ]
    service, _, _ = build_service(promotion=promotion, segments=segments)

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_repeat_hotel_no_booking",
        "seg_family_trip",
        "seg_mobile_user",
        "seg_near_checkin",
    ]
    assert result.target_segments[0].priority == "high"
    assert result.target_segments[0].status == "planned"


def test_service_applies_sms_default_segment_order() -> None:
    promotion = promotion_record(channel="sms")
    service, _, _ = build_service(promotion=promotion, segments=default_segments())

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_near_checkin",
        "seg_mobile_user",
        "seg_family_trip",
        "seg_existing_all",
    ]


def test_service_populates_segment_vector_ids_when_vector_service_is_configured() -> None:
    promotion = promotion_record(channel="onsite_banner")
    vector_service = FakeSegmentVectorService()
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[
            segment_record(
                "seg_mobile_user",
                rule_json={
                    "candidate_user_ids": ["user_001", "user_002"],
                },
            ),
        ],
        segment_vector_service=vector_service,
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert result.target_segments[0].segment_vector_id == "segvec_seg_mobile_user_v1"
    assert analysis_repository.events == ["analysis", "target_segments"]
    assert vector_service.calls == [
        SegmentVectorBuildRequest(
            project_id="hotel-client-a",
            promotion_id=promotion.promotion_id,
            analysis_id=f"analysis_{promotion.promotion_id}",
            segment_id="seg_mobile_user",
            candidate_user_ids=["user_001", "user_002"],
        )
    ]


def test_service_prioritizes_ai_suggested_cluster_segments() -> None:
    promotion = promotion_record(channel="onsite_banner")
    ai_segment = segment_record(
        "seg_ai_cluster_promo_onsite_banner_001_1_abcdef1234",
        source="ai_suggested",
        sample_size=1800,
        rule_json={
            "source": "user_vector_clustering",
            "candidate_user_ids": ["user_101", "user_102"],
        },
    )
    ai_segment = replace(
        ai_segment,
        profile_json={
            "primary_segment": ai_segment.segment_id,
            "source": "user_vector_clustering",
            "cluster_score": 0.99,
        },
    )
    vector_service = FakeSegmentVectorService()
    suggester = FakeSegmentSuggester([ai_segment])
    service, analysis_repository, segment_definition_repository = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_vector_service=vector_service,
        segment_suggester=suggester,
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    expected_segment_ids = [
        ai_segment.segment_id,
        "seg_family_trip",
        "seg_mobile_user",
        "seg_repeat_hotel_no_booking",
    ]
    assert segment_ids(result.target_segments) == expected_segment_ids
    assert segment_definition_repository.saved_ai_suggested == [ai_segment]
    assert suggester.calls == [promotion]
    assert analysis_repository.events == ["analysis", "target_segments"]
    assert result.analysis.output_json == {
        "selected_segment_ids": expected_segment_ids,
        "target_segment_count": 4,
    }
    assert result.analysis.profile_summary_json["candidate_segment_count"] == 7
    assert result.analysis.input_snapshot_json["available_segment_definitions"][6] == {
        "segment_id": ai_segment.segment_id,
        "segment_name": ai_segment.segment_name,
        "source": "ai_suggested",
        "sample_size": 1800,
        "total_eligible_user_count": 74200,
        "sample_ratio": "0.020000",
        "status": "active",
    }
    assert result.target_segments[0].data_evidence_json["source"] == "ai_suggested"
    assert result.target_segments[0].profile_json["cluster_score"] == 0.99
    assert vector_service.calls[0] == SegmentVectorBuildRequest(
        project_id="hotel-client-a",
        promotion_id=promotion.promotion_id,
        analysis_id=f"analysis_{promotion.promotion_id}",
        segment_id=ai_segment.segment_id,
        candidate_user_ids=["user_101", "user_102"],
    )


def test_service_falls_back_to_default_segments_when_no_ai_suggestions_exist() -> None:
    promotion = promotion_record(channel="onsite_banner")
    suggester = FakeSegmentSuggester([])
    service, _, segment_definition_repository = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=suggester,
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_family_trip",
        "seg_mobile_user",
        "seg_repeat_hotel_no_booking",
        "seg_near_checkin",
    ]
    assert segment_definition_repository.saved_ai_suggested == []
    assert suggester.calls == [promotion]


def test_service_reflects_operator_instruction_in_content_brief() -> None:
    promotion = promotion_record(channel="onsite_banner")
    service, _, _ = build_service(promotion=promotion, segments=default_segments())

    result = service.analyze(
        analysis_request(
            promotion_id=promotion.promotion_id,
            operator_instruction="Emphasize breakfast and same-day availability.",
        ),
    )

    assert result.analysis.operator_instruction == (
        "Emphasize breakfast and same-day availability."
    )
    assert result.target_segments[0].content_brief_json["operator_instruction"] == (
        "Emphasize breakfast and same-day availability."
    )


def test_service_marks_small_segments_as_low_priority() -> None:
    promotion = promotion_record(channel="email", min_sample_size=1000)
    service, _, _ = build_service(
        promotion=promotion,
        segments=[
            segment_record("seg_mobile_user", sample_size=250),
            segment_record("seg_family_trip", sample_size=2000),
            segment_record("seg_near_checkin", sample_size=2000),
            segment_record("seg_existing_all", sample_size=2000),
        ],
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert result.target_segments[0].segment_id == "seg_mobile_user"
    assert result.target_segments[0].priority == "low"


def test_service_raises_when_promotion_is_missing() -> None:
    service, _, _ = build_service(
        promotion=None,
        segments=default_segments(),
    )

    with pytest.raises(PromotionNotFoundError):
        service.analyze(analysis_request(promotion_id="promo_missing"))


def test_service_raises_when_no_candidate_matches() -> None:
    promotion = promotion_record(channel="email")
    service, _, _ = build_service(
        promotion=promotion,
        segments=[],
    )

    with pytest.raises(SegmentSelectionError):
        service.analyze(analysis_request(promotion_id=promotion.promotion_id))
