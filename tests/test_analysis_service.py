from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Mapping, Sequence

import pytest

from app.analysis.repositories import (
    BookingTrainingRecord,
    HotelMarketingProfileRecord,
    PromotionAnalysisWrite,
    PromotionRecord,
    PromotionSegmentSuggestionWrite,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
)
from app.analysis.schemas import AnalysisRequest
from app.analysis.service import (
    NextLoopFocusAnalysisRequest,
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
    segment_suggestions: list[PromotionSegmentSuggestionWrite] | None = None


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
        campaign_id: str | None = None,
        promotion_id: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        self.calls.append(
            {
                "project_id": project_id,
                "campaign_id": campaign_id,
                "promotion_id": promotion_id,
                "sources": sources,
            }
        )
        return self.segments

    def save_ai_suggested(
        self,
        segments: Sequence[SegmentDefinitionRecord],
    ) -> None:
        self.saved_ai_suggested.extend(segments)


class FakeHotelProfileRepository:
    def __init__(
        self,
        profiles: list[HotelMarketingProfileRecord],
        *,
        user_profile_summaries: Mapping[str, HotelMarketingProfileRecord] | None = None,
        booking_training_records: list[BookingTrainingRecord] | None = None,
    ) -> None:
        self.profiles = profiles
        self.user_profile_summaries = user_profile_summaries or {}
        self.booking_training_records = booking_training_records or []
        self.calls: list[Mapping[str, str]] = []
        self.user_profile_calls: list[Mapping[str, Any]] = []
        self.training_calls: list[Mapping[str, int]] = []

    def list_marketing_profiles(
        self,
        *,
        project_id: str,
    ) -> list[HotelMarketingProfileRecord]:
        self.calls.append({"project_id": project_id})
        return self.profiles

    def summarize_user_ids(
        self,
        *,
        project_id: str,
        profile_name: str,
        user_ids: Sequence[str],
    ) -> HotelMarketingProfileRecord | None:
        self.user_profile_calls.append(
            {
                "project_id": project_id,
                "profile_name": profile_name,
                "user_ids": list(user_ids),
            }
        )
        return self.user_profile_summaries.get(profile_name)

    def list_booking_training_records(
        self,
        *,
        limit: int = 500,
    ) -> list[BookingTrainingRecord]:
        self.training_calls.append({"limit": limit})
        return self.booking_training_records


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
    user_profile_summaries: Mapping[str, HotelMarketingProfileRecord] | None = None,
    booking_training_records: list[BookingTrainingRecord] | None = None,
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
        hotel_profile_repository=FakeHotelProfileRepository(
            profiles or [],
            user_profile_summaries=user_profile_summaries,
            booking_training_records=booking_training_records,
        ),
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


def suggestion_segment_ids(
    suggestions: Sequence[PromotionSegmentSuggestionWrite],
) -> list[str]:
    return [suggestion.segment_id for suggestion in suggestions]


def test_service_analyzes_email_promotion_and_persists_four_suggestions() -> None:
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
    assert re.fullmatch(
        rf"analysis_{promotion.promotion_id}_run_[0-9a-f]{{8}}",
        result.analysis.analysis_id,
    )
    assert segment_ids(result.target_segments) == [
        "seg_mobile_user",
        "seg_family_trip",
        "seg_near_checkin",
        "seg_existing_all",
    ]
    assert analysis_repository.saved.analysis == result.analysis
    assert analysis_repository.saved.segment_suggestions == result.segment_suggestions
    assert suggestion_segment_ids(result.segment_suggestions) == segment_ids(
        result.target_segments
    )
    assert result.analysis.profile_summary_json["selected_segment_count"] == 4
    assert result.target_segments[0].profile_json["hotel_profile"]["event_count"] == 5000


def test_service_creates_new_analysis_id_for_repeated_ai_recommendations() -> None:
    promotion = promotion_record(channel="email")
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
    )

    first = service.analyze(analysis_request(promotion_id=promotion.promotion_id))
    second = service.analyze(analysis_request(promotion_id=promotion.promotion_id))

    assert first.analysis.analysis_id != second.analysis.analysis_id
    assert re.fullmatch(
        rf"analysis_{promotion.promotion_id}_run_[0-9a-f]{{8}}",
        first.analysis.analysis_id,
    )
    assert re.fullmatch(
        rf"analysis_{promotion.promotion_id}_run_[0-9a-f]{{8}}",
        second.analysis.analysis_id,
    )
    assert {
        suggestion.analysis_id for suggestion in first.segment_suggestions
    } == {first.analysis.analysis_id}
    assert {
        suggestion.analysis_id for suggestion in second.segment_suggestions
    } == {second.analysis.analysis_id}


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


def test_service_analyzes_focus_segment_ids_only() -> None:
    promotion = promotion_record(channel="onsite_banner")
    vector_service = FakeSegmentVectorService()
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_vector_service=vector_service,
    )

    result = service.analyze_focus(
        NextLoopFocusAnalysisRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id=promotion.promotion_id,
            focus_segment_ids=["seg_near_checkin"],
            loop_count=2,
            source_promotion_run_id="prun_banner_001_loop_1",
            source_failed_ad_experiment_ids=["adexp_near_checkin_001"],
            operator_instruction="Stress breakfast.",
        ),
    )

    assert segment_ids(result.target_segments) == ["seg_near_checkin"]
    assert analysis_repository.saved.target_segments == result.target_segments
    assert analysis_repository.events == [
        "analysis",
        "target_segments",
        "segment_suggestions",
    ]
    saved_analysis = analysis_repository.saved.analysis
    assert saved_analysis is not None
    assert saved_analysis.focus_segment_ids_json == ["seg_near_checkin"]
    assert saved_analysis.input_snapshot_json["focus_segment_ids"] == [
        "seg_near_checkin"
    ]
    assert saved_analysis.operator_instruction == "Stress breakfast."
    assert saved_analysis.profile_summary_json["selection_mode"] == "focus"
    assert vector_service.calls == [
        SegmentVectorBuildRequest(
                project_id="hotel-client-a",
                promotion_id=promotion.promotion_id,
                analysis_id=result.analysis.analysis_id,
                segment_id="seg_near_checkin",
                candidate_user_ids=[],
            )
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
    assert analysis_repository.events == ["analysis", "segment_suggestions"]
    assert vector_service.calls == [
        SegmentVectorBuildRequest(
            project_id="hotel-client-a",
            promotion_id=promotion.promotion_id,
            analysis_id=result.analysis.analysis_id,
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
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": ai_segment.segment_id,
            "source": "user_vector_clustering",
            "cluster_score": 0.99,
            "top_common_features": [
                "Booking conversion ready users",
                "Promotion-engaged hotel users",
                "Hotel page viewers",
            ],
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
    assert analysis_repository.events == ["analysis", "segment_suggestions"]
    assert result.analysis.output_json == {
        "selected_segment_ids": expected_segment_ids,
        "target_segment_count": 4,
    }
    assert result.analysis.profile_summary_json["candidate_segment_count"] == 7
    assert result.analysis.input_snapshot_json["available_segment_definitions"][6] == {
        "segment_id": ai_segment.segment_id,
        "campaign_id": promotion.campaign_id,
        "promotion_id": promotion.promotion_id,
        "segment_name": ai_segment.segment_name,
        "source": "ai_suggested",
        "sample_size": 1800,
        "total_eligible_user_count": 74200,
        "sample_ratio": "0.020000",
        "status": "active",
    }
    assert result.target_segments[0].data_evidence_json["source"] == "ai_suggested"
    assert result.target_segments[0].profile_json["cluster_score"] == 0.99
    assert result.segment_suggestions[0].suggestion_source == "ai_generated"
    assert result.segment_suggestions[0].status == "suggested"
    assert result.segment_suggestions[0].suggested_rank == 1
    assert result.segment_suggestions[0].reason_json["primary_signals"] == [
        "booking_conversion_ready",
        "promotion_engaged",
        "hotel_browsing",
    ]
    assert result.segment_suggestions[0].metadata_json["display_copy"] == {
        "title": "예약 가능성이 높은 프로모션 반응 고객",
        "audience_summary": "분석 대상 74200명 중 1800명 · 2%",
        "signal_chips": ["예약 가능성 높음", "프로모션 반응", "호텔 탐색"],
        "reason": "예약 전환 목표에 가까운 행동 패턴을 보인 고객군입니다.",
        "action_hint": "사이트 내 배너로 호텔 혜택을 노출하기 적합합니다.",
    }
    ai_report = result.segment_suggestions[0].metadata_json["ai_report"]
    assert ai_report["version"] == "dec-c8.segment-report.v1"
    assert ai_report["title"] == "예약 가능성이 높은 프로모션 반응 고객"
    assert ai_report["summary"]
    assert ai_report["why_recommended"]
    assert ai_report["evidence"]
    assert ai_report["action_hint"]
    assert all(
        forbidden not in str(ai_report)
        for forbidden in ("벡터", "군집", "클러스터", "centroid", "유사도")
    )
    assert vector_service.calls[0] == SegmentVectorBuildRequest(
        project_id="hotel-client-a",
        promotion_id=promotion.promotion_id,
        analysis_id=result.analysis.analysis_id,
        segment_id=ai_segment.segment_id,
        candidate_user_ids=["user_101", "user_102"],
    )


def test_service_uses_promotion_matched_features_for_ai_suggestion_copy() -> None:
    promotion = promotion_record(channel="email")
    ai_segment = replace(
        segment_record(
            "seg_ai_cluster_promo_email_001_1_intent",
            source="ai_suggested",
            sample_size=120,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_101", "user_102"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_email_001_1_intent",
            "source": "user_vector_clustering",
            "cluster_score": 0.7,
            "promotion_cluster_similarity": 0.92,
            "cluster_quality_score": 0.7,
            "sample_size_score": 0.5,
            "recommendation_score": 0.88,
            "score_components": {
                "promotion_cluster_similarity": 0.92,
                "cluster_quality": 0.7,
                "sample_size": 0.5,
                "final_score": 0.88,
            },
            "promotion_matched_features": [
                "Campaign redirect users",
                "Hotel market bucket 2 affinity users",
                "Free cancellation seekers",
            ],
            "top_common_features": ["Hotel search users"],
            "promotion_vector_basis": {
                "channel": "email",
                "goal_metric": "booking_conversion_rate",
            },
        },
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=FakeSegmentSuggester([ai_segment]),
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    first_suggestion = result.segment_suggestions[0]
    assert first_suggestion.segment_id == ai_segment.segment_id
    assert first_suggestion.reason_json["primary_signals"] == [
        "campaign_redirect",
        "hotel_market_affinity",
        "free_cancellation",
    ]
    assert first_suggestion.score_json["promotion_cluster_similarity"] == 0.92
    assert first_suggestion.score_json["recommendation_score"] == 0.88
    assert first_suggestion.metadata_json["promotion_matched_features"] == [
        "Campaign redirect users",
        "Hotel market bucket 2 affinity users",
        "Free cancellation seekers",
    ]
    assert first_suggestion.metadata_json["display_copy"] == {
        "title": "이메일 링크 반응 고객",
        "audience_summary": "분석 대상 74200명 중 120명 · 2%",
        "signal_chips": ["이메일 링크 클릭", "지역 선호", "무료 취소 선호"],
        "reason": "예약 전환 목표에 가까운 행동 패턴을 보인 고객군입니다.",
        "action_hint": "이메일 예약 혜택 메시지의 우선 타겟으로 적합합니다.",
    }


def test_service_ranks_ai_clusters_by_booking_propensity_model() -> None:
    promotion = promotion_record(channel="onsite_banner")
    high_cluster_low_booking = replace(
        segment_record(
            "seg_ai_cluster_promo_onsite_banner_001_1_highcluster",
            source="ai_suggested",
            sample_size=1800,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_low_001", "user_low_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_onsite_banner_001_1_highcluster",
            "source": "user_vector_clustering",
            "cluster_score": 0.99,
        },
    )
    lower_cluster_high_booking = replace(
        segment_record(
            "seg_ai_cluster_promo_onsite_banner_001_2_ml",
            source="ai_suggested",
            sample_size=1800,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_ml_001", "user_ml_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_onsite_banner_001_2_ml",
            "source": "user_vector_clustering",
            "cluster_score": 0.55,
        },
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=FakeSegmentSuggester(
            [high_cluster_low_booking, lower_cluster_high_booking]
        ),
        booking_training_records=[
            BookingTrainingRecord(
                is_mobile=0,
                is_package=0,
                stay_nights=1,
                days_until_checkin=50,
                event_count=100,
                booking_count=4,
            ),
            BookingTrainingRecord(
                is_mobile=1,
                is_package=1,
                stay_nights=4,
                days_until_checkin=3,
                event_count=100,
                booking_count=72,
            ),
        ],
        user_profile_summaries={
            high_cluster_low_booking.segment_id: HotelMarketingProfileRecord(
                project_id="hotel-client-a",
                profile_name=high_cluster_low_booking.segment_id,
                profile_json={
                    "event_count": 80,
                    "booking_count": 2,
                    "mobile_ratio": 0.0,
                    "package_ratio": 0.0,
                    "avg_stay_nights": 1.0,
                    "avg_days_until_checkin": 50.0,
                },
            ),
            lower_cluster_high_booking.segment_id: HotelMarketingProfileRecord(
                project_id="hotel-client-a",
                profile_name=lower_cluster_high_booking.segment_id,
                profile_json={
                    "event_count": 80,
                    "booking_count": 50,
                    "mobile_ratio": 1.0,
                    "package_ratio": 1.0,
                    "avg_stay_nights": 4.0,
                    "avg_days_until_checkin": 3.0,
                },
            ),
        },
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments)[:2] == [
        lower_cluster_high_booking.segment_id,
        high_cluster_low_booking.segment_id,
    ]
    first_score = result.segment_suggestions[0].score_json
    second_score = result.segment_suggestions[1].score_json
    assert first_score["cluster_score"] < second_score["cluster_score"]
    assert first_score["booking_propensity_score"] > second_score[
        "booking_propensity_score"
    ]
    assert first_score["booking_propensity_model"] == "booking_propensity_logistic_v1"
    assert first_score["booking_propensity_training_sample_count"] == 2
    assert result.segment_suggestions[0].reason_json["ml_model"] == (
        "booking_propensity_logistic_v1"
    )
    assert result.segment_suggestions[0].reason_json["ml_features"] == {
        "mobile_ratio": 1.0,
        "package_ratio": 1.0,
        "stay_nights_scaled": 0.285714,
        "near_checkin_score": 0.95,
    }


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


def test_service_skips_zero_size_default_segments() -> None:
    promotion = promotion_record(channel="email")
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[
            segment_record("seg_mobile_user", sample_size=0),
            segment_record("seg_family_trip", sample_size=0),
            segment_record("seg_near_checkin", sample_size=35),
            segment_record("seg_existing_all", sample_size=20),
        ],
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_near_checkin",
        "seg_existing_all",
    ]
    assert suggestion_segment_ids(result.segment_suggestions) == [
        "seg_near_checkin",
        "seg_existing_all",
    ]
    assert all(
        suggestion.score_json["estimated_size"] > 0
        for suggestion in analysis_repository.saved.segment_suggestions or []
    )


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
