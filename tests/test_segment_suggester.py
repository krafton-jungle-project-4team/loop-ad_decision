from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Mapping

import pytest

from app.analysis.audience_selection import fixed_ratio_audience_selection_policy
from app.analysis.raw_event_segments import (
    DeterministicPromotionIntentExtractor,
    OpenAIPromotionIntentExtractor,
    compile_raw_event_intent,
    generate_raw_event_segment_candidate_pool,
    generate_raw_event_segment_definitions,
)
from app.analysis.repositories import (
    PromotionRecord,
    RawEventUserSignalRecord,
    UserBehaviorVectorRecord,
)
from app.analysis.segment_performance import (
    SegmentPerformanceFeatures,
    build_segment_performance_predictor,
)
from app.analysis.segment_suggester import VectorClusterSegmentSuggester


class FakeUserBehaviorVectorRepository:
    def __init__(self, vectors: list[UserBehaviorVectorRecord]) -> None:
        self.vectors = vectors
        self.calls: list[Mapping[str, Any]] = []

    def list_recent(
        self,
        *,
        project_id: str,
        limit: int = 200,
        vector_version: str = "v1",
    ) -> list[UserBehaviorVectorRecord]:
        self.calls.append(
            {
                "project_id": project_id,
                "limit": limit,
                "vector_version": vector_version,
            }
        )
        return self.vectors

class FakeRawEventSignalRepository:
    def __init__(self, profiles: list[RawEventUserSignalRecord]) -> None:
        self.profiles = profiles
        self.calls: list[Mapping[str, Any]] = []

    def list_raw_event_user_signals(
        self,
        *,
        project_id: str,
        vector_version: str = "v1",
        destination_terms: list[str] | tuple[str, ...] = (),
        season_months: list[int] | tuple[int, ...] = (),
        limit: int = 1000,
    ) -> list[RawEventUserSignalRecord]:
        self.calls.append(
            {
                "project_id": project_id,
                "vector_version": vector_version,
                "destination_terms": tuple(destination_terms),
                "season_months": tuple(season_months),
                "limit": limit,
            }
        )
        return self.profiles


class CandidateTypePerformancePredictor:
    version = "test.goal-performance.v1"
    method = "test_candidate_type_rates"
    calibration_status = "calibrated"

    def __init__(self, rates: Mapping[str, float]) -> None:
        self.rates = rates

    def predict(self, features: SegmentPerformanceFeatures) -> float:
        return self.rates.get(features.candidate_type, 0.01)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "model_version": self.version,
            "method": self.method,
            "calibration_status": self.calibration_status,
            "outcome_days": 30,
        }


def promotion_record(
    *,
    promotion_id: str = "promo_banner_001",
    message_brief: str = "Drive hotel booking conversion for summer stays.",
    goal_metric: str = "booking_conversion_rate",
) -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id=promotion_id,
        channel="onsite_banner",
        goal_metric=goal_metric,
        goal_target_value=Decimal("0.030000"),
        goal_basis="all_segments",
        min_sample_size=2,
        landing_url="https://demo-stay.example.com/hotel-detail",
        message_brief=message_brief,
    )


def vector_values(index: int) -> list[float]:
    values = [0.0] * 64
    values[index] = 1.0
    return values


def user_vector(
    user_id: str,
    values: list[float],
    *,
    vector_dim: int = 64,
) -> UserBehaviorVectorRecord:
    return UserBehaviorVectorRecord(
        project_id="hotel-client-a",
        user_id=user_id,
        vector_dim=vector_dim,
        vector_values=values,
        vector_version="v1",
        source="batch_profile",
    )


def raw_signal(
    user_id: str,
    *,
    hotel_search_count: int = 0,
    hotel_detail_view_count: int = 0,
    promotion_impression_count: int = 0,
    promotion_click_count: int = 0,
    campaign_landing_count: int = 0,
    booking_start_count: int = 0,
    booking_complete_count: int = 0,
    deal_event_count: int = 0,
    free_cancellation_count: int = 0,
    breakfast_included_count: int = 0,
    price_event_count: int = 0,
    destination_match_count: int = 0,
    season_match_count: int = 0,
    destination_values: tuple[str, ...] = (),
    checkin_dates: tuple[str, ...] = (),
    hotel_market_values: tuple[str, ...] = (),
    age_group_values: tuple[str, ...] = (),
    gender_values: tuple[str, ...] = (),
) -> RawEventUserSignalRecord:
    event_count = max(
        1,
        hotel_search_count
        + hotel_detail_view_count
        + promotion_impression_count
        + promotion_click_count
        + campaign_landing_count
        + booking_start_count
        + booking_complete_count
        + deal_event_count
        + free_cancellation_count
        + breakfast_included_count
        + price_event_count,
    )
    return RawEventUserSignalRecord(
        project_id="hotel-client-a",
        user_id=user_id,
        event_count=event_count,
        hotel_search_count=hotel_search_count,
        hotel_click_count=0,
        hotel_detail_view_count=hotel_detail_view_count,
        promotion_impression_count=promotion_impression_count,
        promotion_click_count=promotion_click_count,
        campaign_redirect_click_count=0,
        campaign_landing_count=campaign_landing_count,
        booking_start_count=booking_start_count,
        booking_complete_count=booking_complete_count,
        booking_cancel_count=0,
        deal_event_count=deal_event_count,
        free_cancellation_count=free_cancellation_count,
        breakfast_included_count=breakfast_included_count,
        price_event_count=price_event_count,
        avg_price=0.0,
        destination_values=destination_values,
        checkin_dates=checkin_dates,
        hotel_market_values=hotel_market_values,
        hotel_cluster_values=(),
        age_group_values=age_group_values,
        gender_values=gender_values,
        preferred_category_values=(),
        destination_match_count=destination_match_count,
        season_match_count=season_match_count,
    )


def test_raw_event_suggester_creates_distinct_candidate_types() -> None:
    vector_reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("intent_001", vector_values(0)),
            user_vector("intent_002", vector_values(0)),
            user_vector("funnel_001", vector_values(1)),
            user_vector("funnel_002", vector_values(1)),
            user_vector("promo_001", vector_values(5)),
            user_vector("promo_002", vector_values(5)),
            user_vector("benefit_001", vector_values(61)),
            user_vector("benefit_002", vector_values(61)),
        ]
    )
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "intent_001",
                hotel_search_count=2,
                hotel_detail_view_count=2,
                destination_match_count=1,
                season_match_count=1,
                destination_values=("제주 호텔",),
                checkin_dates=("2026-07-10",),
            ),
            raw_signal(
                "intent_002",
                hotel_search_count=1,
                hotel_detail_view_count=2,
                destination_match_count=1,
                season_match_count=1,
                destination_values=("jeju resort",),
                checkin_dates=("2026-08-11",),
            ),
            raw_signal(
                "funnel_001",
                hotel_detail_view_count=3,
                booking_start_count=1,
                destination_match_count=1,
            ),
            raw_signal(
                "funnel_002",
                hotel_detail_view_count=2,
                booking_start_count=1,
                destination_match_count=1,
            ),
            raw_signal(
                "promo_001",
                promotion_impression_count=3,
                promotion_click_count=1,
                campaign_landing_count=1,
            ),
            raw_signal(
                "promo_002",
                promotion_impression_count=4,
                promotion_click_count=1,
                campaign_landing_count=1,
            ),
            raw_signal(
                "benefit_001",
                hotel_search_count=1,
                deal_event_count=1,
                destination_match_count=1,
            ),
            raw_signal(
                "benefit_002",
                hotel_search_count=1,
                free_cancellation_count=1,
                price_event_count=1,
                destination_match_count=1,
            ),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=3,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여름 제주 숙소 할인 프로모션으로 랜딩 유입을 높인다.",
        )
    )

    assert vector_reader.calls == []
    assert raw_reader.calls[0]["vector_version"] == "v1"
    assert raw_reader.calls[0]["destination_terms"] == ("jeju", "제주")
    assert raw_reader.calls[0]["season_months"] == (6, 7, 8)
    assert len(segments) == 3
    candidate_types = [
        segment.rule_json["candidate_type"]
        for segment in segments
    ]
    assert len(set(candidate_types)) == 3
    assert all(
        re.fullmatch(
            r"seg_ai_raw_promo_banner_001_[a-z_]+_[0-9a-f]{10}",
            segment.segment_id,
        )
        for segment in segments
    )
    assert all(segment.rule_json["source"] == "raw_event_intent" for segment in segments)
    assert all(segment.profile_json["source"] == "raw_event_intent" for segment in segments)
    assert all("strategy_role" in segment.profile_json for segment in segments)
    assert all("rank_role" not in segment.profile_json for segment in segments)
    assert all("display_copy" in segment.profile_json for segment in segments)
    assert all("performance_estimate" in segment.profile_json for segment in segments)
    assert all(
        segment.profile_json["primary_signals"]
        == segment.rule_json["compiled_conditions"][:3]
        for segment in segments
    )
    assert all(
        segment.profile_json["display_copy"]["performance_estimate"]["label"]
        == "예상 예약 전환율"
        for segment in segments
    )
    assert all(
        "final_score" in segment.profile_json["score_components"]
        for segment in segments
    )
    assert all(
        segment.profile_json["score_components"]["weights"]
        ["promotion_condition_match"]
        == 0.10
        for segment in segments
    )
    assert all(
        segment.profile_json["score_components"]["weights"]
        ["expected_goal_performance"]
        == 0.85
        for segment in segments
    )
    assert all(
        "조건 일치" in segment.profile_json["display_copy"]["audience_summary"]
        for segment in segments
    )
    assert all(
        "상위" not in segment.profile_json["display_copy"]["audience_summary"]
        for segment in segments
    )


def test_raw_event_suggester_applies_segment_instruction_to_intent() -> None:
    vector_reader = FakeUserBehaviorVectorRepository(
        [user_vector("jeju_001", vector_values(0))]
    )
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "jeju_001",
                hotel_search_count=2,
                hotel_detail_view_count=1,
                destination_match_count=2,
                destination_values=("제주 호텔",),
            ),
            raw_signal(
                "jeju_booked_001",
                hotel_search_count=2,
                hotel_detail_view_count=1,
                booking_complete_count=1,
                destination_match_count=2,
                destination_values=("제주 호텔",),
            ),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=10,
        vector_sample_limit=10,
        max_suggested_segments=3,
        min_cluster_size=1,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(message_brief="숙소 예약 프로모션"),
        segment_instruction="최근 제주 숙소를 검색하거나 호텔을 본 고객, 예약 완료 고객은 빼줘",
    )

    assert segments
    assert raw_reader.calls[0]["destination_terms"] == ("jeju", "제주")
    assert vector_reader.calls == []
    assert all(
        "jeju" in segment.profile_json["promotion_intent"]["destinations"]
        for segment in segments
    )
    assert all(
        segment.profile_json["promotion_intent"]["excluded_behaviors"]
        == ["booking_complete"]
        for segment in segments
    )
    assert all(
        segment.rule_json["candidate_user_ids"] == ["jeju_001"]
        for segment in segments
    )
    assert {
        segment.rule_json["candidate_type"] for segment in segments
    } == {"intent_matched"}


def test_raw_event_suggester_applies_demographic_instruction_to_profiles() -> None:
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "matched_001",
                hotel_search_count=2,
                age_group_values=("20대",),
                gender_values=("여성",),
            ),
            raw_signal(
                "wrong_age_001",
                hotel_search_count=2,
                age_group_values=("40대",),
                gender_values=("여성",),
            ),
            raw_signal(
                "missing_profile_001",
                hotel_search_count=2,
            ),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=FakeUserBehaviorVectorRepository([]),
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=10,
        vector_sample_limit=10,
        max_suggested_segments=3,
        min_cluster_size=1,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(message_brief="숙소 예약 프로모션"),
        segment_instruction="최근 호텔을 검색한 20대 여성 고객",
    )

    assert len(segments) == 1
    assert segments[0].rule_json["candidate_type"] == "intent_matched"
    assert segments[0].rule_json["candidate_user_ids"] == ["matched_001"]
    assert "profile_hint" in segments[0].rule_json["compiled_conditions"]


def test_openai_intent_extractor_keeps_segment_candidate_constraint() -> None:
    captured: dict[str, Any] = {}

    def transport(
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        captured["payload"] = payload
        return {
            "output_text": json.dumps(
                {
                    "summary": "제주 예약 이탈 고객",
                    "product": "hotel",
                    "season": [],
                    "destinations": ["jeju"],
                    "benefits": [],
                    "audience_hints": [],
                    "channel": "onsite_banner",
                    "goal_metric": "booking_conversion_rate",
                    "funnel_goal": "booking_complete",
                    "desired_behaviors": ["booking_start_without_complete"],
                    "excluded_behaviors": ["booking_complete"],
                    "explicit_conditions": ["제주", "예약 미완료"],
                    "requested_candidate_types": ["funnel_recovery"],
                },
                ensure_ascii=False,
            )
        }

    extractor = OpenAIPromotionIntentExtractor(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    intent = extractor.extract(
        promotion_record(message_brief="제주 숙소 예약 프로모션"),
        segment_instruction="예약을 시작했지만 완료하지 않은 고객만 찾아줘",
    )

    request_text = captured["payload"]["input"][1]["content"][0]["text"]
    assert "예약을 시작했지만 완료하지 않은 고객만 찾아줘" in request_text
    assert intent.requested_candidate_types == ("funnel_recovery",)
    assert intent.excluded_behaviors == ("booking_complete",)


def test_segment_instruction_does_not_fall_back_to_generic_vector_clusters() -> None:
    vector_reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("vector_001", vector_values(0)),
            user_vector("vector_002", vector_values(0)),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=FakeRawEventSignalRepository([]),
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=10,
        vector_sample_limit=10,
        max_suggested_segments=3,
        min_cluster_size=1,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(message_brief="숙소 예약 프로모션"),
        segment_instruction="최근 제주 숙소를 검색한 고객",
    )

    assert segments == []
    assert vector_reader.calls == []


def test_destination_candidates_exclude_users_without_target_interest() -> None:
    vector_reader = FakeUserBehaviorVectorRepository([])
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "intent_1",
                hotel_search_count=2,
                hotel_detail_view_count=1,
                destination_match_count=1,
                season_match_count=1,
            ),
            raw_signal(
                "intent_2",
                hotel_search_count=2,
                hotel_detail_view_count=1,
                destination_match_count=1,
                season_match_count=1,
            ),
            raw_signal(
                "target_repeat_1",
                hotel_search_count=4,
                destination_match_count=3,
                destination_values=("제주 제주",),
            ),
            raw_signal(
                "target_repeat_2",
                hotel_search_count=3,
                destination_match_count=2,
                destination_values=("jeju 제주 제주",),
            ),
            raw_signal(
                "general_1",
                hotel_search_count=4,
                destination_values=("부산", "서울"),
                hotel_market_values=("10", "20"),
            ),
            raw_signal(
                "general_2",
                hotel_search_count=3,
                destination_values=("강릉", "여수"),
                hotel_market_values=("30", "40"),
            ),
            raw_signal(
                "funnel_target_1",
                hotel_detail_view_count=2,
                booking_start_count=1,
                destination_match_count=1,
            ),
            raw_signal(
                "funnel_target_2",
                hotel_detail_view_count=2,
                booking_start_count=1,
                destination_match_count=1,
            ),
            raw_signal(
                "funnel_other_1",
                hotel_detail_view_count=2,
                booking_start_count=1,
            ),
            raw_signal(
                "funnel_other_2",
                hotel_detail_view_count=2,
                booking_start_count=1,
            ),
            raw_signal(
                "benefit_target_1",
                deal_event_count=2,
                destination_match_count=1,
            ),
            raw_signal(
                "benefit_target_2",
                price_event_count=2,
                destination_match_count=1,
            ),
            raw_signal("benefit_other_1", deal_event_count=2),
            raw_signal("benefit_other_2", price_event_count=2),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=30,
        vector_sample_limit=30,
        max_suggested_segments=6,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여름 제주 숙소 할인 프로모션으로 예약 전환을 높인다.",
        )
    )

    by_type = {
        segment.rule_json["candidate_type"]: segment
        for segment in segments
    }
    assert "target_destination_affinity" in by_type
    assert "general_destination_explorer" not in by_type
    assert set(
        by_type["target_destination_affinity"].rule_json["candidate_user_ids"]
    ) == {"target_repeat_1", "target_repeat_2"}
    assert set(by_type["funnel_recovery"].rule_json["candidate_user_ids"]) == {
        "funnel_target_1",
        "funnel_target_2",
    }
    assert set(
        by_type["benefit_value_seeker"].rule_json["candidate_user_ids"]
    ) == {"benefit_target_1", "benefit_target_2"}


def test_general_destination_explorer_is_available_without_target_destination() -> None:
    vector_reader = FakeUserBehaviorVectorRepository([])
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "general_1",
                hotel_search_count=4,
                destination_values=("부산", "서울"),
                hotel_market_values=("10", "20"),
            ),
            raw_signal(
                "general_2",
                hotel_search_count=3,
                destination_values=("강릉", "여수"),
                hotel_market_values=("30", "40"),
            ),
            raw_signal("single_destination_1", hotel_search_count=1),
            raw_signal("single_destination_2", hotel_search_count=1),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        max_suggested_segments=6,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여러 호텔과 여행지를 비교하는 고객을 위한 프로모션",
        )
    )

    general = next(
        segment
        for segment in segments
        if segment.rule_json["candidate_type"] == "general_destination_explorer"
    )
    assert set(general.rule_json["candidate_user_ids"]) == {
        "general_1",
        "general_2",
    }


def test_calibration_candidate_pool_keeps_overlapping_candidate_types() -> None:
    promotion = promotion_record(
        message_brief="여름 제주 숙소 할인 프로모션으로 예약 전환을 높인다.",
    )
    extractor = DeterministicPromotionIntentExtractor()
    intent = extractor.extract(promotion)
    profiles = [
        raw_signal(
            f"overlap_{index}",
            hotel_detail_view_count=2,
            booking_start_count=1,
            destination_match_count=1,
            season_match_count=1,
        )
        for index in range(2)
    ]
    compilation = compile_raw_event_intent(intent)

    ranked = generate_raw_event_segment_definitions(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        max_suggested_segments=6,
        min_sample_size=2,
    )
    calibration_pool = generate_raw_event_segment_candidate_pool(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        min_sample_size=2,
    )

    assert len(ranked) == 1
    assert {
        segment.rule_json["candidate_type"] for segment in calibration_pool
    } == {"intent_matched", "funnel_recovery"}
    ranked_ids = {
        segment.rule_json["candidate_type"]: segment.segment_id for segment in ranked
    }
    calibration_ids = {
        segment.rule_json["candidate_type"]: segment.segment_id
        for segment in calibration_pool
    }
    assert all(
        calibration_ids[candidate_type] == segment_id
        for candidate_type, segment_id in ranked_ids.items()
    )


def test_raw_event_suggester_uses_all_matching_users_until_ratio_is_backtested() -> None:
    vector_reader = FakeUserBehaviorVectorRepository([])
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                f"hotel_user_{index:03d}",
                hotel_detail_view_count=1,
            )
            for index in range(170)
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=200,
        vector_sample_limit=200,
        max_suggested_segments=1,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="호텔 예약 전환을 높이는 프로모션",
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    assert segment.sample_size == 170
    assert segment.total_eligible_user_count == 170
    assert segment.profile_json["signal_metrics"]["matching_profile_count"] == 170
    assert segment.profile_json["display_copy"]["audience_summary"] == (
        "분석 대상 170명 중 조건 일치 170명 · "
        "조건 일치자 전체를 추천 대상으로 사용"
    )
    assert segment.profile_json["display_copy"]["audience"] == {
        "total_eligible_user_count": 170,
        "matching_user_count": 170,
        "selected_user_count": 170,
        "selected_user_ratio": 1.0,
        "matching_user_ratio": 1.0,
        "selection_ratio_within_matching": 1.0,
        "selection_limited": False,
        "selection_basis": "candidate_condition_match",
        "selection_limit": None,
        "selected_user_role": "recommended_audience",
        "selection_policy": {
            "version": "dec.segment-audience-selection.v2",
            "method": "all_matching",
            "configured_ratio": 1.0,
            "applied_ratio": 1.0,
            "calibration_status": "pending_backtest",
            "artifact_hash": None,
            "fallback_reason": "artifact_missing",
        },
    }


def test_raw_event_suggester_applies_validated_ratio_to_behavior_order() -> None:
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                f"hotel_user_{index:03d}",
                hotel_search_count=index + 1,
                hotel_detail_view_count=1,
            )
            for index in range(10)
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=FakeUserBehaviorVectorRepository([]),
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        audience_selection_policy=fixed_ratio_audience_selection_policy(
            goal_metric="booking_conversion_rate",
            selected_ratio=0.4,
            minimum_selected_user_count=2,
            policy_version="test-selection-policy.v1",
        ),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=1,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="호텔 예약 전환을 높이는 프로모션",
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    assert segment.rule_json["candidate_user_ids"] == [
        "hotel_user_009",
        "hotel_user_008",
        "hotel_user_007",
        "hotel_user_006",
    ]
    assert segment.sample_size == 4
    audience = segment.profile_json["audience"]
    assert audience["matching_user_count"] == 10
    assert audience["selected_user_count"] == 4
    assert audience["selection_ratio_within_matching"] == 0.4
    assert audience["selection_limited"] is True
    assert audience["selection_policy"]["method"] == (
        "top_behavior_strength_ratio"
    )
    assert audience["selection_policy"]["configured_ratio"] == 0.4
    assert segment.profile_json["signal_metrics"]["profile_count"] == 4
    assert segment.profile_json["signal_metrics"]["matching_profile_count"] == 10


def test_raw_event_suggester_does_not_repeat_the_same_audience_across_ranks() -> None:
    vector_reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("user_001", vector_values(0)),
            user_vector("user_002", vector_values(1)),
        ]
    )
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                user_id,
                hotel_search_count=2,
                hotel_detail_view_count=2,
                promotion_impression_count=2,
                promotion_click_count=1,
                campaign_landing_count=1,
                booking_start_count=1,
                deal_event_count=1,
                destination_match_count=1,
                season_match_count=1,
                destination_values=("제주 호텔", "오키나와 호텔"),
                checkin_dates=("2026-07-10",),
            )
            for user_id in ("user_001", "user_002")
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=3,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여름 제주 숙소 할인 프로모션으로 예약 전환을 높인다.",
        )
    )

    assert len(segments) == 1
    assert segments[0].rule_json["candidate_user_ids"] == ["user_001", "user_002"]
    assert vector_reader.calls == []


def test_raw_event_suggester_labels_inflow_performance_estimate() -> None:
    vector_reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("promo_001", vector_values(5)),
            user_vector("promo_002", vector_values(5)),
        ]
    )
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "promo_001",
                hotel_search_count=2,
                promotion_impression_count=4,
                promotion_click_count=1,
                campaign_landing_count=1,
            ),
            raw_signal(
                "promo_002",
                hotel_search_count=1,
                promotion_impression_count=5,
                promotion_click_count=2,
                campaign_landing_count=1,
            ),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=1,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            goal_metric="inflow_rate",
            message_brief="여름 호텔 할인 프로모션 랜딩 유입을 높인다.",
        )
    )

    assert len(segments) == 1
    performance_estimate = segments[0].profile_json["performance_estimate"]
    assert performance_estimate["label"] == "예상 유입률"
    assert performance_estimate["metric"] == "inflow_rate"
    assert performance_estimate["basis_label"] == (
        "최근 클릭·랜딩 행동을 전체 고객 기준으로 보정한 추정치"
    )
    assert performance_estimate["calibration_status"] == (
        "historical_signal_estimate"
    )
    assert segments[0].profile_json["display_copy"]["performance_estimate"] == (
        performance_estimate
    )


def test_raw_event_suggester_selects_diverse_portfolio_without_rank_copy() -> None:
    profiles = [
        *[
            raw_signal(
                f"intent_{index}",
                hotel_search_count=2,
                hotel_detail_view_count=1,
            )
            for index in range(6)
        ],
        *[
            raw_signal(
                f"recovery_{index}",
                booking_start_count=1,
            )
            for index in range(6)
        ],
    ]
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=FakeUserBehaviorVectorRepository([]),
        raw_event_signal_repository=FakeRawEventSignalRepository(profiles),
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        performance_predictor=CandidateTypePerformancePredictor(
            {
                "intent_matched": 0.18,
                "funnel_recovery": 0.42,
            }
        ),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=2,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="숙소 예약 전환을 높이는 프로모션",
        )
    )

    assert [segment.rule_json["candidate_type"] for segment in segments] == [
        "funnel_recovery",
        "intent_matched",
    ]
    first_profile = segments[0].profile_json
    assert first_profile["score_components"]["weights"][
        "expected_goal_performance"
    ] == 0.70
    assert first_profile["score_components"]["primary_component"] == (
        "expected_goal_performance"
    )
    estimate = first_profile["performance_estimate"]
    assert estimate["label"] == "예상 예약 전환율"
    assert estimate["value"] == 0.42
    assert estimate["window_days"] == 30
    assert estimate["window_label"] == "향후 30일 내 프로모션 조건 일치 예약"
    assert estimate["confidence_label"] == "high"
    display_copy = first_profile["display_copy"]
    assert display_copy["audience"] == {
        "total_eligible_user_count": 12,
        "matching_user_count": 6,
        "selected_user_count": 6,
        "selected_user_ratio": 0.5,
        "matching_user_ratio": 0.5,
        "selection_ratio_within_matching": 1.0,
        "selection_limited": False,
        "selection_basis": "candidate_condition_match",
        "selection_limit": None,
        "selected_user_role": "recommended_audience",
        "selection_policy": {
            "version": "dec.segment-audience-selection.v2",
            "method": "all_matching",
            "configured_ratio": 1.0,
            "applied_ratio": 1.0,
            "calibration_status": "pending_backtest",
            "artifact_hash": None,
            "fallback_reason": "artifact_missing",
        },
    }
    assert display_copy["strategy_role"] == "예약 이탈 회수형"
    assert display_copy["strength_summary"]
    assert display_copy["tradeoff_summary"]
    assert "recommendation_rank" not in display_copy
    assert "rank_comparison" not in display_copy
    assert "difference_summary" not in display_copy
    assert first_profile["portfolio_position"] == 1
    assert first_profile["selection_basis"]["method"] == (
        "diversified_candidate_portfolio"
    )


def test_raw_event_suggester_keeps_tier_as_candidate_context_not_rank() -> None:
    profiles = [
        *[
            raw_signal(
                f"destination_{index}",
                hotel_search_count=3,
                hotel_detail_view_count=2,
                destination_match_count=3,
                destination_values=("jeju",),
            )
            for index in range(4)
        ],
        *[
            raw_signal(
                f"responsive_{index}",
                promotion_impression_count=3,
                promotion_click_count=1,
                campaign_landing_count=1,
            )
            for index in range(40)
        ],
    ]
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=FakeUserBehaviorVectorRepository([]),
        raw_event_signal_repository=FakeRawEventSignalRepository(profiles),
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        performance_predictor=CandidateTypePerformancePredictor(
            {
                "target_destination_affinity": 0.09,
                "promotion_responsive": 0.05,
            }
        ),
        vector_pool_limit=100,
        vector_sample_limit=100,
        max_suggested_segments=2,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여름 제주 숙소 랜딩 유입을 높이는 프로모션",
            goal_metric="inflow_rate",
        )
    )

    assert [segment.rule_json["candidate_type"] for segment in segments] == [
        "promotion_responsive",
        "target_destination_affinity",
    ]
    primary_profile = segments[0].profile_json
    small_profile = segments[1].profile_json
    assert primary_profile["recommendation_tier"] == "primary"
    assert primary_profile["rank_eligible"] is True
    assert primary_profile["portfolio_position"] == 1
    assert "recommendation_rank" not in primary_profile
    assert primary_profile["performance_estimate"]["expected_count"] == pytest.approx(
        primary_profile["performance_estimate"]["value"] * 40
    )
    assert primary_profile["performance_estimate"]["expected_count_label"] == (
        "예상 유입 인원"
    )
    assert small_profile["recommendation_tier"] == "small_high_intent"
    assert small_profile["rank_eligible"] is False
    assert small_profile["portfolio_position"] == 2
    assert "recommendation_rank" not in small_profile
    assert small_profile["performance_estimate"]["expected_count"] == pytest.approx(
        small_profile["performance_estimate"]["value"] * 4
    )
    assert "표본 신뢰도" in small_profile["recommendation_tier_reason"]
    assert "rank_comparison" not in small_profile["display_copy"]
    assert small_profile["display_copy"]["tradeoff_summary"]


def test_raw_event_suggester_uses_destination_context_for_expected_conversion_rate() -> None:
    vector_reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("booking_001", vector_values(8)),
            user_vector("booking_002", vector_values(8)),
        ]
    )
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "booking_001",
                hotel_detail_view_count=100,
                booking_start_count=20,
                booking_complete_count=10,
                destination_match_count=1,
                season_match_count=1,
            ),
            raw_signal(
                "booking_002",
                hotel_detail_view_count=80,
                booking_start_count=15,
                booking_complete_count=8,
                destination_match_count=1,
                season_match_count=1,
            ),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=1,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여름 제주 숙소 예약 전환을 높인다.",
        )
    )

    performance_estimate = segments[0].profile_json["performance_estimate"]
    assert performance_estimate["label"] == "예상 예약 전환율"
    assert performance_estimate["value"] < 1.0
    assert performance_estimate["formatted"] != "100.0%"
    assert performance_estimate["observed_value"] == 1.0
    assert performance_estimate["method"] == "destination_context_heuristic"
    assert performance_estimate["calibration_status"] == "uncalibrated_fallback"
    assert segments[0].profile_json["performance_features"][
        "destination_match_user_rate"
    ] == 1.0
    score_components = segments[0].profile_json["score_components"]
    assert score_components["predicted_goal_rate"] < 1.0
    assert score_components["expected_goal_performance"] == 1.0


def test_raw_event_suggester_adjusts_small_out_of_distribution_prediction() -> None:
    user_ids = [f"extreme_{index}" for index in range(4)]
    vector_reader = FakeUserBehaviorVectorRepository(
        [user_vector(user_id, vector_values(8)) for user_id in user_ids]
    )
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                user_id,
                hotel_search_count=20,
                hotel_detail_view_count=20,
                promotion_impression_count=5,
                promotion_click_count=2,
                campaign_landing_count=3,
                booking_start_count=5,
                booking_complete_count=4,
                deal_event_count=10,
                destination_match_count=30,
                season_match_count=1,
                destination_values=("jeju",),
                checkin_dates=("2026-07-15",),
            )
            for user_id in user_ids
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        performance_predictor=build_segment_performance_predictor(),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=1,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여름 제주 숙소 예약 전환을 높인다.",
        )
    )

    estimate = segments[0].profile_json["performance_estimate"]
    adjustment = estimate["prediction_adjustment"]
    assert adjustment["raw_model_value"] > adjustment["adjusted_value"]
    assert adjustment["distribution_guarded_value"] > (
        adjustment["adjusted_value"]
    )
    assert adjustment["candidate_sample_size"] == 4
    assert adjustment["out_of_distribution_feature_count"] > 0
    assert estimate["value"] == adjustment["adjusted_value"]
    assert estimate["value"] < 0.2
    assert estimate["confidence_label"] == "low"
    assert "학습 범위" not in estimate["confidence_reason"]
    assert "분포" not in estimate["confidence_reason"]
    assert segments[0].profile_json["recommendation_tier"] == "small_high_intent"
    assert segments[0].profile_json["minimum_primary_sample_size"] == 30
    assert segments[0].profile_json["portfolio_position"] == 1
    assert "recommendation_rank" not in segments[0].profile_json


def test_supported_candidate_keeps_internal_distribution_diagnostics_out_of_user_copy() -> None:
    promotion = promotion_record(
        message_brief="여름 제주 숙소 예약 전환을 높인다.",
    )
    intent = DeterministicPromotionIntentExtractor().extract(promotion)
    compilation = compile_raw_event_intent(intent)
    profiles = [
        raw_signal(
            f"supported_extreme_{index}",
            hotel_search_count=20,
            hotel_detail_view_count=20,
            booking_start_count=5,
            booking_complete_count=4,
            deal_event_count=10,
            destination_match_count=30,
            season_match_count=1,
            destination_values=("jeju",),
            checkin_dates=("2026-07-15",),
        )
        for index in range(40)
    ]

    segments = generate_raw_event_segment_candidate_pool(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        min_sample_size=2,
        performance_predictor=build_segment_performance_predictor(),
    )

    destination_segment = next(
        segment
        for segment in segments
        if segment.rule_json["candidate_type"] == "target_destination_affinity"
    )
    estimate = destination_segment.profile_json["performance_estimate"]
    adjustment = estimate["prediction_adjustment"]
    assert adjustment["candidate_sample_size"] == 40
    assert adjustment["out_of_distribution_feature_count"] > 0
    assert estimate["confidence_label"] == "high"
    assert "학습 범위" not in estimate["confidence_reason"]
    assert "분포" not in estimate["confidence_reason"]


def test_booking_model_excludes_candidate_type_without_training_examples() -> None:
    promotion = promotion_record(
        message_brief="여름 제주 숙소 예약 전환을 높인다.",
    )
    intent = DeterministicPromotionIntentExtractor().extract(promotion)
    compilation = compile_raw_event_intent(intent)
    profiles = [
        raw_signal(
            f"responsive_{index}",
            promotion_impression_count=5,
            promotion_click_count=2,
            campaign_landing_count=1,
        )
        for index in range(160)
    ]

    segments = generate_raw_event_segment_candidate_pool(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        min_sample_size=2,
        performance_predictor=build_segment_performance_predictor(),
    )

    assert segments == []


def test_booking_fallback_blocks_responsive_candidate_but_calibration_can_collect_it() -> None:
    promotion = promotion_record(
        message_brief="여름 제주 숙소 예약 전환을 높인다.",
    )
    intent = DeterministicPromotionIntentExtractor().extract(promotion)
    compilation = compile_raw_event_intent(intent)
    profiles = [
        raw_signal(
            f"responsive_fallback_{index}",
            promotion_impression_count=5,
            promotion_click_count=2,
            campaign_landing_count=1,
        )
        for index in range(20)
    ]

    ranked = generate_raw_event_segment_definitions(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        max_suggested_segments=3,
        min_sample_size=2,
    )
    calibration_pool = generate_raw_event_segment_candidate_pool(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        min_sample_size=2,
        enforce_prediction_support=False,
    )

    assert ranked == []
    assert [
        segment.rule_json["candidate_type"] for segment in calibration_pool
    ] == ["promotion_responsive"]


def test_inflow_metric_keeps_promotion_responsive_candidate() -> None:
    promotion = promotion_record(
        message_brief="여름 제주 숙소 랜딩 유입을 높인다.",
        goal_metric="inflow_rate",
    )
    intent = DeterministicPromotionIntentExtractor().extract(promotion)
    compilation = compile_raw_event_intent(intent)
    profiles = [
        raw_signal(
            f"responsive_{index}",
            promotion_impression_count=5,
            promotion_click_count=2,
            campaign_landing_count=1,
        )
        for index in range(20)
    ]

    segments = generate_raw_event_segment_candidate_pool(
        promotion=promotion,
        intent=intent,
        compilation=compilation,
        profiles=profiles,
        min_sample_size=2,
        performance_predictor=build_segment_performance_predictor(),
    )

    responsive_segment = next(
        segment
        for segment in segments
        if segment.rule_json["candidate_type"] == "promotion_responsive"
    )
    estimate = responsive_segment.profile_json["performance_estimate"]
    assert estimate["metric"] == "inflow_rate"
    assert estimate["calibration_status"] == "historical_signal_estimate"


def test_raw_event_suggester_requests_vector_window_signals() -> None:
    vector_reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("user_001", vector_values(0)),
            user_vector("user_002", vector_values(0)),
        ]
    )
    raw_reader = FakeRawEventSignalRepository(
        [
            raw_signal(
                "user_001",
                hotel_search_count=2,
                hotel_detail_view_count=2,
                destination_match_count=1,
                season_match_count=1,
            ),
            raw_signal(
                "user_002",
                hotel_search_count=2,
                hotel_detail_view_count=2,
                destination_match_count=1,
                season_match_count=1,
            ),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=vector_reader,
        raw_event_signal_repository=raw_reader,
        promotion_intent_extractor=DeterministicPromotionIntentExtractor(),
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=1,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="여름 제주 숙소 예약 전환을 높인다.",
        )
    )

    assert len(segments) == 1
    assert segments[0].rule_json["candidate_user_ids"] == [
        "user_001",
        "user_002",
    ]
    assert raw_reader.calls == [
        {
            "project_id": "hotel-client-a",
            "vector_version": "v1",
            "destination_terms": ("jeju", "제주"),
            "season_months": (6, 7, 8),
            "limit": 20,
        }
    ]


def test_vector_cluster_suggester_groups_similar_users_into_ai_segments() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("user_001", vector_values(0)),
            user_vector("user_002", vector_values(0)),
            user_vector("user_003", vector_values(1)),
            user_vector("user_004", vector_values(1)),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
        vector_pool_limit=20,
        vector_sample_limit=20,
        max_suggested_segments=2,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(promotion=promotion_record())

    assert reader.calls == [
        {
            "project_id": "hotel-client-a",
            "limit": 20,
            "vector_version": "v1",
        }
    ]
    assert len(segments) == 2
    assert {segment.source for segment in segments} == {"ai_suggested"}
    assert all(
        re.fullmatch(
            r"seg_ai_cluster_promo_banner_001_[0-9a-f]{10}",
            segment.segment_id,
        )
        for segment in segments
    )
    assert [segment.sample_size for segment in segments] == [2, 2]
    assert [segment.total_eligible_user_count for segment in segments] == [4, 4]
    assert [segment.sample_ratio for segment in segments] == [
        Decimal("0.500000"),
        Decimal("0.500000"),
    ]
    assert all(
        not segment.segment_name.startswith("AI suggested hotel audience")
        for segment in segments
    )
    assert {
        tuple(segment.rule_json["candidate_user_ids"])
        for segment in segments
    } == {("user_001", "user_002"), ("user_003", "user_004")}
    assert all(
        segment.rule_json["source"] == "user_vector_clustering"
        for segment in segments
    )
    assert all(
        segment.profile_json["source"] == "user_vector_clustering"
        for segment in segments
    )
    assert all("cluster_score" in segment.profile_json for segment in segments)
    assert all("top_common_features" in segment.profile_json for segment in segments)


def test_vector_cluster_suggester_uses_promotion_seed_for_sampling() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [
            user_vector(f"user_{index:03}", vector_values(index % 4))
            for index in range(8)
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
        vector_pool_limit=8,
        vector_sample_limit=4,
        max_suggested_segments=1,
        min_cluster_size=1,
    )

    first_segments = suggester.suggest_segments(
        promotion=promotion_record(
            promotion_id="promo_family_trip",
            message_brief="Promote family hotel stays.",
        )
    )
    second_segments = suggester.suggest_segments(
        promotion=promotion_record(
            promotion_id="promo_last_minute",
            message_brief="Promote last minute hotel deals.",
        )
    )

    assert first_segments[0].rule_json["sample_seed"] != second_segments[0].rule_json[
        "sample_seed"
    ]
    assert set(first_segments[0].rule_json["candidate_user_ids"]) != set(
        second_segments[0].rule_json["candidate_user_ids"]
    )


def test_vector_cluster_suggester_names_segments_from_dominant_features() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("booking_user_001", vector_values(8)),
            user_vector("booking_user_002", vector_values(8)),
            user_vector("promo_user_001", vector_values(5)),
            user_vector("promo_user_002", vector_values(5)),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
        vector_pool_limit=4,
        vector_sample_limit=4,
        max_suggested_segments=2,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(promotion=promotion_record())
    segment_names = {segment.segment_name for segment in segments}

    assert "Booking starters" in segment_names
    assert "Promotion click responders" in segment_names


def test_vector_cluster_suggester_ranks_clusters_by_promotion_intent() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("booking_user_001", vector_values(62)),
            user_vector("booking_user_002", vector_values(62)),
            user_vector("redirect_user_001", vector_values(6)),
            user_vector("redirect_user_002", vector_values(6)),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
        vector_pool_limit=4,
        vector_sample_limit=4,
        max_suggested_segments=2,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief=(
                "여름 호텔 예약 전환을 높이기 위한 이메일 예약 혜택 캠페인"
            ),
        )
    )

    assert segments[0].segment_name == "Booking conversion ready users"
    assert segments[0].profile_json["promotion_matched_features"] == [
        "Booking conversion ready users"
    ]
    assert segments[0].profile_json["promotion_cluster_similarity"] > segments[
        1
    ].profile_json["promotion_cluster_similarity"]
    assert segments[0].profile_json["recommendation_score"] > segments[
        1
    ].profile_json["recommendation_score"]


def test_vector_cluster_suggester_stores_promotion_vector_basis() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("jeju_user_001", vector_values(32)),
            user_vector("jeju_user_002", vector_values(32)),
            user_vector("booking_user_001", vector_values(9)),
            user_vector("booking_user_002", vector_values(9)),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
        vector_pool_limit=4,
        vector_sample_limit=4,
        max_suggested_segments=2,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(
        promotion=promotion_record(
            message_brief="제주 호텔 특가 예약 혜택을 안내한다.",
        )
    )

    profile_json = segments[0].profile_json
    assert profile_json["score_components"]["weights"] == {
        "promotion_cluster_similarity": 0.65,
        "cluster_quality": 0.20,
        "sample_size": 0.15,
    }
    assert profile_json["promotion_vector_basis"]["goal_metric"] == (
        "booking_conversion_rate"
    )
    assert "제주" in profile_json["promotion_vector_basis"]["message_keywords"]
    assert profile_json["promotion_vector_basis"]["weighted_features"]


def test_vector_cluster_suggester_returns_empty_when_vectors_are_insufficient() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [user_vector("user_001", vector_values(0))]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
        min_cluster_size=2,
    )

    assert suggester.suggest_segments(promotion=promotion_record()) == []


def test_vector_cluster_suggester_keeps_cluster_when_mean_vector_is_zero() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [
            user_vector("user_001", vector_values(0)),
            user_vector("user_002", [-1.0, *([0.0] * 63)]),
        ]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
        max_suggested_segments=1,
        min_cluster_size=2,
    )

    segments = suggester.suggest_segments(promotion=promotion_record())

    assert len(segments) == 1
    assert segments[0].rule_json["candidate_user_ids"] == ["user_001", "user_002"]


def test_vector_cluster_suggester_rejects_non_64_dimensional_vectors() -> None:
    reader = FakeUserBehaviorVectorRepository(
        [user_vector("user_001", [1.0] * 63, vector_dim=64)]
    )
    suggester = VectorClusterSegmentSuggester(
        user_behavior_vector_repository=reader,
    )

    with pytest.raises(ValueError, match="64 values"):
        suggester.suggest_segments(promotion=promotion_record())
