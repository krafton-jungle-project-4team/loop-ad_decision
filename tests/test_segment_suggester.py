from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

import pytest

from app.analysis.repositories import PromotionRecord, UserBehaviorVectorRecord
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


def promotion_record(
    *,
    promotion_id: str = "promo_banner_001",
    message_brief: str = "Drive hotel booking conversion for summer stays.",
) -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id=promotion_id,
        channel="onsite_banner",
        goal_metric="booking_conversion_rate",
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
        segment.segment_id.startswith("seg_ai_cluster_promo_banner_001_")
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
