from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

import pytest

from app.analysis.repositories import (
    HotelProfileRepository,
    PromotionAnalysisRepository,
    PromotionAnalysisWrite,
    PromotionRepository,
    PromotionSegmentSuggestionWrite,
    SegmentDefinitionRecord,
    SegmentDefinitionRepository,
    SegmentVectorRecord,
    SegmentVectorRepository,
    UserBehaviorVectorRepository,
)


@dataclass(frozen=True)
class DbCall:
    operation: str
    query: str
    params: Sequence[Any] | Mapping[str, Any]


class FakePostgresExecutor:
    def __init__(
        self,
        *,
        fetchone_result: Mapping[str, Any] | None = None,
        fetchall_result: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.calls: list[DbCall] = []

    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        self.calls.append(DbCall("fetchone", query, params))
        return self.fetchone_result

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        self.calls.append(DbCall("fetchall", query, params))
        return self.fetchall_result

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        self.calls.append(DbCall("execute", query, params))


class FakeClickHouseResult:
    def __init__(self, rows: list[Any]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.calls: list[DbCall] = []

    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> FakeClickHouseResult:
        self.calls.append(DbCall("query", query, parameters or {}))
        return FakeClickHouseResult(self.rows)


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def promotion_row() -> dict[str, Any]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "channel": "onsite_banner",
        "goal_metric": "booking_conversion_rate",
        "goal_target_value": Decimal("0.030000"),
        "goal_basis": "all_segments",
        "min_sample_size": 1000,
        "landing_url": "https://demo-stay.example.com/summer",
        "message_brief": "Drive summer hotel booking.",
    }


def test_promotion_repository_get_for_analysis_success() -> None:
    db = FakePostgresExecutor(fetchone_result=promotion_row())
    repo = PromotionRepository(db)

    promotion = repo.get_for_analysis(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
    )

    assert promotion is not None
    assert promotion.promotion_id == "promo_banner_001"
    assert promotion.project_id == "hotel-client-a"
    assert promotion.campaign_id == "camp_summer_2026"
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from promotions" in sql
    assert "where project_id = %s" in sql
    assert "and campaign_id = %s" in sql
    assert "and promotion_id = %s" in sql
    assert "min_sample_size" in sql
    assert call.params == (
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
    )


def test_promotion_repository_get_for_analysis_returns_none() -> None:
    db = FakePostgresExecutor(fetchone_result=None)
    repo = PromotionRepository(db)

    promotion = repo.get_for_analysis(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="missing_promo",
    )

    assert promotion is None
    assert db.calls[0].operation == "fetchone"


def test_segment_definition_repository_filters_active_sources() -> None:
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "segment_id": "seg_repeat_hotel_no_booking",
                "project_id": "hotel-client-a",
                "campaign_id": "camp_summer_2026",
                "promotion_id": "promo_banner_001",
                "segment_name": "Repeat hotel viewers without booking",
                "source": "custom_chatkit",
                "query_preview_id": "seg_query_preview_001",
                "natural_language_query": "same hotel views without booking",
                "generated_sql": "SELECT user_id FROM hotel_detail_events",
                "rule_json": {"event_name": "hotel_detail_view"},
                "profile_json": {"primary_segment": "seg_repeat_hotel_no_booking"},
                "sample_size": 1342,
                "total_eligible_user_count": 74200,
                "sample_ratio": Decimal("0.018000"),
                "status": "active",
            }
        ]
    )
    repo = SegmentDefinitionRepository(db)

    segments = repo.list_active(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        sources=("custom_chatkit", "system_default"),
    )

    assert [segment.segment_id for segment in segments] == [
        "seg_repeat_hotel_no_booking"
    ]
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from segment_definitions" in sql
    assert "campaign_id" in sql
    assert "promotion_id" in sql
    assert "profile_json" in sql
    assert "total_eligible_user_count" in sql
    assert "status = 'active'" in sql
    assert "(campaign_id is null or campaign_id = %s)" in sql
    assert "(promotion_id is null or promotion_id = %s)" in sql
    assert "source in (%s, %s)" in sql
    assert call.params == (
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
        "custom_chatkit",
        "system_default",
    )


def test_segment_definition_repository_saves_ai_suggested_segments() -> None:
    db = FakePostgresExecutor()
    repo = SegmentDefinitionRepository(db)
    segment = SegmentDefinitionRecord(
        segment_id="seg_ai_cluster_promo_banner_001_1_abcdef1234",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        segment_name="AI suggested hotel audience 1",
        source="ai_suggested",
        query_preview_id=None,
        natural_language_query="Users grouped by similar hotel behavior vectors.",
        generated_sql=None,
        rule_json={
            "source": "user_vector_clustering",
            "candidate_user_ids": ["user_001", "user_002"],
        },
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_banner_001_1_abcdef1234",
            "cluster_score": 0.98,
        },
        sample_size=2,
        total_eligible_user_count=4,
        sample_ratio=Decimal("0.500000"),
        status="active",
    )

    repo.save_ai_suggested([segment])

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert call.operation == "execute"
    assert "insert into segment_definitions" in sql
    assert "on conflict (segment_id) do update" in sql
    assert "where segment_definitions.source = 'ai_suggested'" in sql
    assert call.params == (
        "seg_ai_cluster_promo_banner_001_1_abcdef1234",
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
        "AI suggested hotel audience 1",
        "ai_suggested",
        None,
        "Users grouped by similar hotel behavior vectors.",
        None,
        {
            "source": "user_vector_clustering",
            "candidate_user_ids": ["user_001", "user_002"],
        },
        {
            "primary_segment": "seg_ai_cluster_promo_banner_001_1_abcdef1234",
            "cluster_score": 0.98,
        },
        2,
        4,
        Decimal("0.500000"),
        "active",
    )


def test_segment_definition_repository_rejects_non_ai_suggested_save() -> None:
    db = FakePostgresExecutor()
    repo = SegmentDefinitionRepository(db)
    segment = SegmentDefinitionRecord(
        segment_id="seg_mobile_user",
        project_id="hotel-client-a",
        segment_name="Mobile users",
        source="system_default",
        query_preview_id=None,
        natural_language_query=None,
        generated_sql=None,
        rule_json={},
        profile_json={},
        sample_size=100,
        total_eligible_user_count=1000,
        sample_ratio=Decimal("0.100000"),
        status="active",
    )

    with pytest.raises(ValueError, match="ai_suggested"):
        repo.save_ai_suggested([segment])

    assert db.calls == []


def test_promotion_analysis_repository_saves_analysis() -> None:
    db = FakePostgresExecutor()
    repo = PromotionAnalysisRepository(db)
    analysis = PromotionAnalysisWrite(
        analysis_id="analysis_banner_001",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        status="completed",
        focus_segment_ids_json=["seg_repeat_hotel_no_booking"],
        operator_instruction="Focus on users with booking intent.",
        input_snapshot_json={"promotion": {"promotion_id": "promo_banner_001"}},
        profile_summary_json={"selected_segment_count": 1},
        output_json={"target_segment_count": 1},
    )

    repo.save_analysis(analysis)

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert call.operation == "execute"
    assert "insert into promotion_analyses" in sql
    assert "focus_segment_ids_json" in sql
    assert "operator_instruction" in sql
    assert "input_snapshot_json" in sql
    assert "profile_summary_json" in sql
    assert "output_json" in sql
    assert call.params == (
        "analysis_banner_001",
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
        "completed",
        ["seg_repeat_hotel_no_booking"],
        "Focus on users with booking intent.",
        {"promotion": {"promotion_id": "promo_banner_001"}},
        {"selected_segment_count": 1},
        {"target_segment_count": 1},
    )


def test_promotion_analysis_repository_saves_segment_suggestions() -> None:
    db = FakePostgresExecutor()
    repo = PromotionAnalysisRepository(db)
    suggestion = PromotionSegmentSuggestionWrite(
        suggestion_id="sugg_analysis_banner_001_seg_repeat",
        analysis_id="analysis_banner_001",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        segment_id="seg_repeat_hotel_no_booking",
        suggested_rank=1,
        suggestion_source="ai_ranked_existing",
        status="suggested",
        score_json={"rank": 1, "estimated_size": 1342, "priority": "high"},
        reason_json={"channel": "onsite_banner", "goal_metric": "booking_conversion_rate"},
        metadata_json={
            "segment_name": "Repeat hotel viewers without booking",
            "segment_vector_id": "segvec_repeat_hotel_no_booking_v1",
        },
    )

    repo.save_segment_suggestions([suggestion])

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert call.operation == "execute"
    assert "insert into promotion_segment_suggestions" in sql
    assert "suggestion_id" in sql
    assert "suggested_rank" in sql
    assert "suggestion_source" in sql
    assert "score_json" in sql
    assert "reason_json" in sql
    assert "metadata_json" in sql
    assert "status" in sql
    assert call.params == (
        "sugg_analysis_banner_001_seg_repeat",
        "analysis_banner_001",
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
        "seg_repeat_hotel_no_booking",
        1,
        "ai_ranked_existing",
        "suggested",
        {"rank": 1, "estimated_size": 1342, "priority": "high"},
        {"channel": "onsite_banner", "goal_metric": "booking_conversion_rate"},
        {
            "segment_name": "Repeat hotel viewers without booking",
            "segment_vector_id": "segvec_repeat_hotel_no_booking_v1",
        },
    )


def test_segment_vector_repository_get_and_save_vector() -> None:
    vector_values = [0.1] * 64
    db = FakePostgresExecutor(
        fetchone_result={
            "segment_vector_id": "segvec_repeat_hotel_no_booking_v1",
            "project_id": "hotel-client-a",
            "promotion_id": "promo_banner_001",
            "promotion_run_id": None,
            "analysis_id": "analysis_banner_001",
            "segment_id": "seg_repeat_hotel_no_booking",
            "vector_dim": 64,
            "vector_values": vector_values,
            "vector_version": "v1",
            "source": "decision_analysis",
            "embedding": vector_values,
        }
    )
    repo = SegmentVectorRepository(db)

    loaded = repo.get_by_segment(
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        segment_id="seg_repeat_hotel_no_booking",
    )
    assert loaded is not None
    repo.save(loaded)

    select_call, insert_call = db.calls
    assert "from segment_vectors" in compact_sql(select_call.query)
    assert "embedding::text as embedding" in compact_sql(select_call.query)
    assert select_call.params == (
        "hotel-client-a",
        "promo_banner_001",
        "seg_repeat_hotel_no_booking",
    )
    assert "insert into segment_vectors" in compact_sql(insert_call.query)
    assert "embedding" in compact_sql(insert_call.query)
    assert insert_call.params == (
        "segvec_repeat_hotel_no_booking_v1",
        "hotel-client-a",
        "promo_banner_001",
        None,
        "analysis_banner_001",
        "seg_repeat_hotel_no_booking",
        64,
        vector_values,
        "[" + ",".join(str(value) for value in vector_values) + "]",
        "v1",
        "decision_analysis",
    )


def test_segment_vector_repository_rejects_non_64_dimensional_vectors() -> None:
    db = FakePostgresExecutor()
    repo = SegmentVectorRepository(db)
    vector = SegmentVectorRecord(
        segment_vector_id="segvec_bad",
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        promotion_run_id=None,
        analysis_id="analysis_banner_001",
        segment_id="seg_bad",
        vector_dim=64,
        vector_values=[0.1] * 63,
        vector_version="v1",
        source="decision_analysis",
    )

    with pytest.raises(ValueError, match="64 values"):
        repo.save(vector)

    assert db.calls == []


def test_segment_vector_repository_rejects_zero_vectors() -> None:
    db = FakePostgresExecutor()
    repo = SegmentVectorRepository(db)
    vector = SegmentVectorRecord(
        segment_vector_id="segvec_zero",
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        promotion_run_id=None,
        analysis_id="analysis_banner_001",
        segment_id="seg_zero",
        vector_dim=64,
        vector_values=[0.0] * 64,
        vector_version="v1",
        source="decision_analysis",
    )

    with pytest.raises(ValueError, match="zero vector"):
        repo.save(vector)

    assert db.calls == []


def test_user_behavior_vector_repository_queries_candidate_user_vectors() -> None:
    vector_values = [0.1] * 64
    client = FakeClickHouseClient(
        rows=[
            {
                "project_id": "hotel-client-a",
                "user_id": "user_001",
                "vector_dim": 64,
                "vector_values": vector_values,
                "vector_version": "v1",
                "source": "batch_profile",
            }
        ]
    )
    repo = UserBehaviorVectorRepository(client)

    vectors = repo.list_by_user_ids(
        project_id="hotel-client-a",
        user_ids=["user_001", "user_002"],
        vector_version="v1",
    )

    assert len(vectors) == 1
    assert vectors[0].project_id == "hotel-client-a"
    assert vectors[0].user_id == "user_001"
    assert vectors[0].vector_dim == 64
    assert vectors[0].vector_values == vector_values
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from user_behavior_vectors" in sql
    assert "project_id = {project_id:string}" in sql
    assert "vector_dim = {vector_dim:uint16}" in sql
    assert "vector_version = {vector_version:string}" in sql
    assert "user_id in {user_ids:array(string)}" in sql
    assert "argmax(vector_values, updated_at) as vector_values" in sql
    assert "group by project_id, user_id, vector_version" in sql
    assert call.params == {
        "project_id": "hotel-client-a",
        "vector_dim": 64,
        "vector_version": "v1",
        "user_ids": ["user_001", "user_002"],
    }


def test_user_behavior_vector_repository_skips_empty_user_ids() -> None:
    client = FakeClickHouseClient(rows=[])
    repo = UserBehaviorVectorRepository(client)

    vectors = repo.list_by_user_ids(
        project_id="hotel-client-a",
        user_ids=[],
    )

    assert vectors == []
    assert client.calls == []


def test_user_behavior_vector_repository_queries_recent_project_vectors() -> None:
    vector_values = [0.1] * 64
    client = FakeClickHouseClient(
        rows=[
            {
                "project_id": "hotel-client-a",
                "user_id": "user_001",
                "vector_dim": 64,
                "vector_values": vector_values,
                "vector_version": "v1",
                "source": "batch_profile",
            }
        ]
    )
    repo = UserBehaviorVectorRepository(client)

    vectors = repo.list_recent(
        project_id="hotel-client-a",
        limit=50,
        vector_version="v1",
    )

    assert len(vectors) == 1
    assert vectors[0].user_id == "user_001"
    assert vectors[0].vector_values == vector_values
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from user_behavior_vectors" in sql
    assert "project_id = {project_id:string}" in sql
    assert "vector_dim = {vector_dim:uint16}" in sql
    assert "vector_version = {vector_version:string}" in sql
    assert "argmax(vector_values, updated_at) as vector_values" in sql
    assert "group by project_id, user_id, vector_version" in sql
    assert "order by last_updated_at desc, user_id asc" in sql
    assert "limit {limit:uint32}" in sql
    assert call.params == {
        "project_id": "hotel-client-a",
        "vector_dim": 64,
        "vector_version": "v1",
        "limit": 50,
    }


def test_hotel_profile_repository_queries_marketing_profiles() -> None:
    client = FakeClickHouseClient(
        rows=[
            {
                "primary_segment": "family_trip",
                "event_count": 120,
                "booking_count": 18,
                "mobile_ratio": 0.65,
                "package_ratio": 0.25,
                "avg_stay_nights": 2.4,
                "avg_days_until_checkin": 14.2,
            }
        ]
    )
    repo = HotelProfileRepository(client)

    profiles = repo.list_marketing_profiles(project_id="hotel-client-a")

    assert profiles[0].project_id == "hotel-client-a"
    assert profiles[0].profile_name == "family_trip"
    assert profiles[0].profile_json == {
        "event_count": 120,
        "booking_count": 18,
        "mobile_ratio": 0.65,
        "package_ratio": 0.25,
        "avg_stay_nights": 2.4,
        "avg_days_until_checkin": 14.2,
    }
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from hotel_marketing_profiles" in sql
    assert "primary_segment" in sql
    assert "countif(is_booking = 1) as booking_count" in sql
    assert "group by primary_segment" in sql
    assert "order by event_count desc" in sql
    assert "project_id" not in sql
    assert call.params == {}


def test_hotel_profile_repository_queries_expedia_event_profile() -> None:
    client = FakeClickHouseClient(rows=[{"hotel_cluster": "seoul_center", "event_count": 42}])
    repo = HotelProfileRepository(client)

    summary = repo.summarize_expedia_hotel_events(
        project_id="hotel-client-a",
        limit=5,
    )

    assert summary == [{"hotel_cluster": "seoul_center", "event_count": 42}]
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from expedia_hotel_events" in sql
    assert "group by hotel_cluster" in sql
    assert "limit {limit:uint32}" in sql
    assert "project_id" not in sql
    assert call.params == {"limit": 5}
