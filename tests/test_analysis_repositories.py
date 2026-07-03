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
    PromotionTargetSegmentWrite,
    SegmentDefinitionRepository,
    SegmentVectorRecord,
    SegmentVectorRepository,
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
        sources=("custom_chatkit", "system_default"),
    )

    assert [segment.segment_id for segment in segments] == [
        "seg_repeat_hotel_no_booking"
    ]
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from segment_definitions" in sql
    assert "profile_json" in sql
    assert "total_eligible_user_count" in sql
    assert "status = 'active'" in sql
    assert "source in (%s, %s)" in sql
    assert call.params == ("hotel-client-a", "custom_chatkit", "system_default")


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


def test_promotion_analysis_repository_saves_target_segments() -> None:
    db = FakePostgresExecutor()
    repo = PromotionAnalysisRepository(db)
    segment = PromotionTargetSegmentWrite(
        analysis_id="analysis_banner_001",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        segment_id="seg_repeat_hotel_no_booking",
        segment_name="Repeat hotel viewers without booking",
        rule_json={"event_name": "hotel_detail_view"},
        profile_json={"hotel_cluster": "seoul_center"},
        content_brief_json={"keywords": ["free cancellation"]},
        data_evidence_json={"event_count": 120},
        segment_vector_id="segvec_repeat_hotel_no_booking_v1",
        estimated_size=1342,
        priority="high",
        status="planned",
    )

    repo.save_target_segments([segment])

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert call.operation == "execute"
    assert "insert into promotion_target_segments" in sql
    assert "content_brief_json" in sql
    assert "data_evidence_json" in sql
    assert "segment_vector_id" in sql
    assert "status" in sql
    assert call.params == (
        "analysis_banner_001",
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
        "seg_repeat_hotel_no_booking",
        "Repeat hotel viewers without booking",
        {"event_name": "hotel_detail_view"},
        {"hotel_cluster": "seoul_center"},
        {"keywords": ["free cancellation"]},
        {"event_count": 120},
        "segvec_repeat_hotel_no_booking_v1",
        1342,
        "high",
        "planned",
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
    assert select_call.params == (
        "hotel-client-a",
        "promo_banner_001",
        "seg_repeat_hotel_no_booking",
    )
    assert "insert into segment_vectors" in compact_sql(insert_call.query)
    assert insert_call.params == (
        "segvec_repeat_hotel_no_booking_v1",
        "hotel-client-a",
        "promo_banner_001",
        None,
        "analysis_banner_001",
        "seg_repeat_hotel_no_booking",
        64,
        vector_values,
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
