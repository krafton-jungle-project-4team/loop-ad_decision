from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.analysis.models import SegmentAggregate, StoredSegment
from app.analysis.service import AnalysisService
from app.analysis.time_window import build_analysis_window


class FakeProjectRepository:
    def __init__(self, timezone: str = "Asia/Seoul") -> None:
        self.timezone = timezone
        self.project_ids: list[int] = []

    def get_project_timezone(self, project_id: int) -> str:
        self.project_ids.append(project_id)
        return self.timezone


class FakeSegmentAggregateRepository:
    def __init__(self, aggregates: list[SegmentAggregate]) -> None:
        self.aggregates = aggregates

    def fetch_segment_aggregates(self, project_id, window):
        return self.aggregates


class FakeSegmentMetricsRepository:
    def __init__(self) -> None:
        self.segment_run_ids: list[int | None] = []
        self.metric_run_ids: list[int | None] = []

    def upsert_segments(self, project_id, aggregates, run_id):
        self.segment_run_ids.append(run_id)
        return {
            aggregate.segment_key: StoredSegment(index + 1, aggregate.segment_key)
            for index, aggregate in enumerate(aggregates)
        }

    def upsert_segment_daily_metrics(
        self,
        project_id,
        analysis_date,
        aggregates,
        stored_segments,
        run_id,
    ):
        self.metric_run_ids.append(run_id)
        return len(stored_segments)


def segment_aggregate(segment_key: str = "age_30s__gender_male__device_mobile__channel_kakao__category_fresh") -> SegmentAggregate:
    return SegmentAggregate(
        project_id=1,
        segment_key=segment_key,
        name="30s / male / mobile / kakao / fresh",
        dimensions={
            "age_group": "30s",
            "gender": "male",
            "device_type": "mobile",
            "acquisition_channel": "kakao",
            "primary_category": "fresh",
        },
        user_count=30,
        session_count=40,
        page_view_count=100,
        product_view_count=100,
        add_to_cart_count=20,
        checkout_start_count=10,
        purchase_count=5,
        ad_impression_count=0,
        ad_click_count=0,
        revenue=Decimal("1000"),
        view_to_cart_rate=Decimal("0.2"),
        cart_to_checkout_rate=Decimal("0.5"),
        checkout_to_purchase_rate=Decimal("0.5"),
        view_to_purchase_rate=Decimal("0.05"),
        ctr=None,
        cvr=Decimal("0.05"),
    )


def test_build_analysis_window_uses_project_timezone_day() -> None:
    window = build_analysis_window(date(2021, 1, 4), "Asia/Seoul")

    assert window.window_start.isoformat() == "2021-01-04T00:00:00+09:00"
    assert window.window_end.isoformat() == "2021-01-05T00:00:00+09:00"


def test_analysis_service_accepts_run_id_none() -> None:
    project_repository = FakeProjectRepository()
    service = AnalysisService(project_repository=project_repository)

    result = service.run(project_id=1, analysis_date=date(2021, 1, 4), run_id=None)

    assert project_repository.project_ids == [1]
    assert result.segment_count == 0
    assert result.membership_count == 0
    assert result.metric_count == 0
    assert result.anomaly_ids == []


def test_analysis_service_stores_segments_and_metrics_with_nullable_run_id() -> None:
    aggregate_repository = FakeSegmentAggregateRepository([segment_aggregate()])
    metrics_repository = FakeSegmentMetricsRepository()
    service = AnalysisService(
        project_repository=FakeProjectRepository(),
        segment_aggregate_repository=aggregate_repository,
        segment_metrics_repository=metrics_repository,
    )

    result = service.run(project_id=1, analysis_date=date(2021, 1, 4), run_id=None)

    assert result.segment_count == 1
    assert result.metric_count == 1
    assert metrics_repository.segment_run_ids == [None]
    assert metrics_repository.metric_run_ids == [None]
