from __future__ import annotations

from datetime import date
from typing import Protocol

from app.analysis.models import AnalysisResult, AnalysisWindow, SegmentAggregate, StoredSegment
from app.analysis.time_window import build_analysis_window


class ProjectTimezoneRepository(Protocol):
    def get_project_timezone(self, project_id: int) -> str:
        ...


class SegmentAggregateRepository(Protocol):
    def fetch_segment_aggregates(
        self,
        project_id: int,
        window: AnalysisWindow,
    ) -> list[SegmentAggregate]:
        ...


class SegmentMetricsRepository(Protocol):
    def upsert_segments(
        self,
        project_id: int,
        aggregates: list[SegmentAggregate],
        run_id: int | None,
    ) -> dict[str, StoredSegment]:
        ...

    def upsert_segment_daily_metrics(
        self,
        project_id: int,
        analysis_date: date,
        aggregates: list[SegmentAggregate],
        stored_segments: dict[str, StoredSegment],
        run_id: int | None,
    ) -> int:
        ...


class AnalysisService:
    def __init__(
        self,
        project_repository: ProjectTimezoneRepository,
        segment_aggregate_repository: SegmentAggregateRepository | None = None,
        segment_metrics_repository: SegmentMetricsRepository | None = None,
    ) -> None:
        self.project_repository = project_repository
        self.segment_aggregate_repository = segment_aggregate_repository
        self.segment_metrics_repository = segment_metrics_repository

    def run(
        self,
        project_id: int,
        analysis_date: date,
        run_id: int | None,
    ) -> AnalysisResult:
        timezone = self.project_repository.get_project_timezone(project_id)
        window = build_analysis_window(analysis_date, timezone)
        aggregates = (
            self.segment_aggregate_repository.fetch_segment_aggregates(project_id, window)
            if self.segment_aggregate_repository is not None
            else []
        )
        stored_segments: dict[str, StoredSegment] = {}
        metric_count = 0
        if self.segment_metrics_repository is not None:
            stored_segments = self.segment_metrics_repository.upsert_segments(
                project_id=project_id,
                aggregates=aggregates,
                run_id=run_id,
            )
            metric_count = self.segment_metrics_repository.upsert_segment_daily_metrics(
                project_id=project_id,
                analysis_date=analysis_date,
                aggregates=aggregates,
                stored_segments=stored_segments,
                run_id=run_id,
            )
        return AnalysisResult(
            segment_count=len(stored_segments),
            metric_count=metric_count,
        )
