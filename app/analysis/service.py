from __future__ import annotations

from datetime import date
from typing import Protocol

from app.analysis.anomaly import (
    build_root_cause_candidates,
    derive_matching_weights,
    detect_segment_anomalies,
)
from app.analysis.models import (
    AnalysisResult,
    AnalysisWindow,
    BaselineMetrics,
    RootCauseCandidate,
    SegmentAggregate,
    SegmentAnomalyCandidate,
    StoredAnomaly,
    StoredSegment,
)
from app.analysis.time_window import build_analysis_window


class ProjectRepository(Protocol):
    def get_project_timezone(self, project_id: int) -> str:
        ...

    def get_project_key(self, project_id: int) -> str:
        ...


class SegmentAggregateRepository(Protocol):
    def fetch_segment_aggregates(
        self,
        project_id: str,
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

    def update_segment_daily_metric_matching(
        self,
        *,
        project_id: int,
        analysis_date: date,
        matching_by_segment_id: dict[int, dict],
    ) -> int:
        ...


class SegmentAnomalyRepository(Protocol):
    def fetch_segment_metric_baselines(
        self,
        project_id: int,
        analysis_date: date,
        stored_segments: dict[str, StoredSegment],
    ) -> dict[int, BaselineMetrics]:
        ...

    def update_segment_daily_metric_baselines(
        self,
        project_id: int,
        analysis_date: date,
        baselines: dict[int, BaselineMetrics],
    ) -> int:
        ...

    def upsert_segment_anomalies(
        self,
        project_id: int,
        analysis_date: date,
        anomalies: list[SegmentAnomalyCandidate],
        run_id: int | None,
    ) -> list[StoredAnomaly]:
        ...

    def upsert_root_cause_candidates(
        self,
        root_causes: list[RootCauseCandidate],
    ) -> int:
        ...


class AnalysisService:
    def __init__(
        self,
        project_repository: ProjectRepository,
        segment_aggregate_repository: SegmentAggregateRepository | None = None,
        segment_metrics_repository: SegmentMetricsRepository | None = None,
        anomaly_repository: SegmentAnomalyRepository | None = None,
    ) -> None:
        self.project_repository = project_repository
        self.segment_aggregate_repository = segment_aggregate_repository
        self.segment_metrics_repository = segment_metrics_repository
        self.anomaly_repository = anomaly_repository

    def run(
        self,
        project_id: int,
        analysis_date: date,
        run_id: int | None,
    ) -> AnalysisResult:
        timezone = self.project_repository.get_project_timezone(project_id)
        clickhouse_project_id = self.project_repository.get_project_key(project_id)
        window = build_analysis_window(analysis_date, timezone)
        aggregates = (
            self.segment_aggregate_repository.fetch_segment_aggregates(
                clickhouse_project_id,
                window,
            )
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
        stored_anomalies: list[StoredAnomaly] = []
        root_cause_count = 0
        if self.anomaly_repository is not None and stored_segments:
            baselines = self.anomaly_repository.fetch_segment_metric_baselines(
                project_id=project_id,
                analysis_date=analysis_date,
                stored_segments=stored_segments,
            )
            self.anomaly_repository.update_segment_daily_metric_baselines(
                project_id=project_id,
                analysis_date=analysis_date,
                baselines=baselines,
            )
            anomaly_candidates = detect_segment_anomalies(
                aggregates=aggregates,
                stored_segments=stored_segments,
                baselines=baselines,
            )
            stored_anomalies = self.anomaly_repository.upsert_segment_anomalies(
                project_id=project_id,
                analysis_date=analysis_date,
                anomalies=anomaly_candidates,
                run_id=run_id,
            )
            root_causes = build_root_cause_candidates(
                aggregates=aggregates,
                stored_segments=stored_segments,
                stored_anomalies=stored_anomalies,
            )
            root_cause_count = self.anomaly_repository.upsert_root_cause_candidates(
                root_causes
            )
            if self.segment_metrics_repository is not None:
                matching_by_segment_id = {}
                segment_id_by_anomaly_id = {
                    anomaly.id: anomaly.segment_id
                    for anomaly in stored_anomalies
                }
                aggregate_by_segment_id = {
                    stored_segments[aggregate.segment_key].id: aggregate
                    for aggregate in aggregates
                    if aggregate.segment_key in stored_segments
                }
                for root_cause in sorted(
                    root_causes,
                    key=lambda item: (item.rank_no, -item.impact_score),
                ):
                    segment_id = segment_id_by_anomaly_id.get(root_cause.anomaly_id)
                    if segment_id is None or segment_id in matching_by_segment_id:
                        continue
                    aggregate = aggregate_by_segment_id.get(segment_id)
                    if aggregate is None:
                        continue
                    matching_by_segment_id[segment_id] = derive_matching_weights(
                        aggregate.dimensions,
                        float(root_cause.impact_score),
                    )
                if matching_by_segment_id:
                    self.segment_metrics_repository.update_segment_daily_metric_matching(
                        project_id=project_id,
                        analysis_date=analysis_date,
                        matching_by_segment_id=matching_by_segment_id,
                    )
        return AnalysisResult(
            segment_count=len(stored_segments),
            membership_count=0,
            metric_count=metric_count,
            anomaly_count=len(stored_anomalies),
            root_cause_count=root_cause_count,
            anomaly_segment_ids=[anomaly.segment_id for anomaly in stored_anomalies],
            anomaly_ids=[anomaly.id for anomaly in stored_anomalies],
        )
