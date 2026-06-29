from __future__ import annotations

from datetime import date
from typing import Protocol

from app.analysis.anomaly import build_root_cause_candidates, detect_segment_anomalies
from app.analysis.models import (
    AnalysisResult,
    AnalysisWindow,
    BaselineMetrics,
    RootCauseCandidate,
    SegmentAggregate,
    SegmentAnomalyCandidate,
    StoredAnomaly,
    StoredSegment,
    UserPrimarySegmentCandidate,
)
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


class UserPrimarySegmentRepository(Protocol):
    def fetch_user_primary_segment_candidates(
        self,
        project_id: int,
        window: AnalysisWindow,
    ) -> list[UserPrimarySegmentCandidate]:
        ...


class UserSegmentMembershipRepository(Protocol):
    def upsert_user_segment_memberships(
        self,
        project_id: int,
        analysis_date: date,
        candidates: list[UserPrimarySegmentCandidate],
        stored_segments: dict[str, StoredSegment],
        run_id: int | None,
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
        project_repository: ProjectTimezoneRepository,
        segment_aggregate_repository: SegmentAggregateRepository | None = None,
        segment_metrics_repository: SegmentMetricsRepository | None = None,
        user_primary_segment_repository: UserPrimarySegmentRepository | None = None,
        user_segment_membership_repository: UserSegmentMembershipRepository | None = None,
        anomaly_repository: SegmentAnomalyRepository | None = None,
    ) -> None:
        self.project_repository = project_repository
        self.segment_aggregate_repository = segment_aggregate_repository
        self.segment_metrics_repository = segment_metrics_repository
        self.user_primary_segment_repository = user_primary_segment_repository
        self.user_segment_membership_repository = user_segment_membership_repository
        self.anomaly_repository = anomaly_repository

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
        membership_count = 0
        if (
            self.user_primary_segment_repository is not None
            and self.user_segment_membership_repository is not None
        ):
            candidates = self.user_primary_segment_repository.fetch_user_primary_segment_candidates(
                project_id,
                window,
            )
            membership_count = self.user_segment_membership_repository.upsert_user_segment_memberships(
                project_id=project_id,
                analysis_date=analysis_date,
                candidates=candidates,
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
            root_cause_count = self.anomaly_repository.upsert_root_cause_candidates(
                build_root_cause_candidates(
                    aggregates=aggregates,
                    stored_segments=stored_segments,
                    stored_anomalies=stored_anomalies,
                )
            )
        return AnalysisResult(
            segment_count=len(stored_segments),
            membership_count=membership_count,
            metric_count=metric_count,
            anomaly_count=len(stored_anomalies),
            root_cause_count=root_cause_count,
            anomaly_segment_ids=[anomaly.segment_id for anomaly in stored_anomalies],
            anomaly_ids=[anomaly.id for anomaly in stored_anomalies],
        )
