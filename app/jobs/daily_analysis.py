from __future__ import annotations

from datetime import date
from typing import Protocol

from app.analysis.models import AnalysisResult


class AnalysisRunner(Protocol):
    def run(
        self,
        project_id: int,
        analysis_date: date,
        run_id: int | None,
    ) -> AnalysisResult:
        ...


class DownstreamRunner(Protocol):
    def run(self, result: AnalysisResult) -> None:
        ...


class UserSegmentMatchingRunner(Protocol):
    def run(
        self,
        *,
        project_id: int,
        analysis_date: date,
        run_id: int | None,
    ) -> object:
        ...


class ExperimentResultUpdateRunner(Protocol):
    def run(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> object:
        ...


def run_daily_analysis_flow(
    *,
    project_id: int,
    analysis_date: date,
    run_id: int | None,
    analysis_service: AnalysisRunner,
    user_segment_matching_runner: UserSegmentMatchingRunner | None = None,
    experiment_result_update_runner: ExperimentResultUpdateRunner | None = None,
    downstream_runner: DownstreamRunner | None = None,
) -> AnalysisResult:
    """Run analysis and gate downstream work.

    The API/CLI/daily orchestrator caller owns the database transaction
    boundary and decision_runs status transitions. AnalysisService and optional
    matching/experiment/downstream runners are called inside that boundary and
    do not mark runs. This flow does not commit, rollback, or own transactions.
    """
    result = analysis_service.run(
        project_id=project_id,
        analysis_date=analysis_date,
        run_id=run_id,
    )
    if user_segment_matching_runner is not None:
        user_segment_matching_runner.run(
            project_id=project_id,
            analysis_date=analysis_date,
            run_id=run_id,
        )
    if experiment_result_update_runner is not None:
        experiment_result_update_runner.run(
            project_id=project_id,
            analysis_date=analysis_date,
        )
    if result.anomaly_count > 0 and downstream_runner is not None:
        downstream_runner.run(result)
    return result
