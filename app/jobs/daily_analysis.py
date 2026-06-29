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


def run_daily_analysis_flow(
    *,
    project_id: int,
    analysis_date: date,
    run_id: int | None,
    analysis_service: AnalysisRunner,
    downstream_runner: DownstreamRunner | None = None,
) -> AnalysisResult:
    result = analysis_service.run(
        project_id=project_id,
        analysis_date=analysis_date,
        run_id=run_id,
    )
    if result.anomaly_count > 0 and downstream_runner is not None:
        downstream_runner.run(result)
    return result
