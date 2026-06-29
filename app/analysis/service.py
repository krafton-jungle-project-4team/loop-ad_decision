from __future__ import annotations

from datetime import date
from typing import Protocol

from app.analysis.models import AnalysisResult
from app.analysis.time_window import build_analysis_window


class ProjectTimezoneRepository(Protocol):
    def get_project_timezone(self, project_id: int) -> str:
        ...


class AnalysisService:
    def __init__(self, project_repository: ProjectTimezoneRepository) -> None:
        self.project_repository = project_repository

    def run(
        self,
        project_id: int,
        analysis_date: date,
        run_id: int | None,
    ) -> AnalysisResult:
        timezone = self.project_repository.get_project_timezone(project_id)
        build_analysis_window(analysis_date, timezone)
        return AnalysisResult()
