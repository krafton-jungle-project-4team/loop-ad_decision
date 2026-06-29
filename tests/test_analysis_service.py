from __future__ import annotations

from datetime import date

from app.analysis.service import AnalysisService
from app.analysis.time_window import build_analysis_window


class FakeProjectRepository:
    def __init__(self, timezone: str = "Asia/Seoul") -> None:
        self.timezone = timezone
        self.project_ids: list[int] = []

    def get_project_timezone(self, project_id: int) -> str:
        self.project_ids.append(project_id)
        return self.timezone


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
