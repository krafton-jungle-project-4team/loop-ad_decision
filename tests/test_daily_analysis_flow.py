from __future__ import annotations

from datetime import date
from pathlib import Path

from app.analysis.models import AnalysisResult
from app.jobs.daily_analysis import run_daily_analysis_flow


class FakeAnalysisService:
    def __init__(self, result: AnalysisResult) -> None:
        self.result = result
        self.calls: list[tuple[int, date, int | None]] = []

    def run(
        self,
        project_id: int,
        analysis_date: date,
        run_id: int | None,
    ) -> AnalysisResult:
        self.calls.append((project_id, analysis_date, run_id))
        return self.result


class DownstreamStub:
    def __init__(self) -> None:
        self.results: list[AnalysisResult] = []

    def run(self, result: AnalysisResult) -> None:
        self.results.append(result)


def test_daily_analysis_flow_skips_downstream_when_no_anomaly() -> None:
    analysis_service = FakeAnalysisService(AnalysisResult(anomaly_count=0))
    downstream = DownstreamStub()

    result = run_daily_analysis_flow(
        project_id=1,
        analysis_date=date(2021, 1, 4),
        run_id=None,
        analysis_service=analysis_service,
        downstream_runner=downstream,
    )

    assert result.anomaly_count == 0
    assert analysis_service.calls == [(1, date(2021, 1, 4), None)]
    assert downstream.results == []


def test_daily_analysis_flow_passes_anomaly_ids_to_downstream() -> None:
    analysis_result = AnalysisResult(
        anomaly_count=1,
        anomaly_segment_ids=[10],
        anomaly_ids=[501],
    )
    analysis_service = FakeAnalysisService(analysis_result)
    downstream = DownstreamStub()

    result = run_daily_analysis_flow(
        project_id=1,
        analysis_date=date(2021, 1, 4),
        run_id=77,
        analysis_service=analysis_service,
        downstream_runner=downstream,
    )

    assert result.anomaly_ids == [501]
    assert result.anomaly_segment_ids == [10]
    assert downstream.results == [analysis_result]


def test_public_analysis_or_serving_api_is_not_added() -> None:
    root = Path(__file__).resolve().parents[1]
    current_file = Path(__file__).resolve()
    python_files = [
        path
        for path in root.rglob("*.py")
        if ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.resolve() != current_file
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in python_files)

    assert "/analysis/funnel/recommend" not in combined
    assert "serving" not in {path.name for path in python_files}
