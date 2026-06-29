from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from app.analysis.models import AnalysisWindow


def build_analysis_window(analysis_date: date, timezone: str) -> AnalysisWindow:
    zone = ZoneInfo(timezone)
    window_start = datetime.combine(analysis_date, time.min, tzinfo=zone)
    next_date = analysis_date.fromordinal(analysis_date.toordinal() + 1)
    window_end = datetime.combine(next_date, time.min, tzinfo=zone)
    return AnalysisWindow(
        analysis_date=analysis_date,
        timezone=timezone,
        window_start=window_start,
        window_end=window_end,
    )
