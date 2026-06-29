from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class AnalysisWindow:
    analysis_date: date
    timezone: str
    window_start: datetime
    window_end: datetime


@dataclass(frozen=True)
class AnalysisResult:
    segment_count: int = 0
    membership_count: int = 0
    metric_count: int = 0
    anomaly_count: int = 0
    root_cause_count: int = 0
    anomaly_segment_ids: list[int] = field(default_factory=list)
    anomaly_ids: list[int] = field(default_factory=list)
