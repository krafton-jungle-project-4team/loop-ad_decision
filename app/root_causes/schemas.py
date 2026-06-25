from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.metrics.repository import FILTER_COLUMNS
from app.metrics.schemas import FunnelMetricFilters

RootCauseSeverity = Literal["normal", "warning", "critical", "insufficient_data"]
ALLOWED_CANDIDATE_DIMENSIONS = set(FILTER_COLUMNS)


class RootCauseAnalysisRequest(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    baseline_start: datetime | None = None
    baseline_end: datetime | None = None
    filters: FunnelMetricFilters | None = None
    candidate_dimensions: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=20)
    candidate_limit: int = Field(default=100, ge=1, le=500)
    min_sample_size: int = Field(default=100, ge=1)
    warning_abs_drop: float = Field(default=0.05, ge=0, le=1)
    critical_abs_drop: float = Field(default=0.10, ge=0, le=1)
    warning_relative_drop: float = Field(default=0.30, ge=0, le=1)
    critical_relative_drop: float = Field(default=0.50, ge=0, le=1)
    include_volume_anomalies: bool = True
    min_volume_count: int = Field(default=30, ge=1)
    warning_volume_relative_drop: float = Field(default=0.30, ge=0, le=1)
    critical_volume_relative_drop: float = Field(default=0.50, ge=0, le=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("window_start", "window_end", "baseline_start", "baseline_end")
    @classmethod
    def validate_timezone_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime values must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_root_cause_request(self) -> "RootCauseAnalysisRequest":
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be earlier than window_end")

        has_baseline_start = self.baseline_start is not None
        has_baseline_end = self.baseline_end is not None
        if has_baseline_start != has_baseline_end:
            raise ValueError("baseline_start and baseline_end must be provided together")

        if self.baseline_start is not None and self.baseline_end is not None:
            if self.baseline_start >= self.baseline_end:
                raise ValueError("baseline_start must be earlier than baseline_end")
            if self.baseline_end > self.window_start:
                raise ValueError("baseline_end must be earlier than or equal to window_start")

        if self.candidate_dimensions is not None:
            unknown_dimensions = [
                dimension
                for dimension in self.candidate_dimensions
                if dimension not in ALLOWED_CANDIDATE_DIMENSIONS
            ]
            if unknown_dimensions:
                raise ValueError("candidate_dimensions contains unsupported fields")

        if self.warning_abs_drop > self.critical_abs_drop:
            raise ValueError("warning_abs_drop must be less than or equal to critical_abs_drop")
        if self.warning_relative_drop > self.critical_relative_drop:
            raise ValueError("warning_relative_drop must be less than or equal to critical_relative_drop")
        if self.warning_volume_relative_drop > self.critical_volume_relative_drop:
            raise ValueError(
                "warning_volume_relative_drop must be less than or equal to "
                "critical_volume_relative_drop"
            )

        return self


class RootCauseTargetAnomaly(BaseModel):
    metric: str
    funnel_step: str | None
    severity: str
    current_value: float | int | None
    baseline_value: float | int | None
    drop_point: float | None = None
    relative_drop: float | None = None
    message: str


class RootCauseCandidate(BaseModel):
    rank: int
    cause_type: str
    dimension: str
    value: str
    metric: str
    funnel_step: str | None
    severity: str
    current_value: float | int | None
    baseline_value: float | int | None
    drop_point: float | None
    relative_drop: float | None
    current_denominator: int
    baseline_denominator: int
    support_share: float
    excess_lost_sessions: float
    score: float
    message: str


class RootCauseAnalysisResponse(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    baseline_start: datetime
    baseline_end: datetime
    segment: dict[str, str | None]
    status: str
    target_anomaly: RootCauseTargetAnomaly | None
    total_candidates_evaluated: int
    candidates: list[RootCauseCandidate]
    summary_message: str
