from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.metrics.schemas import FunnelMetricFilters, FunnelMetrics

FunnelAnomalyStatus = Literal["normal", "warning", "critical", "insufficient_data"]


class FunnelAnomalyRequest(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    baseline_start: datetime | None = None
    baseline_end: datetime | None = None
    filters: FunnelMetricFilters | None = None
    min_sample_size: int = Field(default=100, ge=1)
    warning_abs_drop: float = Field(default=0.05, ge=0, le=1)
    critical_abs_drop: float = Field(default=0.10, ge=0, le=1)
    warning_relative_drop: float = Field(default=0.30, ge=0, le=1)
    critical_relative_drop: float = Field(default=0.50, ge=0, le=1)

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
    def validate_anomaly_request(self) -> "FunnelAnomalyRequest":
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be earlier than window_end")

        has_baseline_start = self.baseline_start is not None
        has_baseline_end = self.baseline_end is not None
        if has_baseline_start != has_baseline_end:
            raise ValueError("baseline_start and baseline_end must be provided together")

        if self.baseline_start is not None and self.baseline_end is not None:
            if self.baseline_start >= self.baseline_end:
                raise ValueError("baseline_start must be earlier than baseline_end")

        if self.warning_abs_drop > self.critical_abs_drop:
            raise ValueError("warning_abs_drop must be less than or equal to critical_abs_drop")
        if self.warning_relative_drop > self.critical_relative_drop:
            raise ValueError("warning_relative_drop must be less than or equal to critical_relative_drop")

        return self


class FunnelAnomalyEvaluation(BaseModel):
    metric: str
    funnel_step: str
    severity: FunnelAnomalyStatus
    current_value: float | None
    baseline_value: float | None
    delta_point: float | None
    relative_change: float | None
    drop_point: float | None
    relative_drop: float | None
    current_denominator: int
    baseline_denominator: int
    min_sample_size: int
    message: str


class FunnelAnomalyResponse(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    baseline_start: datetime
    baseline_end: datetime
    segment: dict[str, str | None]
    status: FunnelAnomalyStatus
    current_metrics: FunnelMetrics
    baseline_metrics: FunnelMetrics
    evaluations: list[FunnelAnomalyEvaluation]
    anomalies: list[FunnelAnomalyEvaluation]
