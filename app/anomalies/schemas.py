from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.metrics.schemas import FunnelMetricFilters, FunnelMetrics

FunnelAnomalyStatus = Literal["normal", "warning", "critical", "insufficient_data"]
ALLOWED_SEGMENT_FIELDS = {
    "channel",
    "campaign_id",
    "age_group",
    "gender",
    "device",
    "category",
    "product_id",
    "inventory_status",
}


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
            if self.baseline_end > self.window_start:
                raise ValueError("baseline_end must be earlier than or equal to window_start")

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


class VolumeAnomalyEvaluation(BaseModel):
    metric: str
    severity: FunnelAnomalyStatus
    current_value: int
    baseline_value: int
    delta: int
    relative_change: float | None
    drop: int
    relative_drop: float | None
    min_volume_count: int
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
    volume_evaluations: list[VolumeAnomalyEvaluation] = Field(default_factory=list)
    volume_anomalies: list[VolumeAnomalyEvaluation] = Field(default_factory=list)
    primary_anomaly: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation | None = None
    summary_message: str


class SegmentFunnelAnomalyRequest(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    baseline_start: datetime | None = None
    baseline_end: datetime | None = None
    base_filters: FunnelMetricFilters | None = None
    segment_by: list[str] = Field(min_length=1, max_length=2)
    limit: int = Field(default=10, ge=1, le=50)
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
    def validate_segment_anomaly_request(self) -> "SegmentFunnelAnomalyRequest":
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

        if len(set(self.segment_by)) != len(self.segment_by):
            raise ValueError("segment_by must not contain duplicate fields")

        unknown_fields = [field for field in self.segment_by if field not in ALLOWED_SEGMENT_FIELDS]
        if unknown_fields:
            raise ValueError("segment_by contains unsupported fields")

        base_filter_values = (
            self.base_filters.model_dump(exclude_none=True) if self.base_filters else {}
        )
        duplicated_filters = [field for field in self.segment_by if field in base_filter_values]
        if duplicated_filters:
            raise ValueError("segment_by must not include fields already set in base_filters")

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


class SegmentFunnelAnomalyResult(BaseModel):
    segment: dict[str, str | None]
    status: FunnelAnomalyStatus
    score: float
    primary_anomaly: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation | None
    summary_message: str
    current_metrics: FunnelMetrics
    baseline_metrics: FunnelMetrics
    evaluations: list[FunnelAnomalyEvaluation]
    anomalies: list[FunnelAnomalyEvaluation]
    volume_evaluations: list[VolumeAnomalyEvaluation]
    volume_anomalies: list[VolumeAnomalyEvaluation]


class SegmentFunnelAnomalyResponse(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    baseline_start: datetime
    baseline_end: datetime
    base_segment: dict[str, str | None]
    segment_by: list[str]
    status: FunnelAnomalyStatus
    total_segments_discovered: int
    total_segments_evaluated: int
    segments: list[SegmentFunnelAnomalyResult]
    summary_message: str
