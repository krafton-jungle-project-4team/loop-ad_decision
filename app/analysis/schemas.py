from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Channel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    ONSITE_BANNER = "onsite_banner"


class GoalMetric(str, Enum):
    INFLOW_RATE = "inflow_rate"
    BOOKING_CONVERSION_RATE = "booking_conversion_rate"
    FUNNEL_STEP_RATE = "funnel_step_rate"


class GoalBasis(str, Enum):
    PROMOTION_AVERAGE = "promotion_average"
    ALL_SEGMENTS = "all_segments"


class AnalysisStatus(str, Enum):
    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str
    campaign_id: str
    promotion_id: str
    operator_instruction: str | None = None


class SegmentAnalysisRequest(AnalysisRequest):
    segment_ids: list[str] = Field(min_length=1)

    @field_validator("segment_ids")
    @classmethod
    def validate_segment_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("segment_ids must not contain empty values")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("segment_ids must not contain duplicates")
        return cleaned

class ContentBriefResponse(BaseModel):
    message_direction: str
    keywords: list[str]


class TargetSegmentResponse(BaseModel):
    segment_id: str
    segment_name: str
    segment_vector_id: str
    estimated_size: int
    audience_snapshot_id: str | None = None
    eligible_user_count: int | None = None
    behavior_match_count: int | None = None
    final_audience_count: int | None = None
    meets_min_sample_size: bool | None = None
    targetable: bool | None = None
    audience_status: Literal[
        "targetable",
        "insufficient_sample",
        "no_eligible_audience",
    ] | None = None
    selection_method: str | None = None
    recall_lower_bound: float | None = None
    content_brief: ContentBriefResponse


class AnalysisResponse(BaseModel):
    analysis_id: str
    promotion_id: str
    status: AnalysisStatus
    target_segments: list[TargetSegmentResponse]
