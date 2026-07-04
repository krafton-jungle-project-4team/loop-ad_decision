from enum import Enum

from pydantic import BaseModel, ConfigDict


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
    focus_segment_ids: list[str] | None = None
    operator_instruction: str | None = None


class ContentBriefResponse(BaseModel):
    message_direction: str
    keywords: list[str]


class TargetSegmentResponse(BaseModel):
    segment_id: str
    segment_name: str
    segment_vector_id: str
    estimated_size: int
    content_brief: ContentBriefResponse


class AnalysisResponse(BaseModel):
    analysis_id: str
    promotion_id: str
    status: AnalysisStatus
    target_segments: list[TargetSegmentResponse]
