from enum import StrEnum
from typing import Literal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Channel(StrEnum):
    EMAIL = "email"
    SMS = "sms"
    ONSITE_BANNER = "onsite_banner"


class GoalMetric(StrEnum):
    INFLOW_RATE = "inflow_rate"
    BOOKING_CONVERSION_RATE = "booking_conversion_rate"
    FUNNEL_STEP_RATE = "funnel_step_rate"


class GoalBasis(StrEnum):
    PROMOTION_AVERAGE = "promotion_average"
    ALL_SEGMENTS = "all_segments"


class PromotionRunStatus(StrEnum):
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    EVALUATING = "evaluating"
    PARTIAL_GOAL_MET = "partial_goal_met"
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    INSUFFICIENT_DATA = "insufficient_data"
    STOPPED = "stopped"


class AdExperimentStatus(StrEnum):
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    EVALUATING = "evaluating"
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    INSUFFICIENT_DATA = "insufficient_data"
    STOPPED = "stopped"


class PromotionEvaluationStatus(StrEnum):
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    PARTIAL_GOAL_MET = "partial_goal_met"
    INSUFFICIENT_DATA = "insufficient_data"


class AssignmentSource(StrEnum):
    DECISION_BATCH = "decision_batch"
    FALLBACK = "fallback"
    MANUAL = "manual"
    FIXTURE = "fixture"


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    analysis_id: str | None = Field(default=None, min_length=1)
    generation_id: str | None = Field(default=None, min_length=1)
    loop_count: int = Field(default=1, ge=1)


class AdExperimentCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ad_experiment_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    segment_name: str | None = None
    content_id: str = Field(min_length=1)
    content_option_id: str = Field(min_length=1)
    channel: Channel
    loop_count: int = Field(ge=1)
    status: AdExperimentStatus


class RunCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    promotion_run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    analysis_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    loop_count: int = Field(ge=1)
    status: PromotionRunStatus
    goal_snapshot_json: dict[str, Any]
    ad_experiments: list[AdExperimentCreateResponse]


class SegmentAssignmentBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_ids: list[str] | None = None
    eligible_user_limit: int | None = Field(default=None, ge=1)
    vector_version: str = Field(default="v1", min_length=1)
    expires_in_days: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_scope(self) -> "SegmentAssignmentBuildRequest":
        if not self.user_ids and self.eligible_user_limit is None:
            raise ValueError("eligible_user_limit is required when user_ids is omitted")
        return self


class SegmentAssignmentBuildResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    promotion_run_id: str = Field(min_length=1)
    assignment_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    insufficient_segment_count: int = Field(ge=0)
    status: Literal["completed"] = "completed"
