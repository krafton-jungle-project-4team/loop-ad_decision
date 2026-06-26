from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.actions.schemas import RecommendedAction
from app.automation.schemas import PolicyDecision
from app.root_causes.schemas import RootCauseAnalysisRequest, RootCauseCandidate


class FunnelRecommendationAnalysisRequest(RootCauseAnalysisRequest):
    top_n: int = Field(default=5, ge=1, le=20)


class BlockedActionSummary(BaseModel):
    action_id: str
    reasons: list[str] = Field(default_factory=list)


class FunnelRecommendationAnalysisResponse(BaseModel):
    recommendation_result_id: int
    status: str
    anomaly_summary: str
    root_cause_candidates: list[RootCauseCandidate]
    recommended_actions: list[RecommendedAction]
    auto_executed_action_ids: list[str]
    blocked_actions: list[BlockedActionSummary]
    created_experiment_ids: list[int]
    created_segment_ad_mapping_ids: list[int]
    policy_decision: PolicyDecision | None = None

    model_config = ConfigDict(extra="forbid")


class AnalysisJobCreateResponse(BaseModel):
    job_id: int
    status: str
    recommendation_result_id: int | None = None
    polling_url: str

    model_config = ConfigDict(extra="forbid")


class AnalysisJobStatusResponse(BaseModel):
    job_id: int
    status: str
    recommendation_result_id: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    model_config = ConfigDict(extra="forbid")
