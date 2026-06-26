from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

JsonObject = dict[str, Any]


class RecommendationResultResponse(BaseModel):
    id: int
    project_id: str
    window_start: datetime
    window_end: datetime
    baseline_start: datetime | None = None
    baseline_end: datetime | None = None
    segment_json: JsonObject
    segment_hash: str
    status: str
    anomaly_json: JsonObject
    root_causes_json: JsonObject
    recommendations_json: JsonObject
    policy_decision_json: JsonObject
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RecommendationApproveRequest(BaseModel):
    approved_by: str
    action_ids: list[str] = Field(min_length=1)
    reason: str | None = None

    model_config = ConfigDict(extra="forbid")


class RecommendationRejectRequest(BaseModel):
    rejected_by: str
    reason: str | None = None

    model_config = ConfigDict(extra="forbid")


class RecommendationApprovalResponse(BaseModel):
    recommendation_result_id: int
    status: str
    approved_action_ids: list[str]
    created_experiment_ids: list[int]
    created_segment_ad_mapping_ids: list[int]


class RecommendationRejectionResponse(BaseModel):
    recommendation_result_id: int
    status: str
    inactivated_segment_ad_mapping_ids: list[int]
    stopped_experiment_ids: list[int]


class ActiveSegmentAdMappingResponse(BaseModel):
    mapping_id: int
    project_id: str
    segment_json: JsonObject
    segment_hash: str
    action_id: str
    action_type: str
    execution_hint_json: JsonObject
    experiment_id: int | None = None
    recommendation_result_id: int
    source: str
    status: str

    model_config = ConfigDict(from_attributes=True)
