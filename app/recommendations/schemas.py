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
    bandit_decision_summary_json: JsonObject = Field(default_factory=dict)
    summary_message: str | None = None
    actions: list["RecommendationActionResponse"] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RecommendationActionResponse(BaseModel):
    id: int
    project_id: str
    recommendation_result_id: int
    action_id: str
    action_type: str
    title: str | None = None
    description: str | None = None
    target_step: str | None = None
    priority_score: float | None = None
    expected_impact: str | None = None
    rationale: str | None = None
    triggered_by_json: list[Any] = Field(default_factory=list)
    execution_hint_json: JsonObject = Field(default_factory=dict)
    experiment_json: JsonObject = Field(default_factory=dict)
    policy_status: str | None = None
    policy_reasons_json: list[Any] = Field(default_factory=list)
    policy_decision_json: JsonObject = Field(default_factory=dict)
    selected_by_strategy: str = "rule_based"
    bandit_policy_id: int | None = None
    bandit_arm_id: int | None = None
    sampled_value: float | None = None
    status: str
    auto_executed_at: datetime | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_reason: str | None = None
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
    action_ids: list[str] | None = None
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
    rejected_action_ids: list[str] = Field(default_factory=list)
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
    recommendation_action_id: int
    bandit_policy_id: int | None = None
    bandit_arm_id: int | None = None
    bandit_decision_id: int | None = None
    campaign_id: int | None = None
    creative_id: int | None = None
    coupon_id: int | None = None
    content_url: str | None = None
    title: str | None = None
    message: str | None = None
    landing_url: str | None = None
    serving_weight: float | None = None
    source: str
    status: str

    model_config = ConfigDict(from_attributes=True)
