from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonObject = dict[str, Any]


class AutomationPolicyUpsertRequest(BaseModel):
    enabled: bool = False
    auto_execute_enabled: bool = False
    allowed_action_ids: list[str] = Field(default_factory=list)
    allowed_action_types: list[str] = Field(default_factory=list)
    blocked_action_ids: list[str] = Field(default_factory=list)
    max_experiment_traffic_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    min_priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_discount_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_daily_coupon_budget: float | None = Field(default=None, ge=0.0)
    max_message_per_user_per_day: int | None = Field(default=None, ge=1)
    stop_loss_relative_drop: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("allowed_action_ids", "allowed_action_types", "blocked_action_ids")
    @classmethod
    def dedupe_values(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class AutomationPolicyResponse(AutomationPolicyUpsertRequest):
    id: int
    project_id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class TrafficSplit(BaseModel):
    control: float
    treatment: float


class ActionPolicyDecision(BaseModel):
    action_id: str
    action_type: str
    status: Literal["auto_executed", "blocked"]
    allowed: bool
    blocked: bool
    auto_executed: bool
    reasons: list[str] = Field(default_factory=list)
    traffic_split: TrafficSplit | None = None
    metadata: JsonObject = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    policy_id: int | None = None
    auto_execute_enabled: bool
    actions: list[ActionPolicyDecision]
