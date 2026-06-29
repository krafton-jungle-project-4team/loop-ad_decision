from pydantic import BaseModel, ConfigDict


class BanditActionProbabilityItem(BaseModel):
    bandit_arm_id: int
    action_id: str
    action_type: str
    probability: float
    alpha: float
    beta: float
    impressions: int
    conversions: int
    mapping_id: int | None = None
    creative_id: int | None = None
    content_url: str | None = None

    model_config = ConfigDict(extra="forbid")


class BanditActionProbabilityResponse(BaseModel):
    bandit_policy_id: int
    items: list[BanditActionProbabilityItem]

    model_config = ConfigDict(extra="forbid")
