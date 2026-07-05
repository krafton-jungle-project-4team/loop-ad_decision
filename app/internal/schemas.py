from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserBehaviorVectorBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=100)
    vector_version: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    window_days: int = Field(gt=0, le=3650)


class UserBehaviorVectorBuildResponse(BaseModel):
    project_id: str
    vector_version: str
    source: str
    vector_dim: int
    processed_user_count: int
    window_start: datetime
    window_end: datetime
    status: str
