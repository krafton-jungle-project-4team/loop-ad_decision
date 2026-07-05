from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class UserBehaviorVectorSource(str, Enum):
    EXPEDIA_HOTEL_EVENTS = "expedia_hotel_events"


class UserBehaviorVectorBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=100)
    vector_version: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    source: UserBehaviorVectorSource = UserBehaviorVectorSource.EXPEDIA_HOTEL_EVENTS
    window_days: int = Field(gt=0, le=3650)


class UserBehaviorVectorBuildResponse(BaseModel):
    project_id: str
    vector_version: str
    source: UserBehaviorVectorSource
    vector_dim: int
    processed_user_count: int
    window_start: datetime
    window_end: datetime
    status: str
