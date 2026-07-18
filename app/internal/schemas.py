from __future__ import annotations

from datetime import datetime
from typing import Literal

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
    vector_generation_id: str
    expected_user_count: int
    manifest_hash: str
    window_start: datetime
    window_end: datetime
    status: str


class UserBehaviorVectorSearchSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=100)
    vector_generation_id: str = Field(min_length=1, max_length=120)
    vector_version: str = Field(
        default="hotel_behavior.v2",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    batch_size: int = Field(default=5000, ge=1, le=20000)
    max_batches: int = Field(default=10, ge=1, le=100)


class UserBehaviorVectorSearchSyncResponse(BaseModel):
    project_id: str
    vector_version: str
    vector_generation_id: str
    synced_user_count: int
    expected_user_count: int
    active_generation_id: str | None
    source_cutoff: datetime | None
    status: Literal["in_progress", "activated", "failed"]
