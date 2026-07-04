from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContentChannel(StrEnum):
    EMAIL = "email"
    SMS = "sms"
    ONSITE_BANNER = "onsite_banner"


class GenerationStatus(StrEnum):
    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentCandidateStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    ARCHIVED = "archived"


CHANNEL_REQUIRED_FIELDS: dict[ContentChannel, tuple[str, ...]] = {
    ContentChannel.EMAIL: ("subject", "preheader", "body", "cta", "landing_url"),
    ContentChannel.SMS: ("message", "landing_url"),
    ContentChannel.ONSITE_BANNER: (
        "title",
        "body",
        "cta",
        "image_prompt",
        "landing_url",
    ),
}


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    analysis_id: str = Field(min_length=1)
    content_option_count: int = Field(ge=1)
    operator_instruction: str | None = None


class ContentCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content_id: str = Field(min_length=1)
    content_option_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    channel: ContentChannel
    subject: str | None = None
    preheader: str | None = None
    title: str | None = None
    body: str | None = None
    cta: str | None = None
    message: str | None = None
    image_prompt: str | None = None
    image_url: str | None = None
    landing_url: str | None = None
    status: ContentCandidateStatus = ContentCandidateStatus.DRAFT

    @model_validator(mode="after")
    def validate_channel_fields(self) -> "ContentCandidateResponse":
        missing = missing_channel_fields(self.channel, self.model_dump())
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                f"{self.channel.value} content candidate is missing required fields: "
                f"{missing_fields}"
            )
        return self


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    generation_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    status: GenerationStatus
    content_candidates: list[ContentCandidateResponse]


def missing_channel_fields(channel: ContentChannel, values: dict[str, Any]) -> list[str]:
    required_fields = CHANNEL_REQUIRED_FIELDS[channel]
    return [field for field in required_fields if not values.get(field)]
