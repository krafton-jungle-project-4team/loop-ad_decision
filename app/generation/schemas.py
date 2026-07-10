from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class CreativeFormat(StrEnum):
    EMAIL_HTML = "email_html"
    SMS_TEXT = "sms_text"
    BANNER_HTML = "banner_html"


class ArtifactStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


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


class CreativeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    creative_format: CreativeFormat
    artifact_status: ArtifactStatus
    storage_key: str | None = None
    public_url: str | None = None
    sha256: str | None = None
    bytes: int | None = Field(default=None, ge=0)
    content_type: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    error_code: str | None = None


class LoopAdAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    promotion_run_id: str = Field(min_length=1)
    ad_experiment_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    content_id: str = Field(min_length=1)
    content_option_id: str = Field(min_length=1)
    promotion_channel: ContentChannel
    target_url: str = Field(min_length=1)
    placement_id: str | None = None
    redirect_id: str | None = None


class EmailHtmlSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    creative_format: Literal["email_html"] = "email_html"
    subject: str = Field(min_length=1)
    preheader: str = Field(min_length=1)
    text_body: str = Field(min_length=1)
    required_placeholders: tuple[str, str] = ("{{redirect_url}}", "{{open_pixel_url}}")


class SmsTextSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    creative_format: Literal["sms_text"] = "sms_text"
    message: str = Field(min_length=1)
    required_placeholders: tuple[str] = ("{{redirect_url}}",)


class BannerHtmlSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    creative_format: Literal["banner_html"] = "banner_html"
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    click_protocol: Literal["post_message"] = "post_message"
    allowed_message_type: Literal["loopad:click"] = "loopad:click"


class ContentCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    channel: ContentChannel
    creative_format: CreativeFormat
    attribution: LoopAdAttribution
    source: EmailHtmlSource | SmsTextSource | BannerHtmlSource
    artifact: CreativeArtifact


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    generation_id: str = Field(min_length=1)
    promotion_id: str = Field(min_length=1)
    status: GenerationStatus
    content_candidates: list[ContentCandidateResponse]


def missing_channel_fields(channel: ContentChannel, values: dict[str, Any]) -> list[str]:
    required_fields = CHANNEL_REQUIRED_FIELDS[channel]
    return [field for field in required_fields if not values.get(field)]
