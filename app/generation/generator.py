from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.generation.prompt_builder import GenerationPromptInput, PromptBuildResult
from app.generation.schemas import ContentChannel, missing_channel_fields


CONTENT_GENERATOR_VERSION = "dec-c3.deterministic.v1"


@dataclass(frozen=True)
class GeneratedContent:
    subject: str | None = None
    preheader: str | None = None
    title: str | None = None
    body: str | None = None
    cta: str | None = None
    message: str | None = None
    image_prompt: str | None = None
    image_url: str | None = None
    landing_url: str | None = None

    def to_record_values(self, channel: ContentChannel) -> dict[str, str | None]:
        values = {
            "subject": self.subject,
            "preheader": self.preheader,
            "title": self.title,
            "body": self.body,
            "cta": self.cta,
            "message": self.message,
            "image_prompt": self.image_prompt,
            "image_url": self.image_url,
            "landing_url": self.landing_url,
        }
        missing = missing_channel_fields(channel, values)
        if missing:
            raise ValueError(
                f"{channel.value} generated content is missing required fields: "
                f"{', '.join(missing)}"
            )
        return values


class ContentGenerator(Protocol):
    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> GeneratedContent:
        ...


class DeterministicContentGenerator:
    version = CONTENT_GENERATOR_VERSION

    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> GeneratedContent:
        channel = prompt_input.promotion.channel
        landing_url = prompt_input.promotion.landing_url
        if not landing_url:
            raise ValueError("landing_url is required to generate content")

        segment_name = prompt_input.target_segment.segment_name
        if channel == ContentChannel.EMAIL:
            return _email_content(
                segment_name=segment_name,
                landing_url=landing_url,
                option_index=option_index,
                prompt_result=prompt_result,
            )
        if channel == ContentChannel.SMS:
            return _sms_content(
                segment_name=segment_name,
                landing_url=landing_url,
                option_index=option_index,
            )
        return _banner_content(
            segment_name=segment_name,
            landing_url=landing_url,
            option_index=option_index,
            prompt_result=prompt_result,
        )


def _email_content(
    *,
    segment_name: str,
    landing_url: str,
    option_index: int,
    prompt_result: PromptBuildResult,
) -> GeneratedContent:
    subject = f"Hotel rooms picked for {segment_name}"
    return GeneratedContent(
        subject=_compact(subject, max_length=88),
        preheader=_compact(
            f"Option {option_index}: refundable stays and clear booking steps.",
            max_length=96,
        ),
        body=_compact(
            f"{prompt_result.message_strategy} Invite {segment_name} to compare "
            "available hotel stays before rooms fill.",
            max_length=280,
        ),
        cta="View hotel deals",
        landing_url=landing_url,
    )


def _sms_content(
    *,
    segment_name: str,
    landing_url: str,
    option_index: int,
) -> GeneratedContent:
    return GeneratedContent(
        message=_compact(
            f"Option {option_index}: {segment_name}, refundable hotel deals are "
            f"available now. {landing_url}",
            max_length=220,
        ),
        landing_url=landing_url,
    )


def _banner_content(
    *,
    segment_name: str,
    landing_url: str,
    option_index: int,
    prompt_result: PromptBuildResult,
) -> GeneratedContent:
    return GeneratedContent(
        title=_compact(
            f"Hotel stays ready for {segment_name}",
            max_length=72,
        ),
        body=_compact(
            f"Option {option_index}: {prompt_result.message_strategy}",
            max_length=180,
        ),
        cta="View hotel deals",
        image_prompt=(
            "modern hotel room summer promotion banner, clean bright travel layout"
        ),
        landing_url=landing_url,
    )


def _compact(value: str, *, max_length: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_length:
        return compacted
    return compacted[: max_length - 1].rstrip() + "."
