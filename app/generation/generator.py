from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.generation.prompt_builder import GenerationPromptInput, PromptBuildResult
from app.generation.schemas import ContentChannel, missing_channel_fields


CONTENT_GENERATOR_VERSION = "dec-c3.deterministic.v2"


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
    del segment_name, prompt_result

    return GeneratedContent(
        subject=_compact(
            _variant(
                option_index,
                (
                    "이번 주말 호텔 특가를 확인해보세요",
                    "환불 가능한 호텔 객실을 만나보세요",
                    "여름 숙박 혜택이 준비됐어요",
                ),
            ),
            max_length=88,
        ),
        preheader=_compact(
            _variant(
                option_index,
                (
                    "원하는 일정에 맞는 숙소와 예약 혜택을 비교해보세요.",
                    "지금 예약 가능한 호텔 상품을 한눈에 확인하세요.",
                    "객실이 마감되기 전에 숙박 혜택을 살펴보세요.",
                ),
            ),
            max_length=96,
        ),
        body=_compact(
            _variant(
                option_index,
                (
                    "환불 가능한 객실과 여름 숙박 혜택을 지금 확인하고, "
                    "여행 일정에 맞는 호텔을 놓치지 마세요.",
                    "최근 관심을 보인 호텔 상품을 다시 살펴보고, "
                    "예약 가능한 객실을 편하게 비교해보세요.",
                    "준비된 호텔 혜택을 확인하고 원하는 날짜에 맞는 "
                    "숙소를 빠르게 예약해보세요.",
                ),
            ),
            max_length=280,
        ),
        cta="호텔 특가 보기",
        landing_url=landing_url,
    )


def _sms_content(
    *,
    segment_name: str,
    landing_url: str,
    option_index: int,
) -> GeneratedContent:
    del segment_name

    message = _variant(
        option_index,
        (
            "환불 가능한 호텔 특가가 준비되어 있어요.",
            "지금 예약 가능한 호텔 혜택을 확인해보세요.",
            "원하는 일정에 맞는 숙박 혜택을 놓치지 마세요.",
        ),
    )
    return GeneratedContent(
        message=_compact(
            f"{message} {{{{redirect_url}}}}",
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
    del segment_name, prompt_result

    return GeneratedContent(
        title=_compact(
            _variant(
                option_index,
                (
                    "이번 주말 호텔 특가",
                    "지금 예약 가능한 호텔",
                    "여름 숙박 혜택 확인",
                ),
            ),
            max_length=72,
        ),
        body=_compact(
            _variant(
                option_index,
                (
                    "환불 가능한 객실과 숙박 혜택을 지금 비교해보세요.",
                    "원하는 일정에 맞는 객실을 객실 마감 전에 확인하세요.",
                    "편하게 비교하고 바로 예약할 수 있는 호텔 혜택을 만나보세요.",
                ),
            ),
            max_length=180,
        ),
        cta="호텔 특가 보기",
        image_prompt=(
            "modern hotel room summer promotion banner, clean bright travel "
            "layout, no visible text"
        ),
        landing_url=landing_url,
    )


def _variant(option_index: int, values: tuple[str, ...]) -> str:
    return values[(option_index - 1) % len(values)]


def _compact(value: str, *, max_length: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_length:
        return compacted
    return compacted[: max_length - 1].rstrip() + "."
