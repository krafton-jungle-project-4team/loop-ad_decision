from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.generation.artifacts import ArtifactIdentity, StoredAsset
from app.generation.image_prompt_builder import RichImagePromptBuilder
from app.generation.prompt_builder import GenerationPromptInput, PromptBuildResult
from app.generation.schemas import ContentChannel, missing_channel_fields


CONTENT_GENERATOR_VERSION = "dec-c3.deterministic.v4"
DEFAULT_DETERMINISTIC_IMAGE_URL = (
    "https://gen-ai.asset.dev.loop-ad.org/fixtures/deterministic-hotel.png"
)


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
    image_artifact: StoredAsset | None = None
    landing_url: str | None = None
    artifact_renderer_version: str | None = None
    artifact_template_version: str | None = None

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
        artifact_identity: ArtifactIdentity,
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
        artifact_identity: ArtifactIdentity,
    ) -> GeneratedContent:
        del artifact_identity
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
                prompt_result=prompt_result,
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
    del segment_name

    strategy_lead = _strategy_copy_lead(prompt_result)
    body = _variant(
        option_index,
        (
            "여행 일정에 맞는 호텔과 예약 조건을 지금 확인해보세요.",
            "최근 관심을 보인 호텔 상품을 다시 살펴보고, "
            "객실 정보를 편하게 비교해보세요.",
            "원하는 날짜에 맞는 숙소 정보를 확인하고 예약을 이어가세요.",
        ),
    )
    if strategy_lead:
        body = f"{strategy_lead} {body}"

    return GeneratedContent(
        subject=_compact(
            _variant(
                option_index,
                (
                    "이번 주말 호텔 예약 정보를 확인해보세요",
                    "관심 호텔의 예약 조건을 확인해보세요",
                    "여름 숙박 정보를 확인해보세요",
                ),
            ),
            max_length=88,
        ),
        preheader=_compact(
            _variant(
                option_index,
                (
                    "원하는 일정에 맞는 숙소와 예약 조건을 비교해보세요.",
                    "관심 있는 호텔 정보를 한눈에 확인하세요.",
                    "여행 일정에 맞는 객실 정보를 살펴보세요.",
                ),
            ),
            max_length=96,
        ),
        body=_compact(
            body,
            max_length=280,
        ),
        cta="호텔 정보 보기",
        image_prompt=RichImagePromptBuilder().build(prompt_result),
        image_url=DEFAULT_DETERMINISTIC_IMAGE_URL,
        landing_url=landing_url,
    )


def _sms_content(
    *,
    segment_name: str,
    landing_url: str,
    option_index: int,
    prompt_result: PromptBuildResult,
) -> GeneratedContent:
    del segment_name

    message = _variant(
        option_index,
        (
            "관심 호텔의 예약 정보를 확인해보세요.",
            "여행 일정에 맞는 호텔 정보를 살펴보세요.",
            "원하는 날짜의 숙소와 예약 조건을 비교해보세요.",
        ),
    )
    strategy_lead = _strategy_copy_lead(prompt_result)
    if strategy_lead:
        message = f"{strategy_lead} {message}"
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
    del segment_name

    body = _variant(
        option_index,
        (
            "관심 호텔의 객실 정보와 예약 조건을 지금 비교해보세요.",
            "원하는 일정에 맞는 객실 정보를 확인해보세요.",
            "호텔 정보를 편하게 비교하고 예약을 이어가세요.",
        ),
    )
    strategy_lead = _strategy_copy_lead(prompt_result)
    if strategy_lead:
        body = f"{strategy_lead} {body}"

    return GeneratedContent(
        title=_compact(
            _variant(
                option_index,
                (
                    "이번 주말 호텔 확인",
                    "관심 호텔 예약 정보",
                    "여름 숙박 정보 확인",
                ),
            ),
            max_length=72,
        ),
        body=_compact(
            body,
            max_length=180,
        ),
        cta="호텔 정보 보기",
        image_prompt=RichImagePromptBuilder().build(prompt_result),
        image_url=DEFAULT_DETERMINISTIC_IMAGE_URL,
        landing_url=landing_url,
    )


def _strategy_copy_lead(prompt_result: PromptBuildResult) -> str | None:
    strategy_plan = prompt_result.strategy_plan
    if strategy_plan is None or not strategy_plan.evidence_refs:
        return None

    audience_label = _audience_label(strategy_plan.audience_focus)
    if strategy_plan.message_angle == "booking_confidence":
        action = "예약 결정을 돕는 정보를 안내합니다."
    elif strategy_plan.message_angle == "landing_motivation":
        action = "호텔 혜택을 확인할 이유를 분명하게 전합니다."
    else:
        action = "프로모션 내용을 알기 쉽게 전합니다."
    if audience_label:
        return f"{audience_label}에게 {action}"
    return action


def _audience_label(values: tuple[str, ...]) -> str | None:
    if not values:
        return None
    normalized = values[0].strip().lower()
    labels = (
        (("near_checkin", "checkin"), "체크인 일정이 가까운 고객"),
        (("repeat", "same_hotel"), "관심 호텔을 다시 살펴본 고객"),
        (("booking_start", "no_booking"), "예약을 완료하지 않은 고객"),
        (("mobile",), "모바일로 호텔을 찾는 고객"),
        (("redirect", "landing"), "호텔 정보를 확인한 고객"),
    )
    for terms, label in labels:
        if any(term in normalized for term in terms):
            return label
    return "선택된 호텔 고객군"


def _variant(option_index: int, values: tuple[str, ...]) -> str:
    return values[(option_index - 1) % len(values)]


def _compact(value: str, *, max_length: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_length:
        return compacted
    return compacted[: max_length - 1].rstrip() + "."
