from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol
from urllib.parse import urlsplit

from app.generation.email_variants import (
    COMPARISON_VARIANT,
    EDITORIAL_VARIANT,
    OFFER_CARDS_VARIANT,
    PRICE_DISPLAY_ALL_TIERS,
    PRICE_DISPLAY_PROMOTION_AND_FINAL,
    email_required_placeholders,
)
from app.generation.schemas import ContentChannel, CreativeFormat


HTML_CONTENT_TYPE = "text/html; charset=utf-8"
DEFAULT_BANNER_WIDTH = 320
DEFAULT_BANNER_HEIGHT = 100
DEFAULT_GENAI_PUBLIC_BASE_URL = "https://gen-ai.asset.dev.loop-ad.org"
MAX_CONTRACT_ID_LENGTH = 100
RENDERER_VERSION = "generation.renderer.v1"
EMAIL_TEMPLATE_VERSION = "email.promotion.v1"
BANNER_TEMPLATE_VERSION = "banner.overlay.v1"
LEGACY_RENDERER_VERSION = "generation.renderer.v1"
LEGACY_EMAIL_TEMPLATE_VERSION = "email.promotion.v1"
LEGACY_BANNER_TEMPLATE_VERSION = "banner.overlay.v1"
CREATIVE_SOURCE_SCHEMA_VERSION = "creative.source.v3"
PREVIOUS_CREATIVE_SOURCE_SCHEMA_VERSION = "creative.source.v2"
LEGACY_CREATIVE_SOURCE_SCHEMA_VERSION = "creative.source.v1"
CREATIVE_SOURCE_META_NAME = "loopad:creative-source"
RECOVERED_IMAGE_PROMPT_PREFIX = "recovered-sha256:"
CREATIVE_SOURCE_FIELDS = (
    "subject",
    "preheader",
    "title",
    "body",
    "cta",
    "image_url",
)
CREATIVE_CONTRACT_FIELDS = (
    "variant_type",
    "link_targets",
    "hero_image_url",
    "offers",
    "featured_offers",
    "comparison_offers",
    "catalog",
    "price_display_mode",
)
_CREATIVE_SOURCE_META_PATTERN = re.compile(
    rf'<meta name="{re.escape(CREATIVE_SOURCE_META_NAME)}" content="([A-Za-z0-9_-]+)">'
)
_CREATIVE_NESTED_OWNED_FIELDS = {
    "source": frozenset(
        {
            "creative_format",
            "subject",
            "preheader",
            "text_body",
            "message",
            "required_placeholders",
            "width",
            "height",
            "click_protocol",
            "allowed_message_type",
        }
    ),
    "renderer": frozenset({"version", "template_version"}),
    "artifact": frozenset(
        {
            "creative_format",
            "artifact_status",
            "storage_key",
            "public_url",
            "sha256",
            "bytes",
            "content_type",
            "width",
            "height",
            "error_code",
            "published_at",
        }
    ),
    "image": frozenset(
        {
            "prompt",
            "prompt_sha256",
            "prompt_recovered",
            "storage_key",
            "public_url",
            "sha256",
            "byte_size",
            "content_type",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    project_id: str
    promotion_id: str
    generation_id: str
    content_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "project_id",
            "promotion_id",
            "generation_id",
            "content_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"artifact identity requires {field_name}")


@dataclass(frozen=True, slots=True)
class StoredAsset:
    storage_key: str
    public_url: str
    sha256: str
    bytes: int
    content_type: str
    renderer_version: str | None = None
    template_version: str | None = None

    def __post_init__(self) -> None:
        _validated_sha256(self.sha256)
        if self.bytes < 0:
            raise ValueError("stored asset bytes must not be negative")
        if not self.storage_key or not self.public_url or not self.content_type:
            raise ValueError("stored asset metadata is incomplete")
        if bool(self.renderer_version) != bool(self.template_version):
            raise ValueError("stored HTML renderer provenance is incomplete")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "storage_key": self.storage_key,
            "public_url": self.public_url,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "content_type": self.content_type,
        }


@dataclass(frozen=True)
class CreativeArtifactPublication(Mapping[str, Any]):
    artifact: Mapping[str, Any]
    renderer: Mapping[str, str]

    def __getitem__(self, key: str) -> Any:
        return self.artifact[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.artifact)

    def __len__(self) -> int:
        return len(self.artifact)


class ArtifactRenderError(ValueError):
    """A deterministic renderer or hard-gate failure."""


class HtmlArtifactStorage(Protocol):
    def store_html(
        self,
        *,
        identity: ArtifactIdentity,
        creative_format: CreativeFormat,
        html_body: str,
    ) -> StoredAsset:
        ...


class CreativeArtifactPublisher(Protocol):
    def publish(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        content_values: Mapping[str, str | None],
    ) -> CreativeArtifactPublication | dict[str, Any]:
        ...


@dataclass(frozen=True)
class StaticCreativeArtifactPublisher:
    public_base_url: str = DEFAULT_GENAI_PUBLIC_BASE_URL
    base_prefix: str = "genai/"

    def publish(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        content_values: Mapping[str, str | None],
    ) -> CreativeArtifactPublication:
        creative_format = creative_format_for_channel(channel)
        if creative_format == CreativeFormat.SMS_TEXT:
            return CreativeArtifactPublication(
                artifact=not_required_artifact(creative_format),
                renderer=renderer_metadata_for_content(
                    channel=channel,
                    content_values=content_values,
                ),
            )

        html_body = render_creative_html(
            channel=channel,
            content_values=content_values,
        )
        storage_key = html_artifact_key(
            base_prefix=self.base_prefix,
            identity=identity,
            creative_format=creative_format,
        )
        artifact = html_artifact_metadata(
            creative_format=creative_format,
            storage_key=storage_key,
            public_url=public_asset_url(
                public_base_url=self.public_base_url,
                base_prefix=self.base_prefix,
                key=storage_key,
            ),
            html_body=html_body,
        )
        return CreativeArtifactPublication(
            artifact=artifact,
            renderer=renderer_metadata_for_content(
                channel=channel,
                content_values=content_values,
            ),
        )


@dataclass(frozen=True)
class S3CreativeArtifactPublisher:
    storage: HtmlArtifactStorage

    def publish(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        content_values: Mapping[str, str | None],
    ) -> CreativeArtifactPublication:
        creative_format = creative_format_for_channel(channel)
        if creative_format == CreativeFormat.SMS_TEXT:
            return CreativeArtifactPublication(
                artifact=not_required_artifact(creative_format),
                renderer=renderer_metadata_for_content(
                    channel=channel,
                    content_values=content_values,
                ),
            )

        stored = self.storage.store_html(
            identity=identity,
            creative_format=creative_format,
            html_body=render_creative_html(
                channel=channel,
                content_values=content_values,
            ),
        )

        return CreativeArtifactPublication(
            artifact={
                "creative_format": creative_format.value,
                "artifact_status": "published",
                **stored.to_metadata(),
                **artifact_dimensions(creative_format),
            },
            renderer={
                "version": stored.renderer_version or RENDERER_VERSION,
                "template_version": (
                    stored.template_version
                    or template_version_for_channel(channel)
                ),
            },
        )


def creative_format_for_channel(channel: ContentChannel) -> CreativeFormat:
    if channel == ContentChannel.EMAIL:
        return CreativeFormat.EMAIL_HTML
    if channel == ContentChannel.SMS:
        return CreativeFormat.SMS_TEXT
    return CreativeFormat.BANNER_HTML


def source_for_channel(
    *,
    channel: ContentChannel,
    content_values: Mapping[str, str | None],
) -> dict[str, Any]:
    creative_format = creative_format_for_channel(channel)
    if channel == ContentChannel.EMAIL:
        return {
            "creative_format": creative_format.value,
            "subject": required_value(content_values, "subject"),
            "preheader": required_value(content_values, "preheader"),
            "text_body": required_value(content_values, "body"),
            "required_placeholders": list(
                email_required_placeholders(content_values)
            ),
        }
    if channel == ContentChannel.SMS:
        return {
            "creative_format": creative_format.value,
            "message": required_value(content_values, "message"),
            "required_placeholders": ["{{redirect_url}}"],
        }
    return {
        "creative_format": creative_format.value,
        "width": DEFAULT_BANNER_WIDTH,
        "height": DEFAULT_BANNER_HEIGHT,
        "click_protocol": "post_message",
        "allowed_message_type": "loopad:click",
    }


def attribution_for_candidate(
    *,
    project_id: str,
    campaign_id: str,
    promotion_id: str,
    segment_id: str,
    content_id: str,
    content_option_id: str,
    channel: ContentChannel,
    target_url: str,
) -> dict[str, Any]:
    promotion_run_id = provisional_promotion_run_id(promotion_id)
    attribution = {
        "project_id": project_id,
        "campaign_id": campaign_id,
        "promotion_id": promotion_id,
        "promotion_run_id": promotion_run_id,
        "ad_experiment_id": provisional_ad_experiment_id(
            promotion_run_id=promotion_run_id,
            segment_id=segment_id,
        ),
        "segment_id": segment_id,
        "content_id": content_id,
        "content_option_id": content_option_id,
        "promotion_channel": channel.value,
        "target_url": target_url,
    }
    if channel == ContentChannel.ONSITE_BANNER:
        attribution["placement_id"] = "default"
    return attribution


def build_creative_metadata(
    *,
    channel: ContentChannel,
    identity: ArtifactIdentity,
    content_values: Mapping[str, str | None],
    artifact_publisher: CreativeArtifactPublisher,
) -> dict[str, Any]:
    publication = artifact_publisher.publish(
        identity=identity,
        channel=channel,
        content_values=content_values,
    )
    if isinstance(publication, CreativeArtifactPublication):
        artifact = dict(publication.artifact)
        renderer = dict(publication.renderer)
    else:
        artifact = dict(publication)
        renderer = renderer_metadata_for_content(
            channel=channel,
            content_values=content_values,
        )
    return {
        "creative_format": creative_format_for_channel(channel).value,
        "source": source_for_channel(channel=channel, content_values=content_values),
        "renderer": renderer,
        "artifact": artifact,
    }


def merge_creative_metadata(
    metadata_json: Mapping[str, Any],
    creative_patch: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace service-owned fields without dropping contract extensions."""

    existing_creative = metadata_json.get("creative")
    merged_creative = (
        dict(existing_creative) if isinstance(existing_creative, Mapping) else {}
    )
    for field_name, patch_value in creative_patch.items():
        owned_fields = _CREATIVE_NESTED_OWNED_FIELDS.get(field_name)
        existing_value = merged_creative.get(field_name)
        if (
            owned_fields is not None
            and isinstance(existing_value, Mapping)
            and isinstance(patch_value, Mapping)
        ):
            preserved_extensions = {
                key: value
                for key, value in existing_value.items()
                if key not in owned_fields
            }
            merged_creative[field_name] = {
                **preserved_extensions,
                **dict(patch_value),
            }
        else:
            merged_creative[field_name] = patch_value
    return {
        **dict(metadata_json),
        "creative": merged_creative,
    }


def pending_creative_metadata(
    *,
    channel: ContentChannel,
    content_values: Mapping[str, str | None],
) -> dict[str, Any]:
    return {
        "creative_format": creative_format_for_channel(channel).value,
        "source": source_for_channel(channel=channel, content_values=content_values),
        "renderer": renderer_metadata_for_content(
            channel=channel,
            content_values=content_values,
        ),
        "artifact": default_artifact(channel),
    }


def failed_creative_metadata(
    *,
    channel: ContentChannel,
    content_values: Mapping[str, str | None],
    error_code: str,
) -> dict[str, Any]:
    creative_format = creative_format_for_channel(channel)
    return {
        "creative_format": creative_format.value,
        "source": source_for_channel(channel=channel, content_values=content_values),
        "renderer": renderer_metadata_for_content(
            channel=channel,
            content_values=content_values,
        ),
        "artifact": failed_artifact(creative_format, error_code=error_code),
    }


def renderer_metadata(channel: ContentChannel) -> dict[str, str]:
    return {
        "version": RENDERER_VERSION,
        "template_version": template_version_for_channel(channel),
    }


def renderer_metadata_for_content(
    *,
    channel: ContentChannel,
    content_values: Mapping[str, str | None],
) -> dict[str, str]:
    return {
        "version": _snapshot_version(
            content_values.get("renderer_version"),
            default=RENDERER_VERSION,
        ),
        "template_version": _snapshot_version(
            content_values.get("template_version"),
            default=template_version_for_channel(channel),
        ),
    }


def template_version_for_channel(channel: ContentChannel) -> str:
    if channel == ContentChannel.EMAIL:
        return EMAIL_TEMPLATE_VERSION
    if channel == ContentChannel.ONSITE_BANNER:
        return BANNER_TEMPLATE_VERSION
    return "not_required"


def legacy_template_version_for_channel(channel: ContentChannel) -> str:
    if channel == ContentChannel.EMAIL:
        return LEGACY_EMAIL_TEMPLATE_VERSION
    if channel == ContentChannel.ONSITE_BANNER:
        return LEGACY_BANNER_TEMPLATE_VERSION
    return "not_required"


def default_artifact(channel: ContentChannel) -> dict[str, Any]:
    creative_format = creative_format_for_channel(channel)
    if creative_format == CreativeFormat.SMS_TEXT:
        return not_required_artifact(creative_format)
    return {
        "creative_format": creative_format.value,
        "artifact_status": "pending",
    }


def not_required_artifact(creative_format: CreativeFormat) -> dict[str, Any]:
    return {
        "creative_format": creative_format.value,
        "artifact_status": "not_required",
    }


def failed_artifact(creative_format: CreativeFormat, *, error_code: str) -> dict[str, Any]:
    return {
        "creative_format": creative_format.value,
        "artifact_status": "failed",
        "error_code": error_code,
    }


def render_creative_html(
    *,
    channel: ContentChannel,
    content_values: Mapping[str, str | None],
) -> str:
    if channel == ContentChannel.EMAIL:
        return render_email_html(content_values)
    if channel == ContentChannel.ONSITE_BANNER:
        return render_banner_html(content_values)
    raise ValueError("SMS does not have an HTML creative artifact")


def render_email_html(content_values: Mapping[str, str | None]) -> str:
    variant_type = content_values.get("variant_type")
    if variant_type == EDITORIAL_VARIANT:
        return render_editorial_email(content_values)
    if variant_type == OFFER_CARDS_VARIANT:
        return render_offer_cards_email(content_values)
    if variant_type == COMPARISON_VARIANT:
        return render_comparison_email(content_values)
    subject = html.escape(required_value(content_values, "subject"))
    preheader = html.escape(required_value(content_values, "preheader"))
    body = html.escape(required_value(content_values, "body"))
    cta = html.escape(str(content_values.get("cta") or "Open"))
    image_url = escaped_absolute_image_url(content_values)
    html_body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  {_creative_source_meta(ContentChannel.EMAIL, content_values)}",
            f"  <title>{subject}</title>",
            "</head>",
            '<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,Helvetica,sans-serif;color:#111827;">',
            f'  <div style="display:none;max-height:0;max-width:0;overflow:hidden;opacity:0;color:transparent;">{preheader}</div>',
            '  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:collapse;background:#f3f4f6;">',
            "    <tr>",
            '      <td align="center" style="padding:24px 12px;">',
            '        <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="width:100%;max-width:600px;border-collapse:collapse;background:#ffffff;border-radius:12px;overflow:hidden;">',
            "          <tr>",
            '            <td style="padding:16px 24px 10px;font-size:12px;line-height:18px;color:#6b7280;">광고</td>',
            "          </tr>",
            "          <tr>",
            '            <td style="padding:0;">',
            f'              <img src="{image_url}" width="600" alt="{subject}" style="display:block;width:100%;max-width:600px;height:auto;border:0;line-height:100%;outline:none;text-decoration:none;" />',
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:28px 32px 32px;">',
            f'              <h1 style="margin:0 0 14px;font-size:28px;line-height:36px;color:#111827;">{subject}</h1>',
            f'              <p style="margin:0 0 24px;font-size:16px;line-height:26px;color:#374151;">{body}</p>',
            '              <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="border-collapse:collapse;">',
            "                <tr>",
            '                  <td bgcolor="#0F55C8" style="border-radius:6px;">',
            f'                    <a href="{{{{redirect_url}}}}" style="display:inline-block;padding:13px 22px;color:#ffffff;font-size:15px;line-height:20px;font-weight:700;text-decoration:none;">{cta}</a>',
            "                  </td>",
            "                </tr>",
            "              </table>",
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:18px 24px;border-top:1px solid #e5e7eb;font-size:12px;line-height:18px;color:#6b7280;text-align:center;">',
            '              본 메일은 광고성 정보입니다. 수신을 원하지 않으면 <a href="{{unsubscribe_url}}" style="color:#4b5563;text-decoration:underline;">수신 거부</a>를 선택하세요.',
            "            </td>",
            "          </tr>",
            "        </table>",
            "      </td>",
            "    </tr>",
            "  </table>",
            '  <img src="{{open_pixel_url}}" width="1" height="1" alt="" style="display:block;width:1px;height:1px;border:0;" />',
            "</body>",
            "</html>",
        ]
    )
    validate_rendered_creative(
        channel=ContentChannel.EMAIL,
        html_body=html_body,
        image_url=image_url,
        content_values=content_values,
    )
    return html_body


def render_editorial_email(content_values: Mapping[str, Any]) -> str:
    subject = html.escape(required_value(content_values, "subject"))
    preheader = html.escape(required_value(content_values, "preheader"))
    body = html.escape(required_value(content_values, "body"))
    cta = html.escape(str(content_values.get("cta") or "여름 프로모션 보기"))
    image_url = escaped_absolute_image_url(content_values)
    raw_featured_offers = content_values.get("featured_offers")
    if not isinstance(raw_featured_offers, list) or not raw_featured_offers:
        raise ArtifactRenderError("editorial email requires featured_offers")
    price_display_mode = _price_display_mode(content_values)
    story_sections = [
        _render_editorial_offer_section(
            raw_offer,
            position=index,
            price_display_mode=price_display_mode,
        )
        for index, raw_offer in enumerate(raw_featured_offers, start=1)
    ]
    html_body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  {_creative_source_meta(ContentChannel.EMAIL, content_values)}",
            f"  <title>{subject}</title>",
            "</head>",
            '<body style="margin:0;padding:0;background:#151515;font-family:Arial,Helvetica,sans-serif;color:#f7f7f8;">',
            f'  <div style="display:none;max-height:0;max-width:0;overflow:hidden;opacity:0;color:transparent;">{preheader}</div>',
            '  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:collapse;background:#151515;">',
            "    <tr>",
            '      <td align="center" style="padding:24px 10px;">',
            '        <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="width:100%;max-width:600px;border-collapse:collapse;background:#222222;border-radius:18px;overflow:hidden;">',
            "          <tr>",
            '            <td style="padding:20px 28px;border-bottom:1px solid #363636;font-size:12px;line-height:18px;color:#929292;"><strong style="color:#0F55C8;font-size:17px;">StayLoop</strong><span style="float:right;">[광고] 여름 여행 큐레이션</span></td>',
            "          </tr>",
            "          <tr>",
            '            <td style="padding:0;">',
            f'              <img src="{image_url}" width="600" alt="{subject}" style="display:block;width:100%;max-width:600px;height:auto;border:0;line-height:100%;outline:none;text-decoration:none;" />',
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td align="center" style="padding:30px 34px 28px;">',
            '              <p style="margin:0 0 8px;color:#999999;font-size:12px;line-height:18px;font-weight:700;letter-spacing:1px;">JEJU · OKINAWA SUMMER EDIT</p>',
            f'              <h1 style="margin:0 0 10px;font-size:30px;line-height:39px;color:#ffffff;">{subject}</h1>',
            f'              <p style="margin:0;font-size:15px;line-height:25px;color:#e5e5e5;">{body}</p>',
            "            </td>",
            "          </tr>",
            *story_sections,
            "          <tr>",
            '            <td style="padding:26px 28px 28px;border-top:1px solid #363636;">',
            '              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:separate;border-spacing:14px 0;">',
            "                <tr>",
            '                  <td width="33.33%" valign="top" style="padding:12px 0 0;border-top:2px solid #3b3b3b;text-align:left;"><strong style="display:block;margin:0 0 8px;color:#ffffff;font-size:13px;line-height:19px;">🌊 여행의 리듬</strong><span style="font-size:11px;line-height:17px;color:#9b9b9b;">풍경과 휴식 중 먼저 원하는 여행을 정해보세요.</span></td>',
            '                  <td width="33.33%" valign="top" style="padding:12px 0 0;border-top:2px solid #3b3b3b;text-align:left;"><strong style="display:block;margin:0 0 8px;color:#ffffff;font-size:13px;line-height:19px;">📅 조기 예약</strong><span style="font-size:11px;line-height:17px;color:#9b9b9b;">가격과 할인율을 같은 조건으로 비교하세요.</span></td>',
            '                  <td width="33.33%" valign="top" style="padding:12px 0 0;border-top:2px solid #3b3b3b;text-align:left;"><strong style="display:block;margin:0 0 8px;color:#ffffff;font-size:13px;line-height:19px;">💬 추천 기준</strong><span style="font-size:11px;line-height:17px;color:#9b9b9b;">내 여행 방식에 맞는 숙소를 선택해보세요.</span></td>',
            "                </tr>",
            "              </table>",
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td align="center" style="padding:30px 32px 34px;border-top:1px solid #363636;">',
            '              <h2 style="margin:0 0 8px;font-size:22px;line-height:30px;color:#ffffff;">제주와 오키나와, 어디로 떠날까요?</h2>',
            '              <p style="margin:0 0 18px;font-size:14px;line-height:22px;color:#9b9b9b;">전체 숙소의 가격과 할인 혜택을 한눈에 비교해보세요.</p>',
            f'              <a href="{{{{redirect_url}}}}" style="display:block;padding:14px 24px;border-radius:8px;background:#0F55C8;color:#ffffff;font-size:15px;line-height:20px;font-weight:700;text-align:center;text-decoration:none;">{cta} →</a>',
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:18px 24px;border-top:1px solid #363636;font-size:12px;line-height:18px;color:#858585;text-align:center;">',
            '              표시된 금액은 1객실 1박 기준입니다. 본 메일은 광고성 정보입니다.<br>수신을 원하지 않으면 <a href="{{unsubscribe_url}}" style="color:#9b9b9b;text-decoration:underline;">수신 거부</a>를 선택하세요.',
            "            </td>",
            "          </tr>",
            "        </table>",
            "      </td>",
            "    </tr>",
            "  </table>",
            '  <img src="{{open_pixel_url}}" width="1" height="1" alt="" style="display:block;width:1px;height:1px;border:0;" />',
            "</body>",
            "</html>",
        ]
    )
    validate_rendered_creative(
        channel=ContentChannel.EMAIL,
        html_body=html_body,
        image_url=image_url,
        content_values=content_values,
    )
    return html_body


def _render_editorial_offer_section(
    raw_offer: object,
    *,
    position: int,
    price_display_mode: str | None,
) -> str:
    if not isinstance(raw_offer, Mapping):
        raise ArtifactRenderError("editorial featured offer must be an object")
    hotel_name = html.escape(required_value(raw_offer, "hotel_name"))
    destination_id = str(raw_offer.get("destination_id") or "").strip().casefold()
    destination = html.escape(_destination_label(destination_id))
    image_url = escaped_absolute_url(
        required_value(raw_offer, "image_url"),
        label="editorial offer image_url",
    )
    price_copy = _inline_price_copy(
        _offer_price_tiers(
            raw_offer,
            price_display_mode=price_display_mode,
        )
    )
    discount_rate = _offer_discount_percentage(raw_offer)
    title, description = _destination_editorial_copy(destination_id)
    if discount_rate:
        price_copy = f"{price_copy} · {discount_rate} 할인"
    image_cell = "\n".join(
        [
            '<td width="46%" valign="middle" style="width:46%;padding:0;">',
            f'  <img src="{image_url}" width="276" alt="{hotel_name}" style="display:block;width:100%;height:auto;border:0;" />',
            "</td>",
        ]
    )
    title_html = _editorial_title_html(title)
    copy_cell = "\n".join(
        [
            '<td width="54%" valign="middle" style="width:54%;padding:24px 22px;">',
            f'  <p style="margin:0 0 8px;color:#929292;font-size:11px;line-height:17px;font-weight:700;letter-spacing:1px;">{position:02d} · {destination}</p>',
            f'  <h2 style="margin:0 0 10px;font-size:20px;line-height:28px;color:#ffffff;">{title_html}</h2>',
            f'  <p style="margin:0 0 13px;font-size:13px;line-height:22px;color:#e3e3e3;">{html.escape(description)}</p>',
            f'  <p style="margin:0;font-size:12px;line-height:19px;color:#929292;"><strong style="color:#e5e5e5;">추천:</strong> {hotel_name}<br>{price_copy}</p>',
            "</td>",
        ]
    )
    cells = [image_cell, copy_cell] if position % 2 else [copy_cell, image_cell]
    return "\n".join(
        [
            "          <tr>",
            '            <td style="padding:0 28px;">',
            '              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:collapse;background:#222222;border-top:1px solid #363636;">',
            "                <tr>",
            *[f"                  {line}" for cell in cells for line in cell.splitlines()],
            "                </tr>",
            "              </table>",
            "            </td>",
            "          </tr>",
        ]
    )


def _editorial_title_html(title: str) -> str:
    leading, separator, trailing = title.partition(", ")
    if not separator:
        return html.escape(title)
    return f"{html.escape(leading + ',')}<br>{html.escape(trailing)}"


def _destination_editorial_copy(destination_id: str) -> tuple[str, str]:
    if destination_id == "jeju":
        return (
            "바다와 오름 사이, 천천히 시작하는 하루",
            "제주는 풍경을 따라 움직이는 여행에 잘 어울립니다. "
            "해안의 여유와 숙소에서 보내는 휴식을 함께 계획해보세요.",
        )
    if destination_id == "okinawa":
        return (
            "투명한 바다 곁에서, 오래 머무는 휴식",
            "오키나와에서는 많은 곳을 둘러보기보다 바다 가까운 숙소를 "
            "중심으로 느긋한 여름 일정을 만들어보세요.",
        )
    return (
        "숙소에서 시작하는 여유로운 여행",
        "가격과 할인 혜택을 비교하고 내 여행 방식에 맞는 숙소를 찾아보세요.",
    )


def render_offer_cards_email(content_values: Mapping[str, Any]) -> str:
    subject = html.escape(required_value(content_values, "subject"))
    preheader = html.escape(required_value(content_values, "preheader"))
    body = html.escape(required_value(content_values, "body"))
    cta = html.escape(str(content_values.get("cta") or "프로모션 전체 보기"))
    raw_offers = content_values.get("offers")
    if not isinstance(raw_offers, list) or not raw_offers:
        raise ArtifactRenderError("offer card email requires offers")
    first_offer = raw_offers[0]
    if not isinstance(first_offer, Mapping):
        raise ArtifactRenderError("offer card entry must be an object")
    price_display_mode = _price_display_mode(content_values)
    image_url = escaped_absolute_url(
        required_value(first_offer, "image_url"),
        label="offer image_url",
    )
    grouped_offers: dict[str, list[tuple[int, object]]] = {}
    for original_position, raw_offer in enumerate(raw_offers, start=1):
        if not isinstance(raw_offer, Mapping):
            raise ArtifactRenderError("offer card entry must be an object")
        destination_id = str(raw_offer.get("destination_id") or "").casefold()
        grouped_offers.setdefault(destination_id, []).append(
            (original_position, raw_offer)
        )
    rows: list[str] = []
    for destination_id, destination_offers in grouped_offers.items():
        rows.extend(
            [
                "              <tr>",
                f'                <td colspan="2" style="padding:18px 8px 7px;font-size:20px;line-height:28px;font-weight:700;color:#10233f;">{html.escape(_destination_label(destination_id))} 추천</td>',
                "              </tr>",
            ]
        )
        for offset in range(0, len(destination_offers), 2):
            pair = destination_offers[offset : offset + 2]
            cells = [
                _render_offer_card_cell(
                    raw_offer,
                    position=original_position,
                    cta=cta,
                    price_display_mode=price_display_mode,
                )
                for original_position, raw_offer in pair
            ]
            if len(cells) == 1:
                cells.append(
                    '<td width="50%" valign="top" style="width:50%;padding:7px;"></td>'
                )
            rows.extend(
                [
                    "              <tr>",
                    *cells,
                    "              </tr>",
                ]
            )
    html_body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  {_creative_source_meta(ContentChannel.EMAIL, content_values)}",
            f"  <title>{subject}</title>",
            "</head>",
            '<body style="margin:0;padding:0;background:#eef3fb;font-family:Arial,Helvetica,sans-serif;color:#10233f;">',
            f'  <div style="display:none;max-height:0;max-width:0;overflow:hidden;opacity:0;color:transparent;">{preheader}</div>',
            '  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:collapse;background:#eef3fb;">',
            "    <tr>",
            '      <td align="center" style="padding:24px 10px;">',
            '        <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="width:100%;max-width:600px;border-collapse:collapse;background:#ffffff;border-radius:18px;overflow:hidden;">',
            "          <tr>",
            '            <td style="padding:15px 24px;font-size:12px;line-height:18px;color:#66758a;">STAYLOOP · 광고</td>',
            "          </tr>",
            "          <tr>",
            '            <td style="padding:24px 28px 12px;text-align:center;">',
            '              <p style="margin:0 0 8px;color:#0F55C8;font-size:12px;line-height:18px;font-weight:700;letter-spacing:1px;">SUMMER STAY COLLECTION</p>',
            f'              <h1 style="margin:0 0 10px;font-size:28px;line-height:36px;color:#10233f;">{subject}</h1>',
            f'              <p style="margin:0;font-size:15px;line-height:24px;color:#53657d;">{body}</p>',
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:5px 13px 18px;">',
            '              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:collapse;">',
            *rows,
            "              </table>",
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td align="center" style="padding:0 28px 30px;">',
            f'              <a href="{{{{redirect_url}}}}" style="display:block;padding:14px 26px;border-radius:8px;background:#0F55C8;color:#ffffff;font-size:15px;line-height:20px;font-weight:700;text-align:center;text-decoration:none;">{cta} →</a>',
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:18px 24px;border-top:1px solid #e5eaf1;font-size:12px;line-height:18px;color:#718096;text-align:center;">',
            '              표시된 금액은 1객실 1박 기준입니다. 본 메일은 광고성 정보입니다.<br>수신을 원하지 않으면 <a href="{{unsubscribe_url}}" style="color:#53657d;text-decoration:underline;">수신 거부</a>를 선택하세요.',
            "            </td>",
            "          </tr>",
            "        </table>",
            "      </td>",
            "    </tr>",
            "  </table>",
            '  <img src="{{open_pixel_url}}" width="1" height="1" alt="" style="display:block;width:1px;height:1px;border:0;" />',
            "</body>",
            "</html>",
        ]
    )
    validate_rendered_creative(
        channel=ContentChannel.EMAIL,
        html_body=html_body,
        image_url=image_url,
        content_values=content_values,
    )
    return html_body


def _render_offer_card_cell(
    raw_offer: object,
    *,
    position: int,
    cta: str,
    price_display_mode: str | None,
) -> str:
    if not isinstance(raw_offer, Mapping):
        raise ArtifactRenderError("offer card entry must be an object")
    hotel_name = html.escape(required_value(raw_offer, "hotel_name"))
    destination = html.escape(_destination_label(raw_offer.get("destination_id")))
    image_url = escaped_absolute_url(
        required_value(raw_offer, "image_url"),
        label="offer image_url",
    )
    placeholder = required_value(raw_offer, "redirect_placeholder")
    expected_placeholder = f"{{{{offer_redirect_url_{position}}}}}"
    if placeholder != expected_placeholder:
        raise ArtifactRenderError("offer redirect placeholder order is invalid")
    price_tiers = _offer_price_tiers(
        raw_offer,
        price_display_mode=price_display_mode,
    )
    discount_rate = _offer_discount_percentage(raw_offer)
    has_additional_discount = (
        raw_offer.get("additional_discount_rate_percent") is not None
    )
    badge = (
        f'<span style="display:inline-block;padding:3px 7px;border-radius:999px;background:#e8f1ff;color:#0f55c8;font-size:11px;font-weight:700;">{"추가 " if has_additional_discount else ""}{discount_rate} 할인</span>'
        if discount_rate
        else '<span style="display:inline-block;padding:3px 7px;border-radius:999px;background:#edf2f7;color:#53657d;font-size:11px;font-weight:700;">특별가</span>'
    )
    price_html = _offer_card_price_html(price_tiers)
    return "\n".join(
        [
            '<td width="50%" valign="top" style="width:50%;padding:7px;">',
            '  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:separate;border-spacing:0;border:1px solid #dfe6ef;border-radius:12px;overflow:hidden;">',
            "    <tr>",
            f'      <td><img src="{image_url}" width="260" alt="{hotel_name}" style="display:block;width:100%;height:150px;object-fit:cover;border:0;" /></td>',
            "    </tr>",
            "    <tr>",
            '      <td style="padding:14px 14px 16px;">',
            f'        <div style="margin-bottom:7px;">{badge}</div>',
            f'        <strong style="display:block;min-height:42px;font-size:15px;line-height:21px;color:#10233f;">{hotel_name}</strong>',
            f'        <span style="display:block;margin:4px 0 12px;font-size:12px;line-height:18px;color:#718096;">{destination}</span>',
            f'        <div style="margin-bottom:12px;">{price_html}</div>',
            f'        <a href="{placeholder}" style="display:block;padding:10px 8px;border-radius:7px;background:#0F55C8;color:#ffffff;font-size:13px;line-height:18px;font-weight:700;text-align:center;text-decoration:none;">{cta}</a>',
            "      </td>",
            "    </tr>",
            "  </table>",
            "</td>",
        ]
    )


def render_comparison_email(content_values: Mapping[str, Any]) -> str:
    subject = html.escape(required_value(content_values, "subject"))
    preheader = html.escape(required_value(content_values, "preheader"))
    body = html.escape(required_value(content_values, "body"))
    cta = html.escape(str(content_values.get("cta") or "여름 숙소 비교하기"))
    raw_comparison_offers = content_values.get("comparison_offers")
    if not isinstance(raw_comparison_offers, list) or not raw_comparison_offers:
        raise ArtifactRenderError("comparison email requires comparison_offers")
    first_offer = raw_comparison_offers[0]
    if not isinstance(first_offer, Mapping):
        raise ArtifactRenderError("comparison offer must be an object")
    price_display_mode = _price_display_mode(content_values)
    image_url = escaped_absolute_url(
        required_value(first_offer, "image_url"),
        label="comparison image_url",
    )
    destination_positions: dict[str, int] = {}
    comparison_rows: list[str] = []
    discount_rates: list[int] = []
    for raw_offer in raw_comparison_offers:
        if not isinstance(raw_offer, Mapping):
            raise ArtifactRenderError("comparison offer must be an object")
        destination_id = str(raw_offer.get("destination_id") or "").casefold()
        destination_positions[destination_id] = (
            destination_positions.get(destination_id, 0) + 1
        )
        comparison_rows.append(
            _render_comparison_offer_row(
                raw_offer,
                destination_position=destination_positions[destination_id],
                price_display_mode=price_display_mode,
            )
        )
        raw_discount = raw_offer.get("additional_discount_rate_percent")
        if raw_discount is None:
            raw_discount = raw_offer.get("discount_rate_percent")
        if raw_discount is not None:
            try:
                discount_rates.append(int(raw_discount))
            except (TypeError, ValueError) as exc:
                raise ArtifactRenderError(
                    "comparison discount rate must be an integer"
                ) from exc
    discount_summary = (
        f"최대 {max(discount_rates)}% 할인"
        if discount_rates
        else "숙소별 특별가"
    )
    html_body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  {_creative_source_meta(ContentChannel.EMAIL, content_values)}",
            f"  <title>{subject}</title>",
            "</head>",
            '<body style="margin:0;padding:0;background:#edf2f8;font-family:Arial,Helvetica,sans-serif;color:#ffffff;">',
            f'  <div style="display:none;max-height:0;max-width:0;overflow:hidden;opacity:0;color:transparent;">{preheader}</div>',
            '  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:collapse;background:#edf2f8;">',
            "    <tr>",
            '      <td align="center" style="padding:24px 10px;">',
            '        <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="width:100%;max-width:600px;border-collapse:collapse;background:#0d1828;border-radius:18px;overflow:hidden;">',
            "          <tr>",
            '            <td style="padding:18px 28px;font-size:12px;line-height:18px;color:#9fb0c7;"><strong style="color:#ffffff;font-size:16px;">StayLoop</strong><span style="float:right;">[광고] SUMMER PICKS</span></td>',
            "          </tr>",
            "          <tr>",
            '            <td style="padding:34px 34px 28px;text-align:center;background:#111f33;">',
            '              <p style="margin:0 0 10px;color:#79a9ff;font-size:12px;line-height:18px;font-weight:700;letter-spacing:1px;">JEJU · OKINAWA</p>',
            f'              <h1 style="margin:0 0 14px;font-size:31px;line-height:40px;color:#ffffff;">{subject}</h1>',
            f'              <p style="margin:0 0 12px;font-size:15px;line-height:24px;color:#c7d2e2;">{body}</p>',
            f'              <p style="margin:0;font-size:14px;line-height:22px;color:#ffffff;">✈️ 인기 여행지&nbsp;&nbsp;·&nbsp;&nbsp;📅 {html.escape(discount_summary)}&nbsp;&nbsp;·&nbsp;&nbsp;🔎 간결한 비교</p>',
            "            </td>",
            "          </tr>",
            *comparison_rows,
            "          <tr>",
            '            <td align="center" style="padding:28px 32px 34px;">',
            '              <h2 style="margin:0 0 8px;font-size:22px;line-height:30px;color:#ffffff;">가격과 위치를 한눈에 비교해보세요</h2>',
            '              <p style="margin:0 0 18px;font-size:13px;line-height:21px;color:#9fb0c7;">프로모션 페이지에서 전체 숙소와 예약 조건을 확인할 수 있습니다.</p>',
            f'              <a href="{{{{redirect_url}}}}" style="display:block;padding:14px 24px;border-radius:8px;background:#0F55C8;color:#ffffff;font-size:15px;line-height:20px;font-weight:700;text-align:center;text-decoration:none;">{cta} →</a>',
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:18px 24px;border-top:1px solid #26364d;font-size:12px;line-height:18px;color:#8293aa;text-align:center;">',
            '              표시된 금액은 1객실 1박 기준입니다. 본 메일은 광고성 정보입니다.<br>수신을 원하지 않으면 <a href="{{unsubscribe_url}}" style="color:#b7c5d8;text-decoration:underline;">수신 거부</a>를 선택하세요.',
            "            </td>",
            "          </tr>",
            "        </table>",
            "      </td>",
            "    </tr>",
            "  </table>",
            '  <img src="{{open_pixel_url}}" width="1" height="1" alt="" style="display:block;width:1px;height:1px;border:0;" />',
            "</body>",
            "</html>",
        ]
    )
    validate_rendered_creative(
        channel=ContentChannel.EMAIL,
        html_body=html_body,
        image_url=image_url,
        content_values=content_values,
    )
    return html_body


def _render_comparison_offer_row(
    raw_offer: Mapping[str, Any],
    *,
    destination_position: int,
    price_display_mode: str | None,
) -> str:
    hotel_name = html.escape(required_value(raw_offer, "hotel_name"))
    destination = html.escape(
        _destination_label(raw_offer.get("destination_id"))
    )
    image_url = escaped_absolute_url(
        required_value(raw_offer, "image_url"),
        label="comparison image_url",
    )
    price_copy = _inline_price_copy(
        _offer_price_tiers(
            raw_offer,
            price_display_mode=price_display_mode,
        )
    )
    discount_rate = _offer_discount_percentage(raw_offer)
    if discount_rate:
        price_copy = f"{price_copy} · {discount_rate} 할인"
    return "\n".join(
        [
            "          <tr>",
            '            <td style="padding:10px 28px;">',
            '              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:separate;border-spacing:0;background:#17263a;border:1px solid #293b53;border-radius:12px;overflow:hidden;">',
            "                <tr>",
            '                  <td width="46%" valign="middle" style="width:46%;padding:0;">',
            f'                    <img src="{image_url}" width="250" alt="{hotel_name}" style="display:block;width:100%;height:145px;object-fit:cover;border:0;" />',
            "                  </td>",
            '                  <td width="54%" valign="middle" style="width:54%;padding:20px 22px;">',
            f'                    <p style="margin:0 0 7px;color:#79a9ff;font-size:11px;line-height:17px;font-weight:700;letter-spacing:1px;">{destination.upper()} PICK {destination_position:02d}</p>',
            f'                    <h3 style="margin:0 0 9px;font-size:18px;line-height:25px;color:#ffffff;">{hotel_name}</h3>',
            f'                    <p style="margin:0;color:#ffffff;font-size:15px;line-height:23px;font-weight:700;">{price_copy}</p>',
            "                  </td>",
            "                </tr>",
            "              </table>",
            "            </td>",
            "          </tr>",
        ]
    )


def render_banner_html(content_values: Mapping[str, str | None]) -> str:
    title = html.escape(required_value(content_values, "title"))
    body = html.escape(required_value(content_values, "body"))
    cta = html.escape(required_value(content_values, "cta"))
    image_url = escaped_absolute_image_url(content_values)
    html_body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  {_creative_source_meta(ContentChannel.ONSITE_BANNER, content_values)}",
            "  <style>",
            "    html,body{width:100%;height:100%;margin:0;overflow:hidden;}",
            "    body{font-family:Arial,Helvetica,sans-serif;background:#072b63;color:#fff;}",
            "    button{position:relative;width:100%;height:100%;min-height:100px;border:0;padding:0;overflow:hidden;background:#072b63;color:#fff;text-align:left;cursor:pointer;}",
            "    .loopad-image{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}",
            "    .loopad-scrim{position:absolute;inset:0;background:linear-gradient(90deg,rgba(7,43,99,.94) 0%,rgba(7,43,99,.78) 52%,rgba(7,43,99,.10) 100%);}",
            "    .loopad-copy{position:relative;z-index:1;display:block;box-sizing:border-box;width:72%;height:100%;padding:10px 14px;}",
            "    .loopad-label{display:inline-block;margin-bottom:4px;padding:2px 5px;border:1px solid rgba(255,255,255,.72);border-radius:3px;font-size:9px;line-height:11px;}",
            "    .loopad-title{display:block;overflow:hidden;margin-bottom:3px;font-size:16px;line-height:19px;font-weight:700;white-space:nowrap;text-overflow:ellipsis;}",
            "    .loopad-body{display:block;overflow:hidden;margin-bottom:6px;font-size:11px;line-height:14px;white-space:nowrap;text-overflow:ellipsis;}",
            "    .loopad-cta{display:inline-block;padding:4px 8px;border-radius:4px;background:#fff;color:#0f55c8;font-size:10px;line-height:12px;font-style:normal;font-weight:700;}",
            "  </style>",
            "</head>",
            "<body>",
            f'  <button type="button" id="loopad-click" aria-label="{title}">',
            f'    <img class="loopad-image" src="{image_url}" alt="" aria-hidden="true" />',
            '    <span class="loopad-scrim" aria-hidden="true"></span>',
            '    <span class="loopad-copy">',
            '      <span class="loopad-label">광고</span>',
            f'      <strong class="loopad-title">{title}</strong>',
            f'      <span class="loopad-body">{body}</span>',
            f'      <em class="loopad-cta">{cta}</em>',
            "    </span>",
            "  </button>",
            "  <script>",
            "    document.getElementById('loopad-click').addEventListener('click', function () {",
            "      window.parent.postMessage({ type: 'loopad:click' }, '*');",
            "    });",
            "  </script>",
            "</body>",
            "</html>",
        ]
    )
    validate_rendered_creative(
        channel=ContentChannel.ONSITE_BANNER,
        html_body=html_body,
        image_url=image_url,
        content_values=content_values,
    )
    return html_body


def _creative_source_meta(
    channel: ContentChannel,
    content_values: Mapping[str, Any],
) -> str:
    values: dict[str, str | None] = {}
    for field_name in CREATIVE_SOURCE_FIELDS:
        value = content_values.get(field_name)
        values[field_name] = str(value).strip() if value is not None else None
    payload = {
        "schema_version": CREATIVE_SOURCE_SCHEMA_VERSION,
        "channel": channel.value,
        "values": values,
        "image_prompt_sha256": (
            image_prompt_sha256(content_values.get("image_prompt"))
            if content_values.get("image_prompt")
            else None
        ),
        "landing_url_sha256": (
            hashlib.sha256(
                str(content_values.get("landing_url")).strip().encode("utf-8")
            ).hexdigest()
            if content_values.get("landing_url")
            else None
        ),
        "creative_contract_sha256": creative_contract_sha256(content_values),
        "renderer_version": _snapshot_version(
            content_values.get("renderer_version"),
            default=RENDERER_VERSION,
        ),
        "template_version": _snapshot_version(
            content_values.get("template_version"),
            default=template_version_for_channel(channel),
        ),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    return (
        f'<meta name="{CREATIVE_SOURCE_META_NAME}" '
        f'content="{encoded.rstrip("=")}">'
    )


def creative_contract_sha256(content_values: Mapping[str, Any]) -> str:
    """Fingerprint variant data without exposing offer destinations in HTML."""

    contract = {
        field_name: content_values.get(field_name)
        for field_name in CREATIVE_CONTRACT_FIELDS
    }
    try:
        encoded = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactRenderError(
            "creative renderer contract is not JSON-safe"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def content_values_from_rendered_html(
    *,
    channel: ContentChannel,
    html_body: str,
) -> dict[str, str | None]:
    """Recover the public source snapshot embedded by the renderer."""

    if channel == ContentChannel.SMS:
        raise ArtifactRenderError("SMS does not have rendered HTML source")
    matches = _CREATIVE_SOURCE_META_PATTERN.findall(html_body)
    if len(matches) != 1:
        raise ArtifactRenderError(
            "rendered HTML must contain exactly one creative source snapshot"
        )
    try:
        encoded = matches[0]
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(f"{encoded}{padding}").decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactRenderError("rendered HTML creative source is invalid") from exc
    if not isinstance(payload, dict):
        raise ArtifactRenderError("rendered HTML creative source shape is invalid")
    schema_version = payload.get("schema_version")
    legacy_fields = {
        "schema_version",
        "channel",
        "values",
        "image_prompt_sha256",
        "landing_url_sha256",
    }
    previous_fields = legacy_fields | {
        "renderer_version",
        "template_version",
    }
    current_fields = previous_fields | {"creative_contract_sha256"}
    if schema_version == LEGACY_CREATIVE_SOURCE_SCHEMA_VERSION:
        expected_fields = legacy_fields
    elif schema_version == PREVIOUS_CREATIVE_SOURCE_SCHEMA_VERSION:
        expected_fields = previous_fields
    else:
        expected_fields = current_fields
    if set(payload) != expected_fields:
        raise ArtifactRenderError("rendered HTML creative source shape is invalid")
    if schema_version not in {
        LEGACY_CREATIVE_SOURCE_SCHEMA_VERSION,
        PREVIOUS_CREATIVE_SOURCE_SCHEMA_VERSION,
        CREATIVE_SOURCE_SCHEMA_VERSION,
    }:
        raise ArtifactRenderError("rendered HTML creative source version is invalid")
    if payload.get("channel") != channel.value:
        raise ArtifactRenderError("rendered HTML creative source channel is invalid")
    raw_values = payload.get("values")
    if not isinstance(raw_values, dict) or set(raw_values) != set(
        CREATIVE_SOURCE_FIELDS
    ):
        raise ArtifactRenderError("rendered HTML creative source fields are invalid")

    values: dict[str, str | None] = {}
    for field_name in CREATIVE_SOURCE_FIELDS:
        value = raw_values.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ArtifactRenderError(
                "rendered HTML creative source values are invalid"
            )
        normalized = value.strip() if isinstance(value, str) else None
        if value != normalized:
            raise ArtifactRenderError(
                "rendered HTML creative source values are not canonical"
            )
        values[field_name] = normalized

    prompt_sha256 = payload.get("image_prompt_sha256")
    if prompt_sha256 is not None:
        try:
            prompt_sha256 = _validated_sha256(str(prompt_sha256))
        except ValueError as exc:
            raise ArtifactRenderError(
                "rendered HTML image prompt fingerprint is invalid"
            ) from exc
    values["image_prompt_sha256"] = prompt_sha256

    landing_url_sha256 = payload.get("landing_url_sha256")
    if landing_url_sha256 is not None:
        try:
            landing_url_sha256 = _validated_sha256(str(landing_url_sha256))
        except ValueError as exc:
            raise ArtifactRenderError(
                "rendered HTML landing URL fingerprint is invalid"
            ) from exc
    values["landing_url_sha256"] = landing_url_sha256

    creative_contract_sha256_value = payload.get("creative_contract_sha256")
    if schema_version == CREATIVE_SOURCE_SCHEMA_VERSION:
        try:
            creative_contract_sha256_value = _validated_sha256(
                str(creative_contract_sha256_value)
            )
        except ValueError as exc:
            raise ArtifactRenderError(
                "rendered HTML creative contract fingerprint is invalid"
            ) from exc
    else:
        creative_contract_sha256_value = None
    values["creative_contract_sha256"] = creative_contract_sha256_value

    if schema_version == LEGACY_CREATIVE_SOURCE_SCHEMA_VERSION:
        renderer_version = LEGACY_RENDERER_VERSION
        template_version = legacy_template_version_for_channel(channel)
    else:
        try:
            renderer_version = _snapshot_version(
                payload.get("renderer_version"),
            )
            template_version = _snapshot_version(
                payload.get("template_version"),
            )
        except ValueError as exc:
            raise ArtifactRenderError(
                "rendered HTML renderer provenance is invalid"
            ) from exc
    values["renderer_version"] = renderer_version
    values["template_version"] = template_version

    return values


def _snapshot_version(value: object, *, default: str | None = None) -> str:
    normalized = str(value).strip() if value is not None else ""
    if not normalized and default is not None:
        normalized = default.strip()
    if not normalized:
        raise ValueError("creative source renderer version must not be empty")
    return normalized


def html_artifact_key(
    *,
    base_prefix: str,
    identity: ArtifactIdentity,
    creative_format: CreativeFormat,
) -> str:
    if creative_format == CreativeFormat.EMAIL_HTML:
        filename = "creative.email.html"
    elif creative_format == CreativeFormat.BANNER_HTML:
        filename = "creative.banner.html"
    else:
        raise ValueError("SMS does not have an HTML artifact key")
    return f"{artifact_directory(base_prefix=base_prefix, identity=identity)}/{filename}"


def image_artifact_key(
    *,
    base_prefix: str,
    identity: ArtifactIdentity,
    content_type: str,
    image_prompt_sha256: str,
) -> str:
    extension = image_extension(content_type)
    digest = _validated_sha256(image_prompt_sha256)
    return (
        f"{artifact_directory(base_prefix=base_prefix, identity=identity)}"
        f"/image.{digest}.{extension}"
    )


def image_prompt_sha256(image_prompt: str | None) -> str:
    prompt = str(image_prompt or "").strip()
    if not prompt:
        raise ValueError("image_prompt is required")
    if prompt.startswith(RECOVERED_IMAGE_PROMPT_PREFIX):
        return _validated_sha256(prompt.removeprefix(RECOVERED_IMAGE_PROMPT_PREFIX))
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def recovered_image_prompt(prompt_sha256: str) -> str:
    return f"{RECOVERED_IMAGE_PROMPT_PREFIX}{_validated_sha256(prompt_sha256)}"


def artifact_directory(
    *,
    base_prefix: str,
    identity: ArtifactIdentity,
) -> str:
    prefix = base_prefix.strip("/")
    parts = [
        safe_asset_name(identity.project_id),
        safe_asset_name(identity.promotion_id),
        safe_asset_name(identity.generation_id),
        safe_asset_name(identity.content_id),
    ]
    path = "/".join(parts)
    return f"{prefix}/{path}" if prefix else path


def _validated_sha256(value: str) -> str:
    digest = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("content_sha256 must be 64 lowercase hexadecimal characters")
    return digest


def html_artifact_metadata(
    *,
    creative_format: CreativeFormat,
    storage_key: str,
    public_url: str,
    html_body: str,
) -> dict[str, Any]:
    encoded = html_body.encode("utf-8")
    metadata: dict[str, Any] = {
        "creative_format": creative_format.value,
        "artifact_status": "published",
        "storage_key": storage_key,
        "public_url": public_url,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "content_type": HTML_CONTENT_TYPE,
    }
    metadata.update(artifact_dimensions(creative_format))
    return metadata


def artifact_dimensions(creative_format: CreativeFormat) -> dict[str, int]:
    if creative_format == CreativeFormat.BANNER_HTML:
        return {
            "width": DEFAULT_BANNER_WIDTH,
            "height": DEFAULT_BANNER_HEIGHT,
        }
    return {}


def public_asset_url(
    *,
    public_base_url: str,
    base_prefix: str,
    key: str,
) -> str:
    prefix = base_prefix.strip("/")
    public_path = key
    if prefix and key.startswith(f"{prefix}/"):
        public_path = key[len(prefix) + 1 :]
    return f"{public_base_url.rstrip('/')}/{public_path}"


def required_value(values: Mapping[str, str | None], key: str) -> str:
    value = values.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"creative source requires {key}")
    return str(value).strip()


def escaped_absolute_image_url(
    content_values: Mapping[str, str | None],
) -> str:
    try:
        image_url = required_value(content_values, "image_url")
    except ValueError as exc:
        raise ArtifactRenderError(
            "creative image_url is required before rendering"
        ) from exc
    parsed = urlsplit(image_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ArtifactRenderError(
            "creative image_url must be an absolute HTTPS URL"
        )
    if parsed.username or parsed.password:
        raise ArtifactRenderError("creative image_url must not contain credentials")
    return html.escape(image_url, quote=True)


def escaped_absolute_url(value: str, *, label: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ArtifactRenderError(f"{label} must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        raise ArtifactRenderError(f"{label} must not contain credentials")
    return html.escape(value, quote=True)


def _destination_label(value: object) -> str:
    labels = {"jeju": "제주", "okinawa": "오키나와"}
    destination_id = str(value or "").strip().casefold()
    return labels.get(destination_id, destination_id or "추천 여행지")


@dataclass(frozen=True)
class _OfferPriceTier:
    label: str
    formatted: str
    emphasized: bool
    struck: bool = False


def _price_display_mode(content_values: Mapping[str, Any]) -> str | None:
    value = str(content_values.get("price_display_mode") or "").strip()
    if not value:
        return None
    if value not in {
        PRICE_DISPLAY_PROMOTION_AND_FINAL,
        PRICE_DISPLAY_ALL_TIERS,
    }:
        raise ArtifactRenderError("email price display mode is invalid")
    return value


def _offer_price_tiers(
    raw_offer: Mapping[str, Any],
    *,
    price_display_mode: str | None,
) -> tuple[_OfferPriceTier, ...]:
    final_price = _won_price(raw_offer.get("sale_price_per_night"))
    regular_price = _optional_won_price(
        raw_offer.get("original_price_per_night")
    )
    promotion_price = _optional_won_price(
        raw_offer.get("promotion_price_per_night")
    )

    if promotion_price is not None:
        tiers: list[_OfferPriceTier] = []
        if price_display_mode == PRICE_DISPLAY_ALL_TIERS and regular_price:
            tiers.append(
                _OfferPriceTier(
                    label="정상가",
                    formatted=regular_price,
                    emphasized=False,
                    struck=True,
                )
            )
        tiers.extend(
            (
                _OfferPriceTier(
                    label="프로모션가",
                    formatted=promotion_price,
                    emphasized=False,
                    struck=True,
                ),
                _OfferPriceTier(
                    label="추가 할인가",
                    formatted=final_price,
                    emphasized=True,
                ),
            )
        )
        return tuple(tiers)

    if regular_price is not None:
        return (
            _OfferPriceTier(
                label="정상가",
                formatted=regular_price,
                emphasized=False,
                struck=True,
            ),
            _OfferPriceTier(
                label="프로모션가",
                formatted=final_price,
                emphasized=True,
            ),
        )

    return (
        _OfferPriceTier(
            label="1박",
            formatted=final_price,
            emphasized=True,
        ),
    )


def _inline_price_copy(price_tiers: tuple[_OfferPriceTier, ...]) -> str:
    return " · ".join(
        f"{price_tier.label} {price_tier.formatted}"
        for price_tier in price_tiers
    )


def _offer_discount_percentage(raw_offer: Mapping[str, Any]) -> str | None:
    value = raw_offer.get("additional_discount_rate_percent")
    if value is None:
        value = raw_offer.get("discount_rate_percent")
    return _optional_percentage(value)


def _offer_card_price_html(
    price_tiers: tuple[_OfferPriceTier, ...],
) -> str:
    rows: list[str] = []
    for price_tier in price_tiers:
        if price_tier.emphasized:
            rows.append(
                '<span style="display:block;color:#ef476f;font-size:19px;'
                'line-height:25px;font-weight:800;">'
                f"{html.escape(price_tier.label)} "
                f"{html.escape(price_tier.formatted)} / 박"
                "</span>"
            )
            continue
        text_decoration = "text-decoration:line-through;" if price_tier.struck else ""
        rows.append(
            '<span style="display:block;color:#718096;font-size:12px;'
            f'line-height:18px;{text_decoration}">'
            f"{html.escape(price_tier.label)} "
            f"{html.escape(price_tier.formatted)}"
            "</span>"
        )
    return "".join(rows)


def _won_price(value: object) -> str:
    try:
        amount = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactRenderError("offer sale price must be an integer") from exc
    if amount < 0:
        raise ArtifactRenderError("offer sale price must not be negative")
    return f"{amount:,}원"


def _optional_won_price(value: object) -> str | None:
    if value is None:
        return None
    return _won_price(value)


def _optional_percentage(value: object) -> str | None:
    if value is None:
        return None
    try:
        percentage = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactRenderError("offer discount rate must be an integer") from exc
    if percentage < 0 or percentage > 100:
        raise ArtifactRenderError("offer discount rate must be between 0 and 100")
    return f"{percentage}%"


def validate_rendered_creative(
    *,
    channel: ContentChannel,
    html_body: str,
    image_url: str,
    content_values: Mapping[str, Any],
) -> None:
    if image_url not in html_body:
        raise ArtifactRenderError("rendered HTML does not reference image_url")
    if channel == ContentChannel.EMAIL:
        required_tokens = (
            'role="presentation"',
            'width="600"',
            *email_required_placeholders(content_values),
            "광고",
            "수신 거부",
        )
    elif channel == ContentChannel.ONSITE_BANNER:
        required_tokens = (
            "loopad:click",
            "window.parent.postMessage",
            "loopad-image",
            "loopad-cta",
            "광고",
        )
    else:
        raise ArtifactRenderError("SMS does not have rendered HTML")
    missing = [token for token in required_tokens if token not in html_body]
    if missing:
        raise ArtifactRenderError(
            "rendered HTML is missing required contract tokens"
        )


def image_extension(content_type: str) -> str:
    normalized = str(content_type).split(";", 1)[0].strip().lower()
    if normalized == "image/jpeg":
        return "jpg"
    if normalized == "image/webp":
        return "webp"
    if normalized == "image/png":
        return "png"
    raise ValueError(f"unsupported image content type: {content_type}")


def provisional_promotion_run_id(promotion_id: str) -> str:
    return build_bounded_contract_id("prun", promotion_id, "loop_1")


def provisional_ad_experiment_id(
    *,
    promotion_run_id: str,
    segment_id: str,
) -> str:
    return build_bounded_contract_id("adexp", promotion_run_id, segment_id)


def build_bounded_contract_id(prefix: str, *parts: str) -> str:
    seed = "::".join(parts)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]  # noqa: S324
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", "_".join(parts)).strip("_").lower()
    if not slug:
        slug = "id"

    max_slug_length = MAX_CONTRACT_ID_LENGTH - len(prefix) - len(digest) - 2
    slug = slug[:max_slug_length].rstrip("_") or "id"
    return f"{prefix}_{slug}_{digest}"


def safe_asset_name(value: str) -> str:
    raw_value = str(value)
    if re.fullmatch(
        r"[a-zA-Z0-9](?:[a-zA-Z0-9_.-]*[a-zA-Z0-9])?",
        raw_value,
    ):
        return raw_value
    encoded = base64.urlsafe_b64encode(raw_value.encode("utf-8")).decode("ascii")
    return f"~{encoded.rstrip('=')}"


def safe_error_code(exc: Exception) -> str:
    if isinstance(exc, ArtifactRenderError):
        return "artifact_render_failed"
    error_code = getattr(exc, "code", None)
    if isinstance(error_code, str) and re.fullmatch(r"[a-z0-9_]{1,100}", error_code):
        return error_code
    return "artifact_publish_failed"
