from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from app.generation.schemas import ContentChannel, CreativeFormat


HTML_CONTENT_TYPE = "text/html; charset=utf-8"
DEFAULT_BANNER_WIDTH = 320
DEFAULT_BANNER_HEIGHT = 100
DEFAULT_GENAI_PUBLIC_BASE_URL = "https://gen-ai.asset.dev.loop-ad.org"
MAX_CONTRACT_ID_LENGTH = 100


class HtmlArtifactStorage(Protocol):
    def store_html(
        self,
        *,
        content_id: str,
        creative_format: CreativeFormat,
        html_body: str,
    ) -> Mapping[str, Any]:
        ...


class CreativeArtifactPublisher(Protocol):
    def publish(
        self,
        *,
        content_id: str,
        channel: ContentChannel,
        content_values: Mapping[str, str | None],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class StaticCreativeArtifactPublisher:
    public_base_url: str = DEFAULT_GENAI_PUBLIC_BASE_URL
    base_prefix: str = "genai/"

    def publish(
        self,
        *,
        content_id: str,
        channel: ContentChannel,
        content_values: Mapping[str, str | None],
    ) -> dict[str, Any]:
        creative_format = creative_format_for_channel(channel)
        if creative_format == CreativeFormat.SMS_TEXT:
            return not_required_artifact(creative_format)

        html_body = render_creative_html(
            channel=channel,
            content_values=content_values,
        )
        content_sha256 = hashlib.sha256(html_body.encode("utf-8")).hexdigest()
        storage_key = html_artifact_key(
            base_prefix=self.base_prefix,
            content_id=content_id,
            creative_format=creative_format,
            content_sha256=content_sha256,
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
        return artifact


@dataclass(frozen=True)
class S3CreativeArtifactPublisher:
    storage: HtmlArtifactStorage

    def publish(
        self,
        *,
        content_id: str,
        channel: ContentChannel,
        content_values: Mapping[str, str | None],
    ) -> dict[str, Any]:
        creative_format = creative_format_for_channel(channel)
        if creative_format == CreativeFormat.SMS_TEXT:
            return not_required_artifact(creative_format)

        stored = self.storage.store_html(
            content_id=content_id,
            creative_format=creative_format,
            html_body=render_creative_html(
                channel=channel,
                content_values=content_values,
            ),
        )

        return {
            "creative_format": creative_format.value,
            "artifact_status": "published",
            **dict(stored),
        }


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
            "required_placeholders": ["{{redirect_url}}", "{{open_pixel_url}}"],
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
    content_id: str,
    content_values: Mapping[str, str | None],
    artifact_publisher: CreativeArtifactPublisher,
) -> dict[str, Any]:
    return {
        "creative_format": creative_format_for_channel(channel).value,
        "source": source_for_channel(channel=channel, content_values=content_values),
        "artifact": artifact_publisher.publish(
            content_id=content_id,
            channel=channel,
            content_values=content_values,
        ),
    }


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
    subject = html.escape(required_value(content_values, "subject"))
    preheader = html.escape(required_value(content_values, "preheader"))
    body = html.escape(required_value(content_values, "body"))
    cta = html.escape(str(content_values.get("cta") or "Open"))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            f"  <title>{subject}</title>",
            "</head>",
            '<body style="margin:0;padding:24px;font-family:Arial,sans-serif;">',
            f'  <div style="display:none;max-height:0;overflow:hidden;">{preheader}</div>',
            f"  <p>{body}</p>",
            '  <p><a href="{{redirect_url}}">',
            f"    {cta}",
            "  </a></p>",
            '  <img src="{{open_pixel_url}}" width="1" height="1" alt="" />',
            "</body>",
            "</html>",
        ]
    )


def render_banner_html(content_values: Mapping[str, str | None]) -> str:
    title = html.escape(required_value(content_values, "title"))
    body = html.escape(required_value(content_values, "body"))
    cta = html.escape(required_value(content_values, "cta"))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            "  <style>",
            "    body{margin:0;font-family:Arial,sans-serif;background:#fff;color:#111;}",
            "    button{width:100%;height:100%;border:0;background:#fff;text-align:left;padding:14px;cursor:pointer;}",
            "    strong{display:block;font-size:16px;margin-bottom:6px;}",
            "    span{display:block;font-size:13px;margin-bottom:8px;}",
            "    em{font-style:normal;font-weight:700;}",
            "  </style>",
            "</head>",
            "<body>",
            '  <button type="button" id="loopad-click">',
            f"    <strong>{title}</strong>",
            f"    <span>{body}</span>",
            f"    <em>{cta}</em>",
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


def html_artifact_key(
    *,
    base_prefix: str,
    content_id: str,
    creative_format: CreativeFormat,
    content_sha256: str,
) -> str:
    prefix = base_prefix.strip("/")
    digest = _validated_sha256(content_sha256)
    filename = f"{digest}.{creative_format.value}.html"
    path = f"generated/{safe_asset_name(content_id)}/{filename}"
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
    if creative_format == CreativeFormat.BANNER_HTML:
        metadata["width"] = DEFAULT_BANNER_WIDTH
        metadata["height"] = DEFAULT_BANNER_HEIGHT
    return metadata


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
    safe_value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._-")
    return safe_value or "content"


def safe_error_code(exc: Exception) -> str:
    del exc
    return "artifact_publish_failed"
