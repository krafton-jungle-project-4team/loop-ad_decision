from __future__ import annotations

import base64
import hashlib
import json
import re

import pytest

from app.generation.artifacts import (
    ArtifactIdentity,
    ArtifactRenderError,
    CREATIVE_SOURCE_META_NAME,
    StaticCreativeArtifactPublisher,
    content_values_from_rendered_html,
    creative_contract_sha256,
    html_artifact_key,
    image_artifact_key,
    image_prompt_sha256,
    public_asset_url,
    renderer_metadata,
    render_banner_html,
    render_email_html,
)
from app.generation.schemas import ContentChannel, CreativeFormat


IMAGE_URL = "https://cdn.example.test/demo/banner.png?version=1&size=large"


def identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        generation_id="generation_banner_001",
        content_id="content_banner_001",
    )


def test_render_banner_html_composes_image_copy_ad_label_and_click_contract() -> None:
    rendered = render_banner_html(
        {
            "title": "여름 <특가>",
            "body": "제주 숙소를 한눈에 비교하세요.",
            "cta": "숙소 보기",
            "image_url": IMAGE_URL,
        }
    )

    assert 'src="https://cdn.example.test/demo/banner.png?version=1&amp;size=large"' in rendered
    assert "여름 &lt;특가&gt;" in rendered
    assert "제주 숙소를 한눈에 비교하세요." in rendered
    assert "숙소 보기" in rendered
    assert "광고" in rendered
    assert "loopad:click" in rendered
    assert "window.parent.postMessage" in rendered
    assert "loopad-image" in rendered


@pytest.mark.parametrize(
    "image_url",
    [None, "", "/relative/banner.png", "http://cdn.example.test/banner.png"],
)
def test_render_banner_html_rejects_missing_or_non_https_image(
    image_url: str | None,
) -> None:
    with pytest.raises((ArtifactRenderError, ValueError), match="image_url"):
        render_banner_html(
            {
                "title": "여름 특가",
                "body": "호텔을 비교하세요.",
                "cta": "보기",
                "image_url": image_url,
            }
        )


def test_render_email_html_uses_email_safe_image_tracking_and_unsubscribe_contract() -> None:
    rendered = render_email_html(
        {
            "subject": "제주 숙박 프로모션",
            "preheader": "이번 주말 객실을 확인하세요.",
            "body": "무료 취소 가능한 숙소를 비교해보세요.",
            "cta": "호텔 보기",
            "image_url": IMAGE_URL,
        }
    )

    assert rendered.count('role="presentation"') >= 3
    assert 'width="600"' in rendered
    assert 'src="https://cdn.example.test/demo/banner.png?version=1&amp;size=large"' in rendered
    assert 'alt="제주 숙박 프로모션"' in rendered
    assert "이번 주말 객실을 확인하세요." in rendered
    assert "{{redirect_url}}" in rendered
    assert "{{open_pixel_url}}" in rendered
    assert "{{unsubscribe_url}}" in rendered
    assert "광고" in rendered
    assert "수신 거부" in rendered
    assert "display:flex" not in rendered
    assert "display:grid" not in rendered
    assert "<script" not in rendered


@pytest.mark.parametrize(
    ("channel", "values"),
    [
        (
            ContentChannel.EMAIL,
            {
                "subject": '제주 "숙박" <특가>',
                "preheader": "이번 주말 객실을 확인하세요.",
                "body": "무료 취소 가능한 숙소를 비교해보세요.",
                "cta": "호텔 보기",
                "message": None,
                "title": None,
                "image_prompt": "bright suite, no visible text",
                "image_url": IMAGE_URL,
                "landing_url": "https://demo-stay.example.com/summer",
            },
        ),
        (
            ContentChannel.ONSITE_BANNER,
            {
                "subject": None,
                "preheader": None,
                "title": "제주 </script> 특가",
                "body": '따옴표 "포함" 객실을 확인하세요.',
                "cta": "숙소 보기",
                "message": None,
                "image_prompt": "coastal hotel, no visible text",
                "image_url": IMAGE_URL,
                "landing_url": "https://demo-stay.example.com/summer",
            },
        ),
    ],
)
def test_rendered_html_recovers_exact_canonical_source(
    channel: ContentChannel,
    values: dict[str, str | None],
) -> None:
    rendered = (
        render_email_html(values)
        if channel == ContentChannel.EMAIL
        else render_banner_html(values)
    )

    recovered = content_values_from_rendered_html(
        channel=channel,
        html_body=rendered,
    )

    assert recovered == {
        "subject": values.get("subject"),
        "preheader": values.get("preheader"),
        "title": values.get("title"),
        "body": values.get("body"),
        "cta": values.get("cta"),
        "image_url": values.get("image_url"),
        "image_prompt_sha256": image_prompt_sha256(values.get("image_prompt")),
        "landing_url_sha256": hashlib.sha256(
            str(values.get("landing_url")).encode("utf-8")
        ).hexdigest(),
        "creative_contract_sha256": creative_contract_sha256(values),
        "renderer_version": renderer_metadata(channel)["version"],
        "template_version": renderer_metadata(channel)["template_version"],
    }
    assert "image_prompt" not in recovered
    assert "landing_url" not in recovered


def test_rendered_html_source_recovery_is_template_independent() -> None:
    rendered = render_banner_html(
        {
            "title": "제주 숙박 특가",
            "body": "객실을 확인하세요.",
            "cta": "숙소 보기",
            "image_prompt": "coastal hotel, no visible text",
            "image_url": IMAGE_URL,
            "landing_url": "https://demo-stay.example.com/summer",
        }
    )

    recovered = content_values_from_rendered_html(
        channel=ContentChannel.ONSITE_BANNER,
        html_body=rendered.replace("background:#072b63", "background:#082c64", 1),
    )

    assert recovered["body"] == "객실을 확인하세요."
    assert recovered["image_prompt_sha256"] == image_prompt_sha256(
        "coastal hotel, no visible text"
    )


def test_rendered_html_source_keeps_private_values_out_and_reads_v1() -> None:
    image_prompt = "private coastal hotel prompt"
    landing_url = "https://private.example.test/reservation?token=secret"
    rendered = render_banner_html(
        {
            "title": "제주 숙박 특가",
            "body": "객실을 확인하세요.",
            "cta": "숙소 보기",
            "image_prompt": image_prompt,
            "image_url": IMAGE_URL,
            "landing_url": landing_url,
        }
    )
    match = re.search(
        rf'<meta name="{re.escape(CREATIVE_SOURCE_META_NAME)}" content="([^"]+)">',
        rendered,
    )
    assert match is not None
    encoded = match.group(1)
    payload = json.loads(
        base64.urlsafe_b64decode(
            f"{encoded}{'=' * (-len(encoded) % 4)}"
        ).decode("utf-8")
    )
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    assert image_prompt not in serialized_payload
    assert landing_url not in serialized_payload

    payload["schema_version"] = "creative.source.v1"
    payload.pop("creative_contract_sha256")
    payload.pop("renderer_version")
    payload.pop("template_version")
    legacy_encoded = base64.urlsafe_b64encode(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    legacy_rendered = (
        rendered[: match.start(1)]
        + legacy_encoded
        + rendered[match.end(1) :]
    )

    recovered = content_values_from_rendered_html(
        channel=ContentChannel.ONSITE_BANNER,
        html_body=legacy_rendered,
    )

    assert recovered["renderer_version"] == "generation.renderer.v1"
    assert recovered["template_version"] == "banner.overlay.v1"
    assert recovered["creative_contract_sha256"] is None


def test_rendered_html_source_reads_previous_v2_schema() -> None:
    rendered = render_banner_html(
        {
            "title": "제주 숙박 특가",
            "body": "객실을 확인하세요.",
            "cta": "숙소 보기",
            "image_prompt": "coastal hotel, no visible text",
            "image_url": IMAGE_URL,
            "landing_url": "https://demo-stay.example.com/summer",
        }
    )
    match = re.search(
        rf'<meta name="{re.escape(CREATIVE_SOURCE_META_NAME)}" content="([^"]+)">',
        rendered,
    )
    assert match is not None
    encoded = match.group(1)
    payload = json.loads(
        base64.urlsafe_b64decode(
            f"{encoded}{'=' * (-len(encoded) % 4)}"
        ).decode("utf-8")
    )
    payload["schema_version"] = "creative.source.v2"
    payload.pop("creative_contract_sha256")
    previous_encoded = base64.urlsafe_b64encode(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    previous_rendered = (
        rendered[: match.start(1)]
        + previous_encoded
        + rendered[match.end(1) :]
    )

    recovered = content_values_from_rendered_html(
        channel=ContentChannel.ONSITE_BANNER,
        html_body=previous_rendered,
    )

    assert recovered["renderer_version"] == "generation.renderer.v1"
    assert recovered["template_version"] == "banner.overlay.v1"
    assert recovered["creative_contract_sha256"] is None


def test_hierarchical_artifact_keys_and_public_url_strip_root_prefix() -> None:
    artifact_identity = identity()

    assert html_artifact_key(
        base_prefix="genai/",
        identity=artifact_identity,
        creative_format=CreativeFormat.BANNER_HTML,
    ) == (
        "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
        "content_banner_001/creative.banner.html"
    )
    assert image_artifact_key(
        base_prefix="genai/",
        identity=artifact_identity,
        content_type="image/webp",
        image_prompt_sha256="a" * 64,
    ) == (
        "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
        f"content_banner_001/image.{'a' * 64}.webp"
    )
    assert public_asset_url(
        public_base_url="https://cdn.example.test/",
        base_prefix="genai/",
        key=(
            "genai/hotel-client-a/promo_banner_001/generation_banner_001/"
            "content_banner_001/creative.banner.html"
        ),
    ) == (
        "https://cdn.example.test/hotel-client-a/promo_banner_001/"
        "generation_banner_001/content_banner_001/creative.banner.html"
    )


def test_artifact_keys_encode_identity_segments_without_collisions() -> None:
    slash_identity = ArtifactIdentity(
        project_id="호텔/a%",
        promotion_id="promo",
        generation_id="generation",
        content_id="content",
    )
    underscore_identity = ArtifactIdentity(
        project_id="호텔_a%",
        promotion_id="promo",
        generation_id="generation",
        content_id="content",
    )

    slash_key = html_artifact_key(
        base_prefix="genai/",
        identity=slash_identity,
        creative_format=CreativeFormat.EMAIL_HTML,
    )
    underscore_key = html_artifact_key(
        base_prefix="genai/",
        identity=underscore_identity,
        creative_format=CreativeFormat.EMAIL_HTML,
    )

    assert slash_key != underscore_key
    slash_segment = slash_key.split("/")[1]
    underscore_segment = underscore_key.split("/")[1]
    assert slash_segment.startswith("~")
    assert underscore_segment.startswith("~")
    assert "%" not in slash_segment
    assert "/" not in slash_segment


def test_static_publisher_returns_hash_for_rendered_banner() -> None:
    artifact = StaticCreativeArtifactPublisher().publish(
        identity=identity(),
        channel=ContentChannel.ONSITE_BANNER,
        content_values={
            "title": "여름 특가",
            "body": "호텔을 비교하세요.",
            "cta": "보기",
            "image_url": IMAGE_URL,
        },
    )

    assert artifact["artifact_status"] == "published"
    assert artifact["storage_key"].endswith("/creative.banner.html")
    assert len(artifact["sha256"]) == hashlib.sha256().digest_size * 2
    assert artifact["content_type"] == "text/html; charset=utf-8"
