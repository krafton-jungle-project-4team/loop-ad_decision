from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol

from app.contents.types import GeneratedContentDraft


DEFAULT_GEMINI_IMAGE_MIME_TYPE = "image/png"
GEMINI_VISUAL_PROVIDER = "gemini"
MOCK_VISUAL_PROVIDER = "mock"


@dataclass(frozen=True)
class BannerVisual:
    body: bytes
    content_type: str
    provider: str
    model: str | None = None


class BannerVisualProvider(Protocol):
    def generate(self, draft: GeneratedContentDraft) -> BannerVisual:
        ...


class MockBannerVisualProvider:
    def __init__(self, *, model: str = "mock-visual") -> None:
        self.model = model

    def generate(self, draft: GeneratedContentDraft) -> BannerVisual:
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="628" viewBox="0 0 1200 628">
  <defs>
    <linearGradient id="sea" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#38bdf8"/>
      <stop offset="0.58" stop-color="#0ea5e9"/>
      <stop offset="1" stop-color="#facc15"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="628" fill="url(#sea)"/>
  <circle cx="970" cy="128" r="72" fill="#fef3c7" opacity=".72"/>
  <rect x="716" y="170" width="164" height="250" rx="38" fill="#ffffff" opacity=".82"/>
  <rect x="904" y="208" width="120" height="184" rx="34" fill="#fee2e2" opacity=".86"/>
  <path d="M0 494 C190 438 340 530 520 482 C724 428 854 524 1200 450 L1200 628 L0 628 Z" fill="#fef08a" opacity=".88"/>
  <path d="M90 258 c40 -42 100 -42 140 0 c-40 42 -100 42 -140 0Z" fill="#fb923c" opacity=".82"/>
  <path d="M1040 316 c36 -32 92 -32 128 0 c-36 32 -92 32 -128 0Z" fill="#f472b6" opacity=".78"/>
  <rect x="80" y="88" width="420" height="52" rx="26" fill="#1d4ed8" opacity=".42"/>
  <!-- mock visual for action {draft.recommendation_action_id}, variant {draft.variant_key} -->
</svg>
"""
        return BannerVisual(
            body=svg.encode("utf-8"),
            content_type="image/svg+xml",
            provider=MOCK_VISUAL_PROVIDER,
            model=self.model,
        )


class GeminiBannerVisualProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key must not be empty")
        if not model.strip():
            raise ValueError("Gemini image model must come from env/config and must not be empty")
        self.model = model
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("google-genai package is required for Gemini visuals") from exc
            client = genai.Client(api_key=api_key)
        self.client = client

    def generate(self, draft: GeneratedContentDraft) -> BannerVisual:
        response = self.client.interactions.create(
            model=self.model,
            input=build_gemini_visual_prompt(draft),
            response_format={
                "type": "image",
                "mime_type": DEFAULT_GEMINI_IMAGE_MIME_TYPE,
                "aspect_ratio": "16:9",
            },
        )
        output_image = _read_output_image(response)
        image_data = _read_image_data(output_image)
        content_type = _read_image_mime_type(output_image)

        return BannerVisual(
            body=image_data,
            content_type=content_type,
            provider=GEMINI_VISUAL_PROVIDER,
            model=self.model,
        )


def build_gemini_visual_prompt(draft: GeneratedContentDraft) -> str:
    return "\n".join(
        [
            "Create a wide ecommerce banner visual for a 1200x628 ad.",
            "Generate only background, decorative elements, and generic product-category objects.",
            "Do not render any words, letters, numbers, prices, CTA text, logos, brand marks, UI menus, or labels.",
            "Do not imitate a real branded package unless an approved catalog asset is provided.",
            "Leave clean negative space where app-rendered title, body, and CTA layers can be placed later.",
            f"Visual brief: {draft.image_prompt}",
            f"Content type: {draft.content_type}",
            f"Variant: {draft.variant_key}",
        ]
    )


def _read_output_image(response: Any) -> Any:
    output_image = getattr(response, "output_image", None)
    if output_image is None and isinstance(response, dict):
        output_image = response.get("output_image")
    if output_image is None:
        raise ValueError("Gemini visual response did not include output_image")
    return output_image


def _read_image_data(output_image: Any) -> bytes:
    data = getattr(output_image, "data", None)
    if data is None and isinstance(output_image, dict):
        data = output_image.get("data")
    if isinstance(data, bytes):
        return data
    if not isinstance(data, str) or not data.strip():
        raise ValueError("Gemini visual response did not include image data")
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise ValueError("Gemini visual response image data must be base64") from exc


def _read_image_mime_type(output_image: Any) -> str:
    content_type = getattr(output_image, "mime_type", None)
    if content_type is None and isinstance(output_image, dict):
        content_type = output_image.get("mime_type")
    if not isinstance(content_type, str) or not content_type.strip():
        return DEFAULT_GEMINI_IMAGE_MIME_TYPE
    return content_type.strip()
