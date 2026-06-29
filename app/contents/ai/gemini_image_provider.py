import base64
import json
from typing import Any
from urllib import error, request

from pydantic import SecretStr

from app.contents.ai.image_provider import GeneratedImage


class GeminiImageProvider:
    provider_name = "gemini"

    def __init__(self, api_key: SecretStr | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate_background(self, brief: dict[str, Any]) -> GeneratedImage:
        if self.api_key is None or not self.api_key.get_secret_value().strip():
            raise RuntimeError("GEMINI_API_KEY is required when IMAGE_PROVIDER=gemini")

        prompt = self._build_background_prompt(brief)
        payload = {
            "model": self.model,
            "input": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
            "response_format": {
                "type": "image",
                "mime_type": "image/png",
                "aspect_ratio": "16:9",
            },
        }
        body = json.dumps(payload).encode("utf-8")
        gemini_request = request.Request(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key.get_secret_value(),
            },
        )

        try:
            with request.urlopen(gemini_request, timeout=60) as response:
                response_body = response.read()
        except error.URLError as exc:
            raise RuntimeError("Gemini image generation request failed") from exc

        return self._parse_image_response(response_body)

    def _build_background_prompt(self, brief: dict[str, Any]) -> str:
        return "\n".join(
            [
                "Create a clean, bright ecommerce banner background for a Korean fresh food promotion.",
                "No text.",
                "No letters.",
                "No logo.",
                "No watermark.",
                "Leave enough empty space on the left side for headline and CTA.",
                "Use a fresh grocery delivery mood with soft lighting, delivery box, vegetables,",
                "and subtle premium ecommerce style.",
                "Aspect ratio close to 1200x628.",
            ]
        )

    def _parse_image_response(self, response_body: bytes) -> GeneratedImage:
        payload = json.loads(response_body.decode("utf-8"))
        output_image = payload.get("output_image") or payload.get("outputImage")
        if output_image and output_image.get("data"):
            return GeneratedImage(
                body=base64.b64decode(output_image["data"]),
                content_type=output_image.get("mime_type")
                or output_image.get("mimeType")
                or "image/png",
                provider_name=self.provider_name,
                model=self.model,
            )

        candidates = payload.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData") or part.get("inline_data")
                if not inline_data:
                    continue
                data = inline_data.get("data")
                if not data:
                    continue
                mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
                return GeneratedImage(
                    body=base64.b64decode(data),
                    content_type=mime_type,
                    provider_name=self.provider_name,
                    model=self.model,
                )

        raise RuntimeError("Gemini response did not include image data")
