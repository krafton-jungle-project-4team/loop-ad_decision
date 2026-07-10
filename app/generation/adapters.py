from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol

import boto3

from app.config import Settings
from app.generation.artifacts import (
    HTML_CONTENT_TYPE,
    S3CreativeArtifactPublisher,
    html_artifact_key,
    public_asset_url,
)
from app.generation.generator import GeneratedContent
from app.generation.image_prompt_builder import RichImagePromptBuilder
from app.generation.prompt_builder import GenerationPromptInput, PromptBuildResult
from app.generation.schemas import CHANNEL_REQUIRED_FIELDS, ContentChannel, CreativeFormat
from app.logging import log, duration_ms


EXTERNAL_CONTENT_GENERATOR_VERSION = "dec-c6.external.v3"
DEFAULT_OPENAI_CONTENT_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL = "https://gen-ai.asset.dev.loop-ad.org"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

TEXT_FIELD_NAMES = (
    "subject",
    "preheader",
    "title",
    "body",
    "cta",
    "message",
    "image_prompt",
)


class ContentTextClient(Protocol):
    def generate_content(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> Mapping[str, str | None]:
        ...


class ImageClient(Protocol):
    def generate_image(self, *, image_prompt: str) -> "ImageArtifact":
        ...


class AssetStorage(Protocol):
    def store_image(
        self,
        *,
        content_id: str,
        image: "ImageArtifact",
    ) -> str:
        ...

    def store_html(
        self,
        *,
        content_id: str,
        creative_format: CreativeFormat,
        html_body: str,
    ) -> Mapping[str, Any]:
        ...


JsonTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class ImageArtifact:
    data: bytes
    content_type: str = "image/png"


class OpenAIResponsesContentClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENAI_CONTENT_MODEL,
        endpoint: str = OPENAI_RESPONSES_URL,
        timeout_seconds: float = 30.0,
        transport: JsonTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._transport = transport or _post_json

    def generate_content(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> Mapping[str, str | None]:
        payload = {
            "model": self._model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _system_instruction(prompt_input.promotion.channel),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _user_instruction(
                                prompt_input=prompt_input,
                                prompt_result=prompt_result,
                                option_index=option_index,
                            ),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "content_candidate",
                    "strict": True,
                    "schema": _content_schema(prompt_input.promotion.channel),
                }
            },
            "max_output_tokens": 900,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        started_at = perf_counter()
        log.info(
            "provider_request_prepared",
            {
                "provider": "openai",
                "endpoint": self._endpoint,
                "model": self._model,
                "channel": prompt_input.promotion.channel.value,
                "optionIndex": option_index,
                "request": payload,
            },
        )
        try:
            response_payload = self._transport(
                self._endpoint,
                headers,
                payload,
                self._timeout_seconds,
            )
            content_payload = _parse_output_json(response_payload)
        except Exception as exc:
            log.warn(
                "provider_request_failed",
                {
                    "provider": "openai",
                    "endpoint": self._endpoint,
                    "model": self._model,
                    "err": exc,
                    "durationMs": duration_ms(started_at),
                },
            )
            raise
        log.info(
            "provider_request_completed",
            {
                "provider": "openai",
                "endpoint": self._endpoint,
                "model": self._model,
                "durationMs": duration_ms(started_at),
            },
        )
        return {
            field_name: _optional_text(content_payload.get(field_name))
            for field_name in TEXT_FIELD_NAMES
        }


class GeminiImageClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GEMINI_IMAGE_MODEL,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client

    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        client = self._client or _create_gemini_client(self._api_key)
        started_at = perf_counter()
        log.info("provider_request_prepared", {"provider": "gemini", "model": self._model, "request": {"imagePrompt": image_prompt}})
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=image_prompt,
                config=_gemini_image_config(),
            )
            artifact = _extract_gemini_image(response)
        except Exception as exc:
            log.warn("provider_request_failed", {"provider": "gemini", "model": self._model, "err": exc, "durationMs": duration_ms(started_at)})
            raise
        log.info(
            "provider_request_completed",
            {
                "provider": "gemini",
                "model": self._model,
                "contentType": artifact.content_type,
                "byteLength": len(artifact.data),
                "durationMs": duration_ms(started_at),
            },
        )
        return artifact


class S3AssetStorage:
    def __init__(
        self,
        *,
        bucket_name: str,
        base_prefix: str,
        public_base_url: str = DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL,
        s3_client: Any | None = None,
    ) -> None:
        self._bucket_name = bucket_name
        self._base_prefix = base_prefix
        self._public_base_url = public_base_url
        self._s3_client = s3_client or boto3.client("s3")

    def store_image(
        self,
        *,
        content_id: str,
        image: ImageArtifact,
    ) -> str:
        key = _asset_key(
            base_prefix=self._base_prefix,
            content_id=content_id,
            content_type=image.content_type,
        )
        started_at = perf_counter()
        log.info(
            "provider_request_prepared",
            {
                "provider": "s3",
                "endpoint": "put_object",
                "bucket": self._bucket_name,
                "key": key,
                "contentType": image.content_type,
            },
        )
        try:
            self._s3_client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=image.data,
                ContentType=image.content_type,
                CacheControl="public, max-age=31536000, immutable",
            )
        except Exception as exc:
            log.warn(
                "provider_request_failed",
                {
                    "provider": "s3",
                    "endpoint": "put_object",
                    "bucket": self._bucket_name,
                    "key": key,
                    "err": exc,
                    "durationMs": duration_ms(started_at),
                },
            )
            raise
        image_url = _public_asset_url(
            public_base_url=self._public_base_url,
            base_prefix=self._base_prefix,
            key=key,
        )
        log.info(
            "provider_request_completed",
            {
                "provider": "s3",
                "endpoint": "put_object",
                "bucket": self._bucket_name,
                "key": key,
                "durationMs": duration_ms(started_at),
            },
        )
        return image_url

    def store_html(
        self,
        *,
        content_id: str,
        creative_format: CreativeFormat,
        html_body: str,
    ) -> Mapping[str, Any]:
        key = html_artifact_key(
            base_prefix=self._base_prefix,
            content_id=content_id,
            creative_format=creative_format,
        )
        body = html_body.encode("utf-8")
        started_at = perf_counter()
        log.info(
            "provider_request_prepared",
            {
                "provider": "s3",
                "endpoint": "put_object",
                "bucket": self._bucket_name,
                "key": key,
                "contentType": HTML_CONTENT_TYPE,
            },
        )
        try:
            self._s3_client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=body,
                ContentType=HTML_CONTENT_TYPE,
                CacheControl="public, max-age=31536000, immutable",
            )
        except Exception as exc:
            log.warn(
                "provider_request_failed",
                {
                    "provider": "s3",
                    "endpoint": "put_object",
                    "bucket": self._bucket_name,
                    "key": key,
                    "err": exc,
                    "durationMs": duration_ms(started_at),
                },
            )
            raise
        public_url = public_asset_url(
            public_base_url=self._public_base_url,
            base_prefix=self._base_prefix,
            key=key,
        )
        log.info(
            "provider_request_completed",
            {
                "provider": "s3",
                "endpoint": "put_object",
                "bucket": self._bucket_name,
                "key": key,
                "durationMs": duration_ms(started_at),
            },
        )
        metadata: dict[str, Any] = {
            "storage_key": key,
            "public_url": public_url,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "content_type": HTML_CONTENT_TYPE,
        }
        if creative_format == CreativeFormat.BANNER_HTML:
            metadata["width"] = 320
            metadata["height"] = 100
        return metadata


class ExternalContentGenerator:
    version = EXTERNAL_CONTENT_GENERATOR_VERSION

    def __init__(
        self,
        *,
        content_client: ContentTextClient,
        image_client: ImageClient,
        asset_storage: AssetStorage,
        generate_images: bool = True,
    ) -> None:
        self._content_client = content_client
        self._image_client = image_client
        self._asset_storage = asset_storage
        self._generate_images = generate_images

    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> GeneratedContent:
        channel = prompt_input.promotion.channel
        values = dict(
            self._content_client.generate_content(
                prompt_input=prompt_input,
                prompt_result=prompt_result,
                option_index=option_index,
            )
        )
        content = _generated_content_from_values(
            channel=channel,
            values=values,
            landing_url=prompt_input.promotion.landing_url,
            prompt_result=prompt_result,
        )

        if channel != ContentChannel.ONSITE_BANNER or not self._generate_images:
            return content

        image_prompt = content.image_prompt
        if not image_prompt:
            return content

        image = self._image_client.generate_image(image_prompt=image_prompt)
        image_url = self._asset_storage.store_image(
            content_id=_content_id(prompt_input, option_index),
            image=image,
        )
        return replace(content, image_url=image_url)


def build_external_content_generator(
    settings: Settings,
    *,
    generate_images: bool = True,
) -> ExternalContentGenerator:
    return ExternalContentGenerator(
        content_client=OpenAIResponsesContentClient(
            api_key=settings.openai_api_key,
            model=settings.openai_content_model or DEFAULT_OPENAI_CONTENT_MODEL,
        ),
        image_client=GeminiImageClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_image_model or DEFAULT_GEMINI_IMAGE_MODEL,
        ),
        asset_storage=S3AssetStorage(
            bucket_name=settings.data_storage_bucket,
            base_prefix=settings.genai_assets_base_prefix,
        ),
        generate_images=generate_images,
    )


def build_s3_creative_artifact_publisher(settings: Settings) -> S3CreativeArtifactPublisher:
    return S3CreativeArtifactPublisher(
        storage=S3AssetStorage(
            bucket_name=settings.data_storage_bucket,
            base_prefix=settings.genai_assets_base_prefix,
        )
    )


def _post_json(
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"openai content generation failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("openai content generation failed") from exc


def _create_gemini_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


def _gemini_image_config() -> Any:
    from google.genai import types

    return types.GenerateContentConfig(response_modalities=["IMAGE"])


def _extract_gemini_image(response: Any) -> ImageArtifact:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            raw_data = getattr(inline_data, "data", None)
            if raw_data is None:
                continue
            content_type = getattr(inline_data, "mime_type", None) or "image/png"
            if isinstance(raw_data, bytes):
                return ImageArtifact(data=raw_data, content_type=content_type)
            if isinstance(raw_data, str):
                return ImageArtifact(
                    data=base64.b64decode(raw_data),
                    content_type=content_type,
                )
    raise RuntimeError("gemini image generation returned no image")


def _parse_output_json(response_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    output_text = response_payload.get("output_text")
    if not isinstance(output_text, str):
        output_text = _extract_output_text(response_payload)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        log.warn("provider_response_invalid", {"provider": "openai", "err": exc})
        raise RuntimeError("openai content generation returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        log.warn("provider_response_invalid", {"provider": "openai", "body": parsed})
        raise RuntimeError("openai content generation returned a non-object payload")
    return parsed


def _extract_output_text(response_payload: Mapping[str, Any]) -> str:
    for output_item in response_payload.get("output", []) or []:
        if not isinstance(output_item, dict):
            continue
        for part in output_item.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return part["text"]
    raise RuntimeError("openai content generation returned no output text")


def _generated_content_from_values(
    *,
    channel: ContentChannel,
    values: Mapping[str, str | None],
    landing_url: str | None,
    prompt_result: PromptBuildResult,
) -> GeneratedContent:
    if not landing_url:
        raise ValueError("promotion.landing_url is required to generate content")
    values = _values_with_banner_image_prompt(
        channel=channel,
        values=values,
        prompt_result=prompt_result,
    )
    values = _values_with_sms_redirect_placeholder(channel=channel, values=values)
    content = GeneratedContent(
        subject=values.get("subject"),
        preheader=values.get("preheader"),
        title=values.get("title"),
        body=values.get("body"),
        cta=values.get("cta"),
        message=values.get("message"),
        image_prompt=values.get("image_prompt"),
        landing_url=landing_url,
    )
    content.to_record_values(channel)
    return content


def _values_with_sms_redirect_placeholder(
    *,
    channel: ContentChannel,
    values: Mapping[str, str | None],
) -> Mapping[str, str | None]:
    if channel != ContentChannel.SMS:
        return values

    message = _optional_text(values.get("message"))
    if not message or "{{redirect_url}}" in message:
        return values
    return {**values, "message": f"{message} {{{{redirect_url}}}}"}


def _values_with_banner_image_prompt(
    *,
    channel: ContentChannel,
    values: Mapping[str, str | None],
    prompt_result: PromptBuildResult,
) -> Mapping[str, str | None]:
    if channel != ContentChannel.ONSITE_BANNER:
        return values

    image_prompt = RichImagePromptBuilder().build(
        prompt_result,
        provider_visual_concept=_optional_text(values.get("image_prompt")),
    )
    return {**values, "image_prompt": image_prompt}


def _system_instruction(channel: ContentChannel) -> str:
    return (
        "You generate hotel booking advertisement content. "
        "Write customer-facing copy values in natural Korean for a Korean "
        "hotel booking audience. "
        "Keep JSON field names in English, but write subject, preheader, "
        "title, body, cta, and message values in Korean. "
        "The image_prompt value may stay in English for the image generator "
        "and must not request visible text in the image. "
        f"Return only the JSON fields for the {channel.value} channel contract. "
        "Return non-empty strings for every required channel field. "
        "Do not generate, infer, or override landing URLs. "
        "Do not include legacy naming, marketplace language, or unrelated commerce terms."
    )


def _user_instruction(
    *,
    prompt_input: GenerationPromptInput,
    prompt_result: PromptBuildResult,
    option_index: int,
) -> str:
    channel = prompt_input.promotion.channel
    required_fields = ", ".join(_required_text_fields(channel))
    return "\n".join(
        [
            prompt_result.generation_prompt,
            f"Content option number: {option_index}",
            f"Fixed landing URL assigned by Loop-Ad: {prompt_input.promotion.landing_url or ''}",
            "Do not return landing_url in the JSON response.",
            "Do not copy English source text verbatim; adapt it into natural Korean.",
            f"Required JSON string fields: {required_fields}.",
            "Return concise Korean hotel booking copy. Return JSON only.",
        ]
    )


def _content_schema(channel: ContentChannel) -> dict[str, Any]:
    required_text_fields = set(_required_text_fields(channel))
    properties = {
        field_name: {"type": "string"}
        if field_name in required_text_fields
        else {"type": ["string", "null"]}
        for field_name in TEXT_FIELD_NAMES
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(TEXT_FIELD_NAMES),
        "description": f"Generated content fields for {channel.value}.",
    }


def _required_text_fields(channel: ContentChannel) -> tuple[str, ...]:
    return tuple(
        field_name
        for field_name in CHANNEL_REQUIRED_FIELDS[channel]
        if field_name in TEXT_FIELD_NAMES
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _asset_key(
    *,
    base_prefix: str,
    content_id: str,
    content_type: str,
) -> str:
    prefix = base_prefix.strip("/")
    extension = _image_extension(content_type)
    filename = f"{_safe_asset_name(content_id)}.{extension}"
    path = f"generated/{filename}"
    return f"{prefix}/{path}" if prefix else path


def _public_asset_url(
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


def _image_extension(content_type: str) -> str:
    if content_type == "image/jpeg":
        return "jpg"
    if content_type == "image/webp":
        return "webp"
    return "png"


def _content_id(prompt_input: GenerationPromptInput, option_index: int) -> str:
    channel = prompt_input.promotion.channel
    channel_slug = "banner" if channel == ContentChannel.ONSITE_BANNER else channel.value
    segment_slug = prompt_input.target_segment.content_slug or _safe_asset_name(
        prompt_input.target_segment.segment_id.removeprefix("seg_")
    )
    return f"content_{channel_slug}_{segment_slug}_{option_index:03d}"


def _safe_asset_name(value: str) -> str:
    safe_value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._-")
    return safe_value or "content"
