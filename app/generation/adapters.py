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
    ArtifactIdentity,
    ArtifactRenderError,
    HTML_CONTENT_TYPE,
    RECOVERED_IMAGE_PROMPT_PREFIX,
    S3CreativeArtifactPublisher,
    StoredAsset,
    content_values_from_rendered_html,
    creative_format_for_channel,
    html_artifact_key,
    image_artifact_key,
    image_prompt_sha256,
    public_asset_url,
    recovered_image_prompt,
)
from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
)
from app.generation.generator import GeneratedContent
from app.generation.image_prompt_builder import RichImagePromptBuilder
from app.generation.prompt_builder import GenerationPromptInput, PromptBuildResult
from app.generation.schemas import CHANNEL_REQUIRED_FIELDS, ContentChannel, CreativeFormat
from app.generation.source_manifest import (
    MAX_SOURCE_MANIFEST_BYTES,
    SOURCE_MANIFEST_CONTENT_TYPE,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    SourceManifest,
    SourceManifestError,
    SourceManifestIdentityError,
    source_manifest_key,
    source_request_fingerprint,
)
from app.logging import log, duration_ms


EXTERNAL_CONTENT_GENERATOR_VERSION = "dec-c6.external.v3"
DEFAULT_OPENAI_CONTENT_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_GENAI_ASSETS_PUBLIC_BASE_URL = "https://gen-ai.asset.dev.loop-ad.org"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
MAX_HTML_ARTIFACT_BYTES = 1_000_000

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
    def find_creative_content(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
    ) -> Mapping[str, str | None] | None:
        ...

    def find_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        public_url: str | None = None,
    ) -> StoredAsset | None:
        ...

    def store_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        image: "ImageArtifact",
    ) -> StoredAsset:
        ...


class SourceManifestStorage(Protocol):
    def find_source_manifest(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        request_fingerprint: str,
    ) -> GeneratedContent | None:
        ...

    def store_source_manifest(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        request_fingerprint: str,
        content: GeneratedContent,
    ) -> GeneratedContent:
        ...


JsonTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class ImageArtifact:
    data: bytes
    content_type: str = "image/png"

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("generated image bytes must not be empty")
        if self.content_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("generated image content type is unsupported")


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
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client

    def generate_image(self, *, image_prompt: str) -> ImageArtifact:
        client = self._client or _create_gemini_client(
            self._api_key,
            timeout_seconds=self._timeout_seconds,
        )
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
        source_manifest_prefix: str | None = None,
        s3_client: Any | None = None,
    ) -> None:
        public_prefix = base_prefix.strip("/")
        private_prefix = str(source_manifest_prefix or "").strip("/")
        if private_prefix and (
            not public_prefix
            or private_prefix == public_prefix
            or private_prefix.startswith(f"{public_prefix}/")
        ):
            raise ValueError(
                "source manifest prefix must be outside the public asset prefix"
            )
        self._bucket_name = bucket_name
        self._base_prefix = base_prefix
        self._public_base_url = public_base_url
        self._source_manifest_prefix = source_manifest_prefix
        self._s3_client = s3_client or boto3.client("s3")

    def find_source_manifest(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        request_fingerprint: str,
    ) -> GeneratedContent | None:
        key = source_manifest_key(
            base_prefix=self._required_source_manifest_prefix(),
            identity=identity,
        )
        get_object = getattr(self._s3_client, "get_object", None)
        if not callable(get_object):
            raise PermanentGenerationError(
                code="source_manifest_unavailable",
                safe_message="Generated source checkpoint could not be read.",
            )
        try:
            response = get_object(Bucket=self._bucket_name, Key=key)
        except Exception as exc:
            if _is_s3_not_found(exc):
                return None
            if _is_s3_forbidden(exc):
                raise PermanentGenerationError(
                    code="source_manifest_access_denied",
                    safe_message="Generated source checkpoint could not be accessed.",
                ) from exc
            raise

        try:
            body = _validated_s3_source_manifest_body(response)
            manifest = SourceManifest.from_bytes(
                body,
                expected_identity=identity,
                expected_channel=channel,
                expected_request_fingerprint=request_fingerprint,
            )
        except SourceManifestIdentityError as exc:
            raise PermanentGenerationError(
                code="source_manifest_identity_mismatch",
                safe_message=(
                    "Generated source checkpoint did not match the current request."
                ),
            ) from exc
        except (SourceManifestError, ValueError) as exc:
            raise PermanentGenerationError(
                code="source_manifest_invalid",
                safe_message="Generated source checkpoint was invalid.",
            ) from exc
        log.info(
            "provider_request_reused",
            {
                "provider": "s3",
                "endpoint": "get_object",
                "bucket": self._bucket_name,
                "key": key,
                "contentType": SOURCE_MANIFEST_CONTENT_TYPE,
            },
        )
        return manifest.content

    def store_source_manifest(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        request_fingerprint: str,
        content: GeneratedContent,
    ) -> GeneratedContent:
        manifest = SourceManifest(
            identity=identity,
            channel=channel,
            request_fingerprint=request_fingerprint,
            content=content,
        )
        body = manifest.to_bytes()
        content_sha256 = hashlib.sha256(body).hexdigest()
        key = source_manifest_key(
            base_prefix=self._required_source_manifest_prefix(),
            identity=identity,
        )
        started_at = perf_counter()
        log.info(
            "provider_request_prepared",
            {
                "provider": "s3",
                "endpoint": "put_object",
                "bucket": self._bucket_name,
                "key": key,
                "contentType": SOURCE_MANIFEST_CONTENT_TYPE,
            },
        )
        try:
            self._s3_client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=body,
                ContentType=SOURCE_MANIFEST_CONTENT_TYPE,
                CacheControl="no-store",
                Metadata={
                    "sha256": content_sha256,
                    "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
                },
                IfNoneMatch="*",
            )
        except Exception as exc:
            if _is_s3_precondition_failed(exc):
                existing = self.find_source_manifest(
                    identity=identity,
                    channel=channel,
                    request_fingerprint=request_fingerprint,
                )
                if existing is None:
                    raise RetryableGenerationError(
                        code="source_manifest_visibility_delayed",
                        safe_message=(
                            "Generated source checkpoint visibility was delayed."
                        ),
                    ) from exc
                return existing
            if _is_s3_conditional_conflict(exc):
                raise RetryableGenerationError(
                    code="source_manifest_write_conflict",
                    safe_message=(
                        "Generated source checkpoint publication conflicted temporarily."
                    ),
                    status_code=409,
                ) from exc
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
        return manifest.content

    def _required_source_manifest_prefix(self) -> str:
        prefix = str(self._source_manifest_prefix or "").strip()
        if not prefix:
            raise PermanentGenerationError(
                code="source_manifest_unconfigured",
                safe_message="Generated source checkpoint storage was not configured.",
            )
        return prefix

    def store_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        image: ImageArtifact,
    ) -> StoredAsset:
        content_sha256 = hashlib.sha256(image.data).hexdigest()
        key = image_artifact_key(
            base_prefix=self._base_prefix,
            identity=identity,
            content_type=image.content_type,
            image_prompt_sha256=image_prompt_sha256,
        )
        stored_sha256, stored_bytes, stored_content_type = self._store_immutable(
            key=key,
            body=image.data,
            content_type=image.content_type,
            content_sha256=content_sha256,
            accept_existing_content=True,
        )
        image_url = public_asset_url(
            public_base_url=self._public_base_url,
            base_prefix=self._base_prefix,
            key=key,
        )
        return StoredAsset(
            storage_key=key,
            public_url=image_url,
            sha256=stored_sha256,
            bytes=stored_bytes,
            content_type=stored_content_type,
        )

    def find_image(
        self,
        *,
        identity: ArtifactIdentity,
        image_prompt_sha256: str,
        public_url: str | None = None,
    ) -> StoredAsset | None:
        for content_type in ("image/png", "image/jpeg", "image/webp"):
            key = image_artifact_key(
                base_prefix=self._base_prefix,
                identity=identity,
                content_type=content_type,
                image_prompt_sha256=image_prompt_sha256,
            )
            candidate_public_url = public_asset_url(
                public_base_url=self._public_base_url,
                base_prefix=self._base_prefix,
                key=key,
            )
            if public_url is not None and candidate_public_url != public_url:
                continue
            try:
                existing = self._head_existing_object(key=key)
            except Exception as exc:
                if _is_s3_forbidden(exc):
                    return None
                raise
            if existing is None:
                continue
            stored_sha256, stored_bytes, stored_content_type = existing
            if stored_content_type != content_type:
                raise PermanentGenerationError(
                    code="artifact_metadata_invalid",
                    safe_message="Stored generated artifact metadata was invalid.",
                )
            return StoredAsset(
                storage_key=key,
                public_url=candidate_public_url,
                sha256=stored_sha256,
                bytes=stored_bytes,
                content_type=stored_content_type,
            )
        return None

    def find_creative_content(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
    ) -> Mapping[str, str | None] | None:
        if channel == ContentChannel.SMS:
            return None
        key = html_artifact_key(
            base_prefix=self._base_prefix,
            identity=identity,
            creative_format=creative_format_for_channel(channel),
        )
        get_object = getattr(self._s3_client, "get_object", None)
        if not callable(get_object):
            raise PermanentGenerationError(
                code="artifact_source_unavailable",
                safe_message="Stored generated artifact source could not be read.",
            )
        try:
            response = get_object(Bucket=self._bucket_name, Key=key)
        except Exception as exc:
            if _is_s3_not_found(exc):
                return None
            if _is_s3_forbidden(exc):
                raise PermanentGenerationError(
                    code="artifact_source_access_denied",
                    safe_message="Stored generated artifact source could not be accessed.",
                ) from exc
            raise

        try:
            body = _validated_s3_html_body(response)
            html_body = body.decode("utf-8")
            values = content_values_from_rendered_html(
                channel=channel,
                html_body=html_body,
            )
        except (ArtifactRenderError, UnicodeDecodeError, ValueError) as exc:
            raise PermanentGenerationError(
                code="artifact_source_invalid",
                safe_message="Stored generated artifact source was invalid.",
            ) from exc
        log.info(
            "provider_request_reused",
            {
                "provider": "s3",
                "endpoint": "get_object",
                "bucket": self._bucket_name,
                "key": key,
                "contentType": HTML_CONTENT_TYPE,
            },
        )
        return values

    def store_html(
        self,
        *,
        identity: ArtifactIdentity,
        creative_format: CreativeFormat,
        html_body: str,
    ) -> StoredAsset:
        body = html_body.encode("utf-8")
        content_sha256 = hashlib.sha256(body).hexdigest()
        channel = (
            ContentChannel.EMAIL
            if creative_format == CreativeFormat.EMAIL_HTML
            else ContentChannel.ONSITE_BANNER
        )
        try:
            current_values = content_values_from_rendered_html(
                channel=channel,
                html_body=html_body,
            )
            renderer_version = current_values["renderer_version"]
            template_version = current_values["template_version"]
        except (ArtifactRenderError, KeyError, ValueError):
            renderer_version = None
            template_version = None
        key = html_artifact_key(
            base_prefix=self._base_prefix,
            identity=identity,
            creative_format=creative_format,
        )
        try:
            stored_sha256, stored_bytes, stored_content_type = self._store_immutable(
                key=key,
                body=body,
                content_type=HTML_CONTENT_TYPE,
                content_sha256=content_sha256,
                retry_content_conflict=True,
            )
        except RetryableGenerationError as exc:
            if exc.code != "artifact_hash_conflict":
                raise
            existing = self._matching_existing_html(
                key=key,
                creative_format=creative_format,
                html_body=html_body,
            )
            if existing is None:
                raise
            (
                stored_sha256,
                stored_bytes,
                stored_content_type,
                renderer_version,
                template_version,
            ) = existing
        public_url = public_asset_url(
            public_base_url=self._public_base_url,
            base_prefix=self._base_prefix,
            key=key,
        )
        return StoredAsset(
            storage_key=key,
            public_url=public_url,
            sha256=stored_sha256,
            bytes=stored_bytes,
            content_type=stored_content_type,
            renderer_version=renderer_version,
            template_version=template_version,
        )

    def _matching_existing_html(
        self,
        *,
        key: str,
        creative_format: CreativeFormat,
        html_body: str,
    ) -> tuple[str, int, str, str, str] | None:
        get_object = getattr(self._s3_client, "get_object", None)
        if not callable(get_object):
            return None
        try:
            response = get_object(Bucket=self._bucket_name, Key=key)
            existing_body = _validated_s3_html_body(response)
            channel = (
                ContentChannel.EMAIL
                if creative_format == CreativeFormat.EMAIL_HTML
                else ContentChannel.ONSITE_BANNER
            )
            existing_values = content_values_from_rendered_html(
                channel=channel,
                html_body=existing_body.decode("utf-8"),
            )
            current_values = content_values_from_rendered_html(
                channel=channel,
                html_body=html_body,
            )
        except (ArtifactRenderError, UnicodeDecodeError, ValueError):
            return None
        semantic_fields = set(existing_values) - {
            "renderer_version",
            "template_version",
        }
        if semantic_fields != set(current_values) - {
            "renderer_version",
            "template_version",
        } or any(
            existing_values.get(field_name) != current_values.get(field_name)
            for field_name in semantic_fields
        ):
            return None
        log.info(
            "provider_request_reused",
            {
                "provider": "s3",
                "endpoint": "get_object",
                "bucket": self._bucket_name,
                "key": key,
                "contentType": HTML_CONTENT_TYPE,
            },
        )
        return (
            hashlib.sha256(existing_body).hexdigest(),
            len(existing_body),
            HTML_CONTENT_TYPE,
            str(existing_values["renderer_version"]),
            str(existing_values["template_version"]),
        )

    def _store_immutable(
        self,
        *,
        key: str,
        body: bytes,
        content_type: str,
        content_sha256: str,
        accept_existing_content: bool = False,
        retry_content_conflict: bool = False,
    ) -> tuple[str, int, str]:
        started_at = perf_counter()
        log.info(
            "provider_request_prepared",
            {
                "provider": "s3",
                "endpoint": "put_object",
                "bucket": self._bucket_name,
                "key": key,
                "contentType": content_type,
            },
        )
        try:
            self._s3_client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type,
                CacheControl="public, max-age=31536000, immutable",
                Metadata={"sha256": content_sha256},
                IfNoneMatch="*",
            )
        except Exception as exc:
            if _is_s3_conditional_conflict(exc):
                raise RetryableGenerationError(
                    code="artifact_write_conflict",
                    safe_message="Generated artifact publication conflicted temporarily.",
                    status_code=409,
                ) from exc
            if _is_s3_precondition_failed(exc):
                existing = self._head_existing_object(key=key)
                if existing is None:
                    raise RetryableGenerationError(
                        code="artifact_visibility_delayed",
                        safe_message="Generated artifact visibility was delayed temporarily.",
                    ) from exc
                existing_sha256, existing_bytes, existing_content_type = existing
                content_matches = (
                    existing_sha256 == content_sha256
                    and existing_bytes == len(body)
                    and existing_content_type == content_type
                )
                if not content_matches and not (
                    accept_existing_content
                    and existing_content_type == content_type
                ):
                    error_kwargs = {
                        "code": "artifact_hash_conflict",
                        "safe_message": (
                            "An immutable generated artifact already exists with "
                            "different content."
                        ),
                    }
                    if retry_content_conflict:
                        raise RetryableGenerationError(
                            **error_kwargs,
                            status_code=409,
                        ) from exc
                    raise PermanentGenerationError(**error_kwargs) from exc
                log.info(
                    "provider_request_reused",
                    {
                        "provider": "s3",
                        "endpoint": "head_object",
                        "bucket": self._bucket_name,
                        "key": key,
                        "contentType": content_type,
                    },
                )
                return existing
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
        return content_sha256, len(body), content_type

    def _head_existing_object(
        self,
        *,
        key: str,
    ) -> tuple[str, int, str] | None:
        head_object = getattr(self._s3_client, "head_object", None)
        if not callable(head_object):
            return None
        try:
            existing = head_object(Bucket=self._bucket_name, Key=key)
        except Exception as exc:
            if _is_s3_not_found(exc):
                return None
            raise

        metadata = existing.get("Metadata")
        existing_sha256 = (
            str(metadata.get("sha256") or "")
            if isinstance(metadata, Mapping)
            else ""
        )
        existing_content_type = str(existing.get("ContentType") or "").strip()
        try:
            existing_length = int(existing.get("ContentLength"))
        except (TypeError, ValueError) as exc:
            raise PermanentGenerationError(
                code="artifact_metadata_invalid",
                safe_message="Stored generated artifact metadata was invalid.",
            ) from exc
        if (
            not re.fullmatch(r"[0-9a-f]{64}", existing_sha256)
            or existing_length < 0
            or not existing_content_type
        ):
            raise PermanentGenerationError(
                code="artifact_metadata_invalid",
                safe_message="Stored generated artifact metadata was invalid.",
            )
        return existing_sha256, existing_length, existing_content_type


class ExternalContentGenerator:
    version = EXTERNAL_CONTENT_GENERATOR_VERSION

    def __init__(
        self,
        *,
        content_client: ContentTextClient,
        image_client: ImageClient,
        asset_storage: AssetStorage,
        source_manifest_storage: SourceManifestStorage | None = None,
        generate_images: bool = True,
    ) -> None:
        self._content_client = content_client
        self._image_client = image_client
        self._asset_storage = asset_storage
        self._source_manifest_storage = source_manifest_storage
        self._generate_images = generate_images

    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
        artifact_identity: ArtifactIdentity,
    ) -> GeneratedContent:
        content = self.generate_source(
            prompt_input=prompt_input,
            prompt_result=prompt_result,
            option_index=option_index,
            artifact_identity=artifact_identity,
        )
        return self.ensure_image(
            channel=prompt_input.promotion.channel,
            content=content,
            artifact_identity=artifact_identity,
        )

    def generate_source(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
        artifact_identity: ArtifactIdentity,
    ) -> GeneratedContent:
        channel = prompt_input.promotion.channel
        request_fingerprint = source_request_fingerprint(
            identity=artifact_identity,
            prompt_input=prompt_input,
            option_index=option_index,
        )
        if self._source_manifest_storage is not None:
            source_manifest = self._source_manifest_storage.find_source_manifest(
                identity=artifact_identity,
                channel=channel,
                request_fingerprint=request_fingerprint,
            )
            if source_manifest is not None:
                return source_manifest

        recovered_values: Mapping[str, str | None] | None = None
        find_creative_content = getattr(
            self._asset_storage,
            "find_creative_content",
            None,
        )
        if channel != ContentChannel.SMS and callable(find_creative_content):
            recovered_values = find_creative_content(
                identity=artifact_identity,
                channel=channel,
            )

        if recovered_values is not None:
            content = _generated_content_from_recovered_values(
                channel=channel,
                values=recovered_values,
                landing_url=prompt_input.promotion.landing_url,
                prompt_result=prompt_result,
            )
        else:
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
        if self._source_manifest_storage is not None:
            content = self._source_manifest_storage.store_source_manifest(
                identity=artifact_identity,
                channel=channel,
                request_fingerprint=request_fingerprint,
                content=content,
            )
        return content

    def ensure_image(
        self,
        *,
        channel: ContentChannel,
        content: GeneratedContent,
        artifact_identity: ArtifactIdentity,
    ) -> GeneratedContent:
        if channel == ContentChannel.SMS or not self._generate_images:
            return content

        image_prompt = content.image_prompt
        if not image_prompt:
            return content
        prompt_sha256 = image_prompt_sha256(image_prompt)

        find_image = getattr(self._asset_storage, "find_image", None)
        if callable(find_image):
            stored_image = find_image(
                identity=artifact_identity,
                image_prompt_sha256=prompt_sha256,
                public_url=content.image_url,
            )
            if stored_image is not None:
                if (
                    content.image_url is not None
                    and content.image_url != stored_image.public_url
                ):
                    raise PermanentGenerationError(
                        code="artifact_source_image_mismatch",
                        safe_message=(
                            "Stored generated artifact source referenced a different image."
                        ),
                    )
                return replace(
                    content,
                    image_url=stored_image.public_url,
                    image_artifact=stored_image,
                )

        if image_prompt.startswith(RECOVERED_IMAGE_PROMPT_PREFIX):
            raise PermanentGenerationError(
                code="artifact_source_image_missing",
                safe_message="Stored generated artifact image was missing.",
            )

        image = self._image_client.generate_image(image_prompt=image_prompt)
        stored_image = self._asset_storage.store_image(
            identity=artifact_identity,
            image_prompt_sha256=prompt_sha256,
            image=image,
        )
        return replace(
            content,
            image_url=stored_image.public_url,
            image_artifact=stored_image,
        )


def build_external_content_generator(
    settings: Settings,
    *,
    generate_images: bool = True,
) -> ExternalContentGenerator:
    storage = S3AssetStorage(
        bucket_name=settings.data_storage_bucket,
        base_prefix=settings.genai_assets_base_prefix,
        public_base_url=settings.genai_assets_public_base_url,
        source_manifest_prefix=settings.genai_source_manifest_prefix,
    )
    return ExternalContentGenerator(
        content_client=OpenAIResponsesContentClient(
            api_key=settings.openai_api_key,
            model=settings.openai_content_model or DEFAULT_OPENAI_CONTENT_MODEL,
            timeout_seconds=settings.generation_provider_timeout_seconds,
        ),
        image_client=GeminiImageClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_image_model or DEFAULT_GEMINI_IMAGE_MODEL,
            timeout_seconds=settings.generation_provider_timeout_seconds,
        ),
        asset_storage=storage,
        source_manifest_storage=storage,
        generate_images=generate_images,
    )


def build_s3_creative_artifact_publisher(settings: Settings) -> S3CreativeArtifactPublisher:
    return S3CreativeArtifactPublisher(
        storage=S3AssetStorage(
            bucket_name=settings.data_storage_bucket,
            base_prefix=settings.genai_assets_base_prefix,
            public_base_url=settings.genai_assets_public_base_url,
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


def _create_gemini_client(api_key: str, *, timeout_seconds: float) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )


def _gemini_image_config() -> Any:
    from google.genai import types

    return types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(output_mime_type="image/png"),
    )


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
                    data=base64.b64decode(raw_data, validate=True),
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
    values = _values_with_visual_image_prompt(
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


def _generated_content_from_recovered_values(
    *,
    channel: ContentChannel,
    values: Mapping[str, str | None],
    landing_url: str | None,
    prompt_result: PromptBuildResult,
) -> GeneratedContent:
    current_landing_url = _optional_text(landing_url)
    recovered_landing_url_sha256 = _optional_text(
        values.get("landing_url_sha256")
    )
    if (
        not current_landing_url
        or not recovered_landing_url_sha256
        or hashlib.sha256(current_landing_url.encode("utf-8")).hexdigest()
        != recovered_landing_url_sha256
    ):
        raise PermanentGenerationError(
            code="artifact_source_identity_mismatch",
            safe_message=(
                "Stored generated artifact source did not match the current request."
            ),
        )
    del prompt_result
    prompt_sha256 = _optional_text(values.get("image_prompt_sha256"))
    if not prompt_sha256:
        raise PermanentGenerationError(
            code="artifact_source_invalid",
            safe_message="Stored generated artifact source was incomplete.",
        )
    content = GeneratedContent(
        subject=_optional_text(values.get("subject")),
        preheader=_optional_text(values.get("preheader")),
        title=_optional_text(values.get("title")),
        body=_optional_text(values.get("body")),
        cta=_optional_text(values.get("cta")),
        message=_optional_text(values.get("message")),
        image_prompt=recovered_image_prompt(prompt_sha256),
        image_url=_optional_text(values.get("image_url")),
        landing_url=current_landing_url,
        artifact_renderer_version=_optional_text(values.get("renderer_version")),
        artifact_template_version=_optional_text(values.get("template_version")),
    )
    try:
        content.to_record_values(channel)
    except ValueError as exc:
        raise PermanentGenerationError(
            code="artifact_source_invalid",
            safe_message="Stored generated artifact source was incomplete.",
        ) from exc
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


def _values_with_visual_image_prompt(
    *,
    channel: ContentChannel,
    values: Mapping[str, str | None],
    prompt_result: PromptBuildResult,
) -> Mapping[str, str | None]:
    if channel == ContentChannel.SMS:
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


def _validated_s3_html_body(response: Mapping[str, Any]) -> bytes:
    try:
        content_length = int(response.get("ContentLength"))
    except (TypeError, ValueError) as exc:
        raise ValueError("stored HTML content length is invalid") from exc
    if content_length < 0 or content_length > MAX_HTML_ARTIFACT_BYTES:
        raise ValueError("stored HTML content length is outside the allowed range")
    if str(response.get("ContentType") or "").strip() != HTML_CONTENT_TYPE:
        raise ValueError("stored HTML content type is invalid")

    body_value = response.get("Body")
    if isinstance(body_value, (bytes, bytearray)):
        body = bytes(body_value)
    else:
        read = getattr(body_value, "read", None)
        if not callable(read):
            raise ValueError("stored HTML body is unavailable")
        close = getattr(body_value, "close", None)
        try:
            body = bytes(read(MAX_HTML_ARTIFACT_BYTES + 1))
        finally:
            if callable(close):
                close()
    if len(body) != content_length:
        raise ValueError("stored HTML content length does not match its body")

    metadata = response.get("Metadata")
    stored_sha256 = (
        str(metadata.get("sha256") or "")
        if isinstance(metadata, Mapping)
        else ""
    )
    actual_sha256 = hashlib.sha256(body).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", stored_sha256):
        raise ValueError("stored HTML hash metadata is invalid")
    if stored_sha256 != actual_sha256:
        raise ValueError("stored HTML hash does not match its body")
    return body


def _validated_s3_source_manifest_body(response: Mapping[str, Any]) -> bytes:
    try:
        content_length = int(response.get("ContentLength"))
    except (TypeError, ValueError) as exc:
        raise ValueError("stored source manifest content length is invalid") from exc
    if content_length <= 0 or content_length > MAX_SOURCE_MANIFEST_BYTES:
        raise ValueError("stored source manifest size is outside the allowed range")
    if (
        str(response.get("ContentType") or "").strip()
        != SOURCE_MANIFEST_CONTENT_TYPE
    ):
        raise ValueError("stored source manifest content type is invalid")

    body_value = response.get("Body")
    if isinstance(body_value, (bytes, bytearray)):
        body = bytes(body_value)
    else:
        read = getattr(body_value, "read", None)
        if not callable(read):
            raise ValueError("stored source manifest body is unavailable")
        close = getattr(body_value, "close", None)
        try:
            body = bytes(read(MAX_SOURCE_MANIFEST_BYTES + 1))
        finally:
            if callable(close):
                close()
    if len(body) != content_length:
        raise ValueError(
            "stored source manifest content length does not match its body"
        )

    metadata = response.get("Metadata")
    stored_sha256 = (
        str(metadata.get("sha256") or "")
        if isinstance(metadata, Mapping)
        else ""
    )
    schema_version = (
        str(metadata.get("schema_version") or "")
        if isinstance(metadata, Mapping)
        else ""
    )
    if schema_version != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("stored source manifest metadata version is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", stored_sha256):
        raise ValueError("stored source manifest hash metadata is invalid")
    if stored_sha256 != hashlib.sha256(body).hexdigest():
        raise ValueError("stored source manifest hash does not match its body")
    return body


def _is_s3_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    response_metadata = response.get("ResponseMetadata")
    status_code = (
        response_metadata.get("HTTPStatusCode")
        if isinstance(response_metadata, Mapping)
        else None
    )
    error = response.get("Error")
    error_code = error.get("Code") if isinstance(error, Mapping) else None
    return status_code == 404 or str(error_code or "") in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


def _is_s3_precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    response_metadata = response.get("ResponseMetadata")
    status_code = (
        response_metadata.get("HTTPStatusCode")
        if isinstance(response_metadata, Mapping)
        else None
    )
    error = response.get("Error")
    error_code = error.get("Code") if isinstance(error, Mapping) else None
    return status_code == 412 or str(error_code or "") in {
        "412",
        "PreconditionFailed",
    }


def _is_s3_conditional_conflict(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    response_metadata = response.get("ResponseMetadata")
    status_code = (
        response_metadata.get("HTTPStatusCode")
        if isinstance(response_metadata, Mapping)
        else None
    )
    error = response.get("Error")
    error_code = error.get("Code") if isinstance(error, Mapping) else None
    return status_code == 409 and str(error_code or "") in {
        "409",
        "ConditionalRequestConflict",
    }


def _is_s3_forbidden(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    response_metadata = response.get("ResponseMetadata")
    status_code = (
        response_metadata.get("HTTPStatusCode")
        if isinstance(response_metadata, Mapping)
        else None
    )
    error = response.get("Error")
    error_code = error.get("Code") if isinstance(error, Mapping) else None
    return status_code == 403 or str(error_code or "") in {
        "403",
        "AccessDenied",
        "Forbidden",
    }
