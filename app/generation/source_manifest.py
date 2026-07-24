from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from app.generation.artifacts import ArtifactIdentity, artifact_directory
from app.generation.generator import GeneratedContent
from app.generation.prompt_builder import GenerationPromptInput
from app.generation.schemas import ContentChannel


SOURCE_MANIFEST_SCHEMA_VERSION = "generation.source.v1"
SOURCE_MANIFEST_CONTENT_TYPE = "application/json"
SOURCE_MANIFEST_FILENAME = "source.json"
MAX_SOURCE_MANIFEST_BYTES = 256_000
SOURCE_VALUE_FIELDS = (
    "subject",
    "preheader",
    "title",
    "body",
    "cta",
    "message",
    "image_prompt",
    "image_url",
    "landing_url",
    "artifact_renderer_version",
    "artifact_template_version",
)


class SourceManifestError(ValueError):
    """A malformed or unsupported private source manifest."""


class SourceManifestIdentityError(SourceManifestError):
    """A manifest that belongs to a different immutable generation input."""


@dataclass(frozen=True, slots=True)
class SourceManifest:
    identity: ArtifactIdentity
    channel: ContentChannel
    request_fingerprint: str
    content: GeneratedContent

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.request_fingerprint):
            raise SourceManifestError(
                "source manifest request fingerprint must be SHA-256"
            )
        if self.content.image_artifact is not None:
            raise SourceManifestError(
                "source manifest must not contain generated image bytes"
            )
        try:
            self.content.to_record_values(self.channel)
        except ValueError as exc:
            raise SourceManifestError(
                "source manifest content is incomplete"
            ) from exc

    def to_bytes(self) -> bytes:
        payload = {
            "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
            "identity": {
                "project_id": self.identity.project_id,
                "promotion_id": self.identity.promotion_id,
                "generation_id": self.identity.generation_id,
                "content_id": self.identity.content_id,
            },
            "channel": self.channel.value,
            "request_fingerprint": self.request_fingerprint,
            "source": {
                field_name: getattr(self.content, field_name)
                for field_name in SOURCE_VALUE_FIELDS
            },
        }
        body = _canonical_json_bytes(payload)
        if len(body) > MAX_SOURCE_MANIFEST_BYTES:
            raise SourceManifestError("source manifest exceeds the size limit")
        return body

    @classmethod
    def from_bytes(
        cls,
        body: bytes,
        *,
        expected_identity: ArtifactIdentity,
        expected_channel: ContentChannel,
        expected_request_fingerprint: str,
    ) -> SourceManifest:
        if not body or len(body) > MAX_SOURCE_MANIFEST_BYTES:
            raise SourceManifestError("source manifest body size is invalid")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceManifestError("source manifest JSON is invalid") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "identity",
            "channel",
            "request_fingerprint",
            "source",
        }:
            raise SourceManifestError("source manifest shape is invalid")
        if payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise SourceManifestError("source manifest version is unsupported")

        identity = _manifest_identity(payload.get("identity"))
        request_fingerprint = str(payload.get("request_fingerprint") or "")
        if (
            identity != expected_identity
            or request_fingerprint != expected_request_fingerprint
        ):
            raise SourceManifestIdentityError(
                "source manifest does not match the current generation input"
            )

        try:
            channel = ContentChannel(str(payload.get("channel") or ""))
        except ValueError as exc:
            raise SourceManifestError("source manifest channel is invalid") from exc
        if channel != expected_channel:
            raise SourceManifestIdentityError(
                "source manifest does not match the current generation channel"
            )
        source = payload.get("source")
        if not isinstance(source, dict) or set(source) != set(SOURCE_VALUE_FIELDS):
            raise SourceManifestError("source manifest fields are invalid")
        values = {
            field_name: _optional_canonical_text(source.get(field_name))
            for field_name in SOURCE_VALUE_FIELDS
        }
        return cls(
            identity=identity,
            channel=channel,
            request_fingerprint=request_fingerprint,
            content=GeneratedContent(**values),
        )


def source_request_fingerprint(
    *,
    identity: ArtifactIdentity,
    prompt_input: GenerationPromptInput,
    option_index: int,
) -> str:
    if option_index < 1:
        raise ValueError("source manifest option_index must be positive")
    request = prompt_input.request.model_dump(mode="json")
    if prompt_input.request.offer_set_id is None:
        # Keep legacy source fingerprints byte-for-byte stable after adding the
        # optional offer-set selection contract.
        request.pop("offer_set_id", None)
        request.pop("expected_catalog_id", None)
        request.pop("expected_catalog_version", None)
    segment_ids = request.get("segment_ids")
    if isinstance(segment_ids, list):
        request["segment_ids"] = sorted(str(value) for value in segment_ids)
    promotion = asdict(prompt_input.promotion)
    promotion["channel"] = prompt_input.promotion.channel.value
    promotion = {
        key: value
        for key, value in promotion.items()
        if value is not None or key not in {"offer_type", "landing_type"}
    }
    target_segment = asdict(prompt_input.target_segment)
    # These contract-only raw snapshots are already represented by the merged
    # generation brief and must not invalidate pre-RAG private checkpoints.
    target_segment.pop("source_content_brief_json", None)
    target_segment.pop("data_evidence_json", None)
    payload = {
        "schema_version": "generation.source.request.v1",
        "identity": {
            "project_id": identity.project_id,
            "promotion_id": identity.promotion_id,
            "generation_id": identity.generation_id,
            "content_id": identity.content_id,
        },
        "option_index": option_index,
        "request": request,
        "promotion": promotion,
        "target_segment": target_segment,
    }
    if prompt_input.brand_context is not None:
        payload["brand_context"] = prompt_input.brand_context.to_snapshot()
    if prompt_input.request.offer_set_id is not None:
        offer_catalog = prompt_input.offer_catalog
        if not isinstance(offer_catalog, Mapping):
            raise ValueError(
                "source manifest offer_set_id requires a snapshotted offer catalog"
            )
        catalog_sha256 = _optional_canonical_text(
            offer_catalog.get("catalog_sha256")
        )
        if catalog_sha256 is None:
            catalog_sha256 = hashlib.sha256(
                _canonical_json_bytes(offer_catalog)
            ).hexdigest()
        payload["offer_selection"] = {
            "offer_set_id": prompt_input.request.offer_set_id,
            "catalog_id": _required_canonical_text(
                offer_catalog.get("catalog_id")
            ),
            "catalog_version": _required_canonical_text(
                offer_catalog.get("catalog_version")
            ),
            "catalog_sha256": catalog_sha256,
        }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def source_manifest_key(
    *,
    base_prefix: str,
    identity: ArtifactIdentity,
) -> str:
    directory = artifact_directory(base_prefix=base_prefix, identity=identity)
    return f"{directory}/{SOURCE_MANIFEST_FILENAME}"


def _manifest_identity(value: object) -> ArtifactIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "project_id",
        "promotion_id",
        "generation_id",
        "content_id",
    }:
        raise SourceManifestError("source manifest identity is invalid")
    try:
        return ArtifactIdentity(
            project_id=_required_canonical_text(value.get("project_id")),
            promotion_id=_required_canonical_text(value.get("promotion_id")),
            generation_id=_required_canonical_text(value.get("generation_id")),
            content_id=_required_canonical_text(value.get("content_id")),
        )
    except ValueError as exc:
        raise SourceManifestError("source manifest identity is invalid") from exc


def _required_canonical_text(value: object) -> str:
    text = _optional_canonical_text(value)
    if text is None:
        raise ValueError("source manifest value is required")
    return text


def _optional_canonical_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceManifestError("source manifest text is not canonical")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SourceManifestError("source manifest input is not JSON-safe") from exc
    return encoded.encode("utf-8")
