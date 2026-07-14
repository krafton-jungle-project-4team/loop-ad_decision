from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import boto3

from app.generation.brand_context import (
    BrandContextSnapshot,
    RetrievedBrandDocument,
)
from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
)
from app.generation.schemas import ContentChannel
from app.logging import log


BRAND_CONTEXT_POINTER_SCHEMA_VERSION = "loopad.brand-context-pointer.v1"
BRAND_CONTEXT_MANIFEST_SCHEMA_VERSION = "loopad.brand-context-manifest.v1"
MAX_POINTER_BYTES = 16_384
MAX_MANIFEST_BYTES = 2_000_000
MAX_GUIDE_BYTES = 256_000
MAX_BRAND_KIT_BYTES = 256_000
MAX_ASSET_VALIDATION_BYTES = 20_000_000

_PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HEX_COLOUR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}\b")
_MARKDOWN_BULLET_PATTERN = re.compile(r"^\s*[-*]\s+(.+?)\s*$")


class S3BrandContextLoader:
    """Resolve and load immutable brand context through its private S3 manifest."""

    def __init__(
        self,
        *,
        bucket_name: str,
        base_prefix: str,
        s3_client: Any | None = None,
    ) -> None:
        self._bucket_name = _required_text(bucket_name, "bucket_name")
        self._base_prefix = _normalised_prefix(base_prefix)
        self._s3_client = s3_client
        self._manifest_cache: dict[tuple[str, str, str], Mapping[str, Any]] = {}

    def resolve_snapshot(self, *, project_id: str) -> BrandContextSnapshot | None:
        project_id = _validated_project_id(project_id)
        pointer_key = f"{self._base_prefix}{project_id}/current.json"
        pointer_bytes = self._read_object(
            pointer_key,
            max_bytes=MAX_POINTER_BYTES,
            optional=True,
            expected_content_type="application/json; charset=utf-8",
        )
        if pointer_bytes is None:
            log.info(
                "brand_context_not_configured",
                {"projectId": project_id, "pointerKey": pointer_key},
            )
            return None
        pointer = _json_object(pointer_bytes, label="brand context pointer")
        _require_schema(
            pointer,
            BRAND_CONTEXT_POINTER_SCHEMA_VERSION,
            label="brand context pointer",
        )
        _require_equal(pointer, "project_id", project_id, label="brand context pointer")
        context_version = _required_text(
            pointer.get("context_version"),
            "pointer.context_version",
        )
        manifest_key = _required_text(
            pointer.get("manifest_key"),
            "pointer.manifest_key",
        )
        self._validate_project_key(
            manifest_key,
            project_id=project_id,
            expected_parent="manifests",
        )
        expected_manifest_sha256 = _required_sha256(
            pointer.get("manifest_sha256"),
            "pointer.manifest_sha256",
        )
        manifest = self._load_manifest(
            project_id=project_id,
            context_version=context_version,
            manifest_key=manifest_key,
            expected_sha256=expected_manifest_sha256,
        )
        snapshot = BrandContextSnapshot(
            context_version=context_version,
            manifest_key=manifest_key,
            manifest_sha256=expected_manifest_sha256,
            guide_version=_manifest_guide_version(manifest, context_version),
            asset_manifest_version=context_version,
            catalog_version=_manifest_catalog_version(manifest, context_version),
        )
        log.info(
            "brand_context_snapshot_resolved",
            {
                "projectId": project_id,
                "contextVersion": context_version,
                "manifestKey": manifest_key,
                "manifestSha256": expected_manifest_sha256,
            },
        )
        return snapshot

    def load_documents(
        self,
        *,
        project_id: str,
        snapshot: BrandContextSnapshot,
        channel: ContentChannel,
    ) -> tuple[RetrievedBrandDocument, ...]:
        project_id = _validated_project_id(project_id)
        manifest = self._load_manifest(
            project_id=project_id,
            context_version=snapshot.context_version,
            manifest_key=snapshot.manifest_key,
            expected_sha256=snapshot.manifest_sha256,
        )
        brand_kit_entry = _required_mapping(manifest, "brand_kit")
        brand_kit_bytes = self._read_verified_reference(
            brand_kit_entry,
            project_id=project_id,
            max_bytes=MAX_BRAND_KIT_BYTES,
            label="brand kit",
        )
        brand_kit = _json_object(brand_kit_bytes, label="brand kit")
        brand_rules = _brand_kit_rules(brand_kit)

        documents: list[RetrievedBrandDocument] = []
        for raw_guide in _mapping_list(manifest, "guidelines"):
            applies_to = _string_list(raw_guide.get("applies_to"))
            if not bool(raw_guide.get("required")) or channel.value not in applies_to:
                continue
            guide_id = _required_text(raw_guide.get("guide_id"), "guide.guide_id")
            version = _required_text(raw_guide.get("version"), "guide.version")
            guide_bytes = self._read_verified_reference(
                raw_guide,
                project_id=project_id,
                max_bytes=MAX_GUIDE_BYTES,
                label=f"guide {guide_id}",
            )
            guide_text = _utf8_text(guide_bytes, label=f"guide {guide_id}")
            metadata: dict[str, Any] = {
                "required": True,
                "channels": applies_to,
                **brand_rules,
            }
            forbidden_terms = _markdown_section_bullets(guide_text, "금지 문구")
            if forbidden_terms:
                metadata["forbidden_terms"] = forbidden_terms
            documents.append(
                RetrievedBrandDocument(
                    document_id=_document_id(
                        project_id,
                        "brand_guide",
                        guide_id,
                        version,
                    ),
                    source_kind="brand_guide",
                    source_id=guide_id,
                    source_version=version,
                    document_text=guide_text,
                    metadata=metadata,
                    s3_key=_required_text(raw_guide.get("s3_key"), "guide.s3_key"),
                )
            )

        if not documents:
            raise PermanentGenerationError(
                code="brand_guide_unavailable",
                safe_message="The snapshotted brand guide is unavailable.",
            )

        selected_asset_id: str | None = None
        if channel != ContentChannel.SMS:
            asset = _select_representative_asset(_mapping_list(manifest, "assets"))
            if asset is None:
                raise PermanentGenerationError(
                    code="brand_asset_unavailable",
                    safe_message="The snapshotted approved brand asset is unavailable.",
                )
            selected_asset_id = _required_text(asset.get("asset_id"), "asset.asset_id")
            version = _required_text(asset.get("version"), "asset.version")
            self._read_verified_reference(
                asset,
                project_id=project_id,
                max_bytes=MAX_ASSET_VALIDATION_BYTES,
                label=f"asset {selected_asset_id}",
            )
            documents.append(
                RetrievedBrandDocument(
                    document_id=_document_id(
                        project_id,
                        "brand_asset",
                        selected_asset_id,
                        version,
                    ),
                    source_kind="brand_asset",
                    source_id=selected_asset_id,
                    source_version=version,
                    document_text=_asset_document_text(asset),
                    metadata=dict(asset),
                    s3_key=_required_text(asset.get("s3_key"), "asset.s3_key"),
                )
            )

        log.info(
            "brand_context_documents_loaded",
            {
                "projectId": project_id,
                "contextVersion": snapshot.context_version,
                "channel": channel.value,
                "documentCount": len(documents),
                "selectedAssetId": selected_asset_id,
            },
        )
        return tuple(documents)

    def _load_manifest(
        self,
        *,
        project_id: str,
        context_version: str,
        manifest_key: str,
        expected_sha256: str,
    ) -> Mapping[str, Any]:
        cache_key = (project_id, context_version, expected_sha256)
        cached = self._manifest_cache.get(cache_key)
        if cached is not None:
            return cached
        self._validate_project_key(
            manifest_key,
            project_id=project_id,
            expected_parent="manifests",
        )
        manifest_bytes = self._read_object(
            manifest_key,
            max_bytes=MAX_MANIFEST_BYTES,
            expected_content_type="application/json; charset=utf-8",
        )
        assert manifest_bytes is not None
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256:
            raise PermanentGenerationError(
                code="brand_context_manifest_checksum_mismatch",
                safe_message="The brand context manifest checksum did not match.",
            )
        manifest = _json_object(manifest_bytes, label="brand context manifest")
        _require_schema(
            manifest,
            BRAND_CONTEXT_MANIFEST_SCHEMA_VERSION,
            label="brand context manifest",
        )
        _require_equal(manifest, "project_id", project_id, label="brand context manifest")
        _require_equal(
            manifest,
            "context_version",
            context_version,
            label="brand context manifest",
        )
        self._validate_manifest_references(manifest, project_id=project_id)
        self._manifest_cache[cache_key] = manifest
        return manifest

    def _validate_manifest_references(
        self,
        manifest: Mapping[str, Any],
        *,
        project_id: str,
    ) -> None:
        references = [
            _required_mapping(manifest, "brand_kit"),
            *_mapping_list(manifest, "guidelines"),
            *_mapping_list(manifest, "assets"),
            *_mapping_list(manifest, "catalogs"),
        ]
        seen: set[str] = set()
        for reference in references:
            key = _required_text(reference.get("s3_key"), "manifest reference s3_key")
            self._validate_project_key(key, project_id=project_id)
            if key in seen:
                raise PermanentGenerationError(
                    code="brand_context_manifest_invalid",
                    safe_message="The brand context manifest was invalid.",
                )
            seen.add(key)
            _required_sha256(reference.get("sha256"), "manifest reference sha256")
            _required_nonnegative_int(
                reference.get("byte_size"),
                "manifest reference byte_size",
            )
            _required_text(
                reference.get("content_type"),
                "manifest reference content_type",
            )

    def _read_verified_reference(
        self,
        reference: Mapping[str, Any],
        *,
        project_id: str,
        max_bytes: int,
        label: str,
    ) -> bytes:
        key = _required_text(reference.get("s3_key"), f"{label}.s3_key")
        self._validate_project_key(key, project_id=project_id)
        expected_content_type = _required_text(
            reference.get("content_type"),
            f"{label}.content_type",
        )
        body = self._read_object(
            key,
            max_bytes=max_bytes,
            expected_content_type=expected_content_type,
        )
        assert body is not None
        if len(body) != _required_nonnegative_int(
            reference.get("byte_size"),
            f"{label}.byte_size",
        ):
            raise PermanentGenerationError(
                code="brand_context_object_size_mismatch",
                safe_message="A brand context object size did not match its manifest.",
            )
        if hashlib.sha256(body).hexdigest() != _required_sha256(
            reference.get("sha256"),
            f"{label}.sha256",
        ):
            raise PermanentGenerationError(
                code="brand_context_object_checksum_mismatch",
                safe_message="A brand context object checksum did not match its manifest.",
            )
        return body

    def _read_object(
        self,
        key: str,
        *,
        max_bytes: int,
        optional: bool = False,
        expected_content_type: str | None = None,
    ) -> bytes | None:
        try:
            response = self._client().get_object(Bucket=self._bucket_name, Key=key)
        except Exception as exc:
            if optional and _is_s3_not_found(exc):
                return None
            if _is_s3_not_found(exc):
                raise PermanentGenerationError(
                    code="brand_context_object_missing",
                    safe_message="A required brand context object was not found.",
                ) from exc
            raise RetryableGenerationError(
                code="brand_context_read_failed",
                safe_message="Brand context storage could not be read temporarily.",
            ) from exc
        if not isinstance(response, Mapping):
            raise PermanentGenerationError(
                code="brand_context_object_invalid",
                safe_message="A brand context object response was invalid.",
            )
        if expected_content_type is not None and not _content_types_match(
            response.get("ContentType"),
            expected_content_type,
        ):
            raise PermanentGenerationError(
                code="brand_context_object_content_type_mismatch",
                safe_message=(
                    "A brand context object content type did not match its manifest."
                ),
            )
        body = response.get("Body")
        reader = getattr(body, "read", None)
        if not callable(reader):
            raise PermanentGenerationError(
                code="brand_context_object_invalid",
                safe_message="A brand context object response was invalid.",
            )
        try:
            data = reader(max_bytes + 1)
        except Exception as exc:
            raise RetryableGenerationError(
                code="brand_context_read_failed",
                safe_message="Brand context storage could not be read temporarily.",
            ) from exc
        finally:
            closer = getattr(body, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        if not isinstance(data, bytes):
            raise PermanentGenerationError(
                code="brand_context_object_invalid",
                safe_message="A brand context object response was invalid.",
            )
        if len(data) > max_bytes:
            raise PermanentGenerationError(
                code="brand_context_object_too_large",
                safe_message="A brand context object exceeded its size limit.",
            )
        return data

    def _client(self) -> Any:
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def _validate_project_key(
        self,
        key: str,
        *,
        project_id: str,
        expected_parent: str | None = None,
    ) -> None:
        project_prefix = f"{self._base_prefix}{project_id}/"
        if (
            not key.startswith(project_prefix)
            or ".." in key.split("/")
            or key.startswith("/")
        ):
            raise PermanentGenerationError(
                code="brand_context_key_invalid",
                safe_message="A brand context object key was outside its project prefix.",
            )
        if expected_parent is not None and not key.startswith(
            f"{project_prefix}{expected_parent}/"
        ):
            raise PermanentGenerationError(
                code="brand_context_key_invalid",
                safe_message="A brand context object key did not match its contract.",
            )


def _normalised_prefix(value: str) -> str:
    prefix = _required_text(value, "base_prefix").strip("/")
    if not prefix or ".." in prefix.split("/"):
        raise ValueError("brand context base_prefix is invalid")
    return f"{prefix}/"


def _content_types_match(actual: object, expected: str) -> bool:
    actual_media_type = str(actual or "").split(";", 1)[0].strip().casefold()
    expected_media_type = str(expected).split(";", 1)[0].strip().casefold()
    return bool(actual_media_type) and actual_media_type == expected_media_type


def _validated_project_id(value: str) -> str:
    project_id = str(value).strip()
    if not _PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError("brand context project_id is invalid")
    return project_id


def _json_object(value: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        decoded = value.decode("utf-8")
        parsed = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermanentGenerationError(
            code="brand_context_json_invalid",
            safe_message=f"The {label} was not valid JSON.",
        ) from exc
    if not isinstance(parsed, Mapping):
        raise PermanentGenerationError(
            code="brand_context_json_invalid",
            safe_message=f"The {label} was not a JSON object.",
        )
    return parsed


def _utf8_text(value: bytes, *, label: str) -> str:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PermanentGenerationError(
            code="brand_context_text_invalid",
            safe_message=f"The {label} was not valid UTF-8 text.",
        ) from exc
    if not text.strip():
        raise PermanentGenerationError(
            code="brand_context_text_invalid",
            safe_message=f"The {label} was empty.",
        )
    return text


def _require_schema(value: Mapping[str, Any], expected: str, *, label: str) -> None:
    _require_equal(value, "schema_version", expected, label=label)


def _require_equal(
    value: Mapping[str, Any],
    key: str,
    expected: str,
    *,
    label: str,
) -> None:
    if str(value.get(key) or "") != expected:
        raise PermanentGenerationError(
            code="brand_context_contract_mismatch",
            safe_message=f"The {label} did not match the expected contract.",
        )


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise PermanentGenerationError(
            code="brand_context_manifest_invalid",
            safe_message="The brand context manifest was invalid.",
        )
    return item


def _mapping_list(value: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    raw = value.get(key, [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise PermanentGenerationError(
            code="brand_context_manifest_invalid",
            safe_message="The brand context manifest was invalid.",
        )
    if not all(isinstance(item, Mapping) for item in raw):
        raise PermanentGenerationError(
            code="brand_context_manifest_invalid",
            safe_message="The brand context manifest was invalid.",
        )
    return list(raw)


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PermanentGenerationError(
            code="brand_context_contract_invalid",
            safe_message=f"The brand context {field_name} was missing.",
        )
    return text


def _required_sha256(value: object, field_name: str) -> str:
    digest = _required_text(value, field_name)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise PermanentGenerationError(
            code="brand_context_contract_invalid",
            safe_message=f"The brand context {field_name} was invalid.",
        )
    return digest


def _required_nonnegative_int(value: object, field_name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise PermanentGenerationError(
            code="brand_context_contract_invalid",
            safe_message=f"The brand context {field_name} was invalid.",
        ) from exc
    if integer < 0:
        raise PermanentGenerationError(
            code="brand_context_contract_invalid",
            safe_message=f"The brand context {field_name} was invalid.",
        )
    return integer


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _manifest_guide_version(
    manifest: Mapping[str, Any],
    fallback: str,
) -> str:
    versions = {
        str(item.get("version") or "").strip()
        for item in _mapping_list(manifest, "guidelines")
        if bool(item.get("required")) and str(item.get("version") or "").strip()
    }
    return next(iter(versions)) if len(versions) == 1 else fallback


def _manifest_catalog_version(
    manifest: Mapping[str, Any],
    fallback: str,
) -> str:
    versions = {
        str(item.get("version") or "").strip()
        for item in _mapping_list(manifest, "catalogs")
        if str(item.get("version") or "").strip()
    }
    return next(iter(versions)) if len(versions) == 1 else fallback


def _brand_kit_rules(value: Mapping[str, Any]) -> dict[str, Any]:
    colours = tuple(dict.fromkeys(_all_hex_colours(value)))
    brand = value.get("brand")
    brand_description = None
    if isinstance(brand, Mapping):
        parts = [
            str(brand.get(key) or "").strip()
            for key in ("name", "category", "locale")
        ]
        brand_description = ", ".join(part for part in parts if part)
    rules: dict[str, Any] = {}
    if colours:
        rules["approved_colors"] = colours
    if brand_description:
        rules["brand_description"] = brand_description
    return rules


def _all_hex_colours(value: object) -> list[str]:
    if isinstance(value, Mapping):
        colours: list[str] = []
        for item in value.values():
            colours.extend(_all_hex_colours(item))
        return colours
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        colours = []
        for item in value:
            colours.extend(_all_hex_colours(item))
        return colours
    return [match.casefold() for match in _HEX_COLOUR_PATTERN.findall(str(value))]


def _markdown_section_bullets(value: str, heading: str) -> tuple[str, ...]:
    in_section = False
    items: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if in_section and title != heading:
                break
            in_section = title == heading
            continue
        if not in_section:
            continue
        match = _MARKDOWN_BULLET_PATTERN.fullmatch(line)
        if match is None:
            continue
        item = match.group(1).strip().strip('"\'`“”')
        if item:
            items.append(item)
    return tuple(dict.fromkeys(items))


def _select_representative_asset(
    assets: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    eligible = [item for item in assets if _asset_is_eligible(item)]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            0 if str(item.get("role") or "") == "hero" else 1,
            0 if str(item.get("asset_id") or "") == "home-hero" else 1,
            str(item.get("asset_id") or ""),
        ),
    )


def _asset_is_eligible(value: Mapping[str, Any]) -> bool:
    if value.get("active") is not True:
        return False
    advertising_use = str(value.get("advertising_use") or "").casefold()
    if advertising_use.startswith(("blocked", "pending")):
        return False
    return bool(str(value.get("asset_id") or "").strip())


def _asset_document_text(value: Mapping[str, Any]) -> str:
    parts = [
        str(value.get("alt_text") or "").strip(),
        "role=" + str(value.get("role") or "").strip(),
    ]
    tags = _string_list(value.get("tags"))
    if tags:
        parts.append("tags=" + ", ".join(tags))
    return ". ".join(part for part in parts if part and not part.endswith("="))


def _document_id(
    project_id: str,
    source_kind: str,
    source_id: str,
    version: str,
) -> str:
    raw = f"brandctx_{project_id}_{source_kind}_{source_id}_{version}"
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_")
    if len(slug) <= 100:
        return slug
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{slug[:83]}_{digest}"


def _is_s3_not_found(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return False
    metadata = response.get("ResponseMetadata")
    status_code = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    details = response.get("Error")
    code = details.get("Code") if isinstance(details, Mapping) else None
    return status_code == 404 or str(code or "") in {"404", "NoSuchKey", "NotFound"}
