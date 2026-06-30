from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.contents.types import GeneratedContentDraft


DEFAULT_ASSET_PREFIX = "generated-contents"
DEFAULT_S3_CACHE_CONTROL = "public, max-age=31536000, immutable"
SVG_CONTENT_TYPE = "image/svg+xml"


@dataclass(frozen=True)
class AssetObject:
    key: str
    body: bytes
    content_type: str


@dataclass(frozen=True)
class StoredAsset:
    key: str
    public_url: str | None = None
    content_type: str = SVG_CONTENT_TYPE


class AssetStorage(Protocol):
    def put_object(self, asset: AssetObject) -> StoredAsset:
        ...


class S3ClientLike(Protocol):
    def put_object(self, **kwargs: Any) -> Any:
        ...


class InMemoryAssetStorage:
    def __init__(self, public_base_url: str | None = "https://assets.example.test") -> None:
        self.public_base_url = _normalize_base_url(public_base_url)
        self.objects: dict[str, AssetObject] = {}

    def put_object(self, asset: AssetObject) -> StoredAsset:
        _validate_asset_key(asset.key)
        self.objects[asset.key] = asset
        return StoredAsset(
            key=asset.key,
            public_url=_join_public_url(self.public_base_url, asset.key),
            content_type=asset.content_type,
        )


class LocalAssetStorage:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        public_base_url: str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.public_base_url = _normalize_base_url(public_base_url)

    def put_object(self, asset: AssetObject) -> StoredAsset:
        _validate_asset_key(asset.key)
        root = self.root_dir.resolve()
        destination = (root / asset.key).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("asset key must stay inside the local asset root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(asset.body)
        return StoredAsset(
            key=asset.key,
            public_url=_join_public_url(self.public_base_url, asset.key),
            content_type=asset.content_type,
        )


class S3AssetStorage:
    def __init__(
        self,
        *,
        bucket: str,
        public_base_url: str,
        client: S3ClientLike | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        cache_control: str | None = DEFAULT_S3_CACHE_CONTROL,
        public_url_strip_prefix: str | None = None,
    ) -> None:
        normalized_bucket = bucket.strip()
        if not normalized_bucket:
            raise ValueError("S3 bucket must not be empty")
        normalized_public_base_url = _normalize_base_url(public_base_url)
        if normalized_public_base_url is None:
            raise ValueError("S3 public base URL must not be empty")
        self.bucket = normalized_bucket
        self.public_base_url = normalized_public_base_url
        self._client = client
        self.region_name = _clean(region_name)
        self.endpoint_url = _clean(endpoint_url)
        self.cache_control = _clean(cache_control)
        self.public_url_strip_prefix = _normalize_key_prefix(public_url_strip_prefix)

    def put_object(self, asset: AssetObject) -> StoredAsset:
        _validate_asset_key(asset.key)
        params: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": asset.key,
            "Body": asset.body,
            "ContentType": asset.content_type,
        }
        if self.cache_control is not None:
            params["CacheControl"] = self.cache_control
        self._get_client().put_object(**params)
        return StoredAsset(
            key=asset.key,
            public_url=_join_public_url(
                self.public_base_url,
                _strip_key_prefix(asset.key, self.public_url_strip_prefix),
            ),
            content_type=asset.content_type,
        )

    def _get_client(self) -> S3ClientLike:
        if self._client is None:
            self._client = _build_boto3_s3_client(
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            )
        return self._client


class SvgBannerRenderer:
    def render(self, draft: GeneratedContentDraft) -> bytes:
        title = html.escape(draft.title)
        body = html.escape(draft.body)
        cta = html.escape(draft.cta_label)
        variant = html.escape(draft.variant_key)
        content_type = html.escape(draft.content_type)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="628" viewBox="0 0 1200 628" role="img">
  <rect width="1200" height="628" fill="#f8fafc"/>
  <rect x="48" y="48" width="1104" height="532" rx="24" fill="#ffffff" stroke="#d9e2ec" stroke-width="4"/>
  <text x="88" y="150" font-family="Arial, sans-serif" font-size="34" fill="#64748b">{variant} / {content_type}</text>
  <text x="88" y="270" font-family="Arial, sans-serif" font-size="64" font-weight="700" fill="#111827">{title}</text>
  <text x="88" y="360" font-family="Arial, sans-serif" font-size="34" fill="#374151">{body}</text>
  <rect x="88" y="430" width="320" height="84" rx="42" fill="#2563eb"/>
  <text x="132" y="484" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#ffffff">{cta}</text>
</svg>
"""
        return svg.encode("utf-8")


class ContentAssetService:
    def __init__(
        self,
        *,
        storage: AssetStorage,
        renderer: SvgBannerRenderer | None = None,
        asset_prefix: str = DEFAULT_ASSET_PREFIX,
    ) -> None:
        self.storage = storage
        self.renderer = renderer or SvgBannerRenderer()
        self.asset_prefix = asset_prefix.strip("/") or DEFAULT_ASSET_PREFIX

    def store_banner(self, draft: GeneratedContentDraft) -> GeneratedContentDraft:
        asset_key = build_asset_key(
            project_id=draft.project_id,
            recommendation_action_id=draft.recommendation_action_id,
            variant_key=draft.variant_key,
            prefix=self.asset_prefix,
            extension="svg",
        )
        stored = self.storage.put_object(
            AssetObject(
                key=asset_key,
                body=self.renderer.render(draft),
                content_type=SVG_CONTENT_TYPE,
            )
        )
        metadata = {
            **draft.metadata,
            "asset_key": stored.key,
            "asset_content_type": stored.content_type,
            "asset_storage": type(self.storage).__name__,
        }
        return GeneratedContentDraft(
            project_id=draft.project_id,
            recommendation_action_id=draft.recommendation_action_id,
            segment_id=draft.segment_id,
            variant_key=draft.variant_key,
            content_type=draft.content_type,
            title=draft.title,
            body=draft.body,
            cta_label=draft.cta_label,
            landing_url=draft.landing_url,
            image_prompt=draft.image_prompt,
            generation_model=draft.generation_model,
            generation_status=draft.generation_status,
            created_run_id=draft.created_run_id,
            image_url=stored.public_url,
            media_s3_key=stored.key,
            metadata=metadata,
        )


def build_asset_key(
    *,
    project_id: int | str,
    recommendation_action_id: int,
    variant_key: str,
    prefix: str = DEFAULT_ASSET_PREFIX,
    extension: str = "svg",
) -> str:
    normalized_prefix = prefix.strip("/") or DEFAULT_ASSET_PREFIX
    normalized_project = _safe_path_part(str(project_id))
    normalized_variant = _safe_path_part(variant_key)
    normalized_extension = _safe_path_part(extension).lstrip(".") or "svg"
    return (
        f"{normalized_prefix}/projects/{normalized_project}/actions/"
        f"{recommendation_action_id}/variants/{normalized_variant}/banner.{normalized_extension}"
    )


def _validate_asset_key(key: str) -> None:
    if not key.strip():
        raise ValueError("asset key must not be empty")
    path = Path(key)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("asset key must be a relative path without traversal")


def _normalize_key_prefix(value: str | None) -> str | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    return cleaned.strip("/") or None


def _strip_key_prefix(key: str, prefix: str | None) -> str:
    if prefix is None:
        return key
    normalized_key = key.strip("/")
    if normalized_key == prefix:
        return ""
    prefix_with_slash = f"{prefix}/"
    if normalized_key.startswith(prefix_with_slash):
        return normalized_key[len(prefix_with_slash) :]
    return key


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.strip())
    return safe.strip("-") or "unknown"


def _normalize_base_url(value: str | None) -> str | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    return cleaned.rstrip("/")


def _join_public_url(base_url: str | None, key: str) -> str | None:
    if base_url is None:
        return None
    return f"{base_url}/{key}"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _build_boto3_s3_client(
    *,
    region_name: str | None,
    endpoint_url: str | None,
) -> S3ClientLike:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 package is required for S3AssetStorage") from exc

    kwargs: dict[str, str] = {}
    if region_name is not None:
        kwargs["region_name"] = region_name
    if endpoint_url is not None:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)
