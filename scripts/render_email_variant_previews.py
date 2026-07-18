#!/usr/bin/env python3
"""Render and validate three email variants from a local V2 handoff bundle."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.generation.artifacts import (  # noqa: E402
    ArtifactIdentity,
    CreativeArtifactPublication,
    StaticCreativeArtifactPublisher,
    content_values_from_rendered_html,
    render_creative_html,
)
from app.generation.brand_context import (  # noqa: E402
    BRAND_CONTEXT_EMBEDDING_DIMENSIONS,
    MAX_RETRIEVAL_DOCUMENTS,
    BrandContextSnapshot,
    BrandContextRetrievalService,
    RetrievedBrandDocument,
)
from app.generation.brand_context_s3 import S3BrandContextLoader  # noqa: E402
from app.generation.errors import GenerationError  # noqa: E402
from app.generation.generator import (  # noqa: E402
    DeterministicContentGenerator,
    GeneratedContent,
)
from app.generation.prompt_builder import (  # noqa: E402
    GenerationInputBuilder,
    GenerationPromptInput,
    PromotionOfferLink,
    PromotionPromptInput,
    PromptBuildResult,
    TargetSegmentPromptInput,
)
from app.generation.schemas import (  # noqa: E402
    ContentChannel,
    GenerationRequest,
)
from app.generation.service import GenerationService  # noqa: E402


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/email-variant-previews"
DEFAULT_LANDING_URL = (
    "https://demo-shoppingmall.dev.loop-ad.org/promotions/black-friday"
)
DEFAULT_CATALOG_ID = "black-friday-hotels"
EXPECTED_VARIANTS = ("editorial", "offer_cards", "comparison")
EXPECTED_LOCAL_IMAGE_COUNTS = {
    "editorial": 3,
    "offer_cards": 8,
    "comparison": 4,
}
OUTPUT_NAMES = {
    "editorial": "01-editorial",
    "offer_cards": "02-eight-offers",
    "comparison": "03-comparison",
}
BRAND_BLUE = "#0F55C8"


class PreviewValidationError(ValueError):
    """Raised when the local bundle or rendered preview is inconsistent."""


class LocalObjectNotFound(RuntimeError):
    response = {
        "ResponseMetadata": {"HTTPStatusCode": 404},
        "Error": {"Code": "NoSuchKey"},
    }


class LocalBundleS3Client:
    """Expose a handoff directory through the S3 get_object shape."""

    def __init__(self, bundle_root: Path) -> None:
        self.bundle_root = bundle_root.expanduser().resolve()

    def get_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs.get("Key") or "")
        path = (self.bundle_root / key).resolve()
        if not path.is_relative_to(self.bundle_root) or not path.is_file():
            raise LocalObjectNotFound(key)
        body = path.read_bytes()
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ContentType": _content_type_for_path(path),
        }


class ZeroEmbeddingClient:
    def embed(self, text: str) -> Sequence[float]:
        if not text.strip():
            raise PreviewValidationError("brand retrieval query must not be empty")
        return (0.0,) * BRAND_CONTEXT_EMBEDDING_DIMENSIONS


class EmptyBrandDocumentReader:
    def retrieve(
        self,
        *,
        project_id: str,
        context_version: str,
        channel: ContentChannel,
        query_embedding: Sequence[float],
        limit: int = MAX_RETRIEVAL_DOCUMENTS,
    ) -> list[RetrievedBrandDocument]:
        del project_id, context_version, channel, limit
        if len(query_embedding) != BRAND_CONTEXT_EMBEDDING_DIMENSIONS:
            raise PreviewValidationError("brand retrieval embedding is invalid")
        return []


class CatalogPreviewContentGenerator:
    """Use deterministic copy and a verified catalog image without provider calls."""

    version = "local-preview.catalog-deterministic.v1"

    def __init__(self) -> None:
        self._delegate = DeterministicContentGenerator()

    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
        artifact_identity: ArtifactIdentity,
    ) -> GeneratedContent:
        content = self._delegate.generate(
            prompt_input=prompt_input,
            prompt_result=prompt_result,
            option_index=option_index,
            artifact_identity=artifact_identity,
        )
        hotels = _mapping_list(
            (prompt_input.offer_catalog or {}).get("hotels"),
            label="offer catalog hotels",
        )
        first_image_path = _required_text(
            hotels[0].get("image_path"),
            "offer catalog image_path",
        )
        landing_url = _required_text(
            prompt_input.promotion.landing_url,
            "promotion landing_url",
        )
        return replace(
            content,
            image_url=f"{_validated_https_origin(landing_url)}{first_image_path}",
        )


class CapturingCreativeArtifactPublisher:
    """Capture the exact renderer input while delegating artifact metadata."""

    def __init__(self) -> None:
        self._delegate = StaticCreativeArtifactPublisher()
        self._lock = threading.Lock()
        self._html_by_content_id: dict[str, str] = {}

    def publish(
        self,
        *,
        identity: ArtifactIdentity,
        channel: ContentChannel,
        content_values: Mapping[str, Any],
    ) -> CreativeArtifactPublication:
        html_body = render_creative_html(
            channel=channel,
            content_values=content_values,
        )
        publication = self._delegate.publish(
            identity=identity,
            channel=channel,
            content_values=content_values,
        )
        with self._lock:
            if identity.content_id in self._html_by_content_id:
                raise PreviewValidationError(
                    f"duplicate HTML capture for {identity.content_id}"
                )
            self._html_by_content_id[identity.content_id] = html_body
        return publication

    def html_for(self, content_id: str) -> str:
        with self._lock:
            html_body = self._html_by_content_id.get(content_id)
        if html_body is None:
            raise PreviewValidationError(f"HTML was not captured for {content_id}")
        return html_body


@dataclass(frozen=True)
class LoadedPreviewBundle:
    bundle_root: Path
    manifest: Mapping[str, Any]
    offer_catalog: Mapping[str, Any]
    source_loader: S3BrandContextLoader
    snapshot: BrandContextSnapshot


def main() -> int:
    args = parse_args()
    try:
        bundle = load_preview_bundle(
            bundle_root=args.bundle_root,
            project_id=args.project_id,
            catalog_id=args.catalog_id,
        )
        report = render_local_previews(
            bundle=bundle,
            project_id=args.project_id,
            landing_url=args.landing_url,
            output_dir=args.output_dir,
        )
    except (GenerationError, OSError, PreviewValidationError, ValueError) as exc:
        print(f"email preview validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render editorial, eight-offer, and comparison email previews "
            "with a local StayLoop V2 brand-context bundle."
        )
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help=(
            "Handoff root containing brand-context/<project-id>/current.json"
        ),
    )
    parser.add_argument("--project-id", default="demo_project")
    parser.add_argument("--catalog-id", default=DEFAULT_CATALOG_ID)
    parser.add_argument("--landing-url", default=DEFAULT_LANDING_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def load_preview_bundle(
    *,
    bundle_root: Path,
    project_id: str,
    catalog_id: str,
) -> LoadedPreviewBundle:
    resolved_root = bundle_root.expanduser().resolve()
    source_loader = S3BrandContextLoader(
        bucket_name="local-brand-context-bundle",
        base_prefix="brand-context/",
        s3_client=LocalBundleS3Client(resolved_root),
    )
    snapshot = source_loader.resolve_snapshot(project_id=project_id)
    if snapshot is None:
        raise PreviewValidationError(
            f"brand context pointer is missing for {project_id}"
        )
    offer_catalog = source_loader.load_offer_catalog(
        project_id=project_id,
        snapshot=snapshot,
    )
    if not isinstance(offer_catalog, Mapping):
        raise PreviewValidationError("promotion price catalog is missing")
    if str(offer_catalog.get("catalog_id") or "") != catalog_id:
        raise PreviewValidationError(
            f"expected catalog {catalog_id}, found {offer_catalog.get('catalog_id')}"
        )
    manifest_path = _bundle_object_path(resolved_root, snapshot.manifest_key)
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != snapshot.manifest_sha256:
        raise PreviewValidationError("brand context manifest checksum does not match")
    manifest = _json_mapping(manifest_bytes, label="brand context manifest")
    return LoadedPreviewBundle(
        bundle_root=resolved_root,
        manifest=manifest,
        offer_catalog=offer_catalog,
        source_loader=source_loader,
        snapshot=snapshot,
    )


def render_local_previews(
    *,
    bundle: LoadedPreviewBundle,
    project_id: str,
    landing_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    public_origin = _validated_https_origin(landing_url)
    hotels = _mapping_list(
        bundle.offer_catalog.get("hotels"),
        label="offer catalog hotels",
    )
    if len(hotels) != 8:
        raise PreviewValidationError(
            "three-variant preview requires exactly eight catalog hotels"
        )
    offer_links = tuple(
        PromotionOfferLink(
            offer_id=_required_text(hotel.get("offer_id"), "offer_id"),
            destination_url=(
                f"{public_origin}/hotel/"
                + _required_text(hotel.get("offer_id"), "offer_id")
            ),
        )
        for hotel in hotels
    )
    prompt_inputs = GenerationInputBuilder().build(
        request=_preview_request(project_id),
        promotion=_preview_promotion(
            project_id=project_id,
            landing_url=landing_url,
            offer_links=offer_links,
        ),
        target_segments=[_preview_target_segment()],
        brand_context=bundle.snapshot,
        offer_catalog=bundle.offer_catalog,
    )
    brand_context_provider = BrandContextRetrievalService(
        repository=EmptyBrandDocumentReader(),
        embedding_client=ZeroEmbeddingClient(),
        source_loader=bundle.source_loader,
    )
    publisher = CapturingCreativeArtifactPublisher()
    started = time.perf_counter()
    result = GenerationService(
        brand_context_provider=brand_context_provider,
        content_generator=CatalogPreviewContentGenerator(),
        artifact_publisher=publisher,
    ).execute_durable(
        generation_id="generation_local_three_email_variants",
        prompt_inputs=prompt_inputs,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if result.generation_report_json.get("status") != "completed":
        raise PreviewValidationError("local generation did not complete")
    if len(result.content_candidates) != 3:
        raise PreviewValidationError("local generation did not return three candidates")

    resolved_output_dir = output_dir.expanduser().resolve()
    assets_output_dir = resolved_output_dir / "assets"
    assets_output_dir.mkdir(parents=True, exist_ok=True)
    replacements = _prepare_local_image_replacements(
        bundle=bundle,
        hotels=hotels,
        public_origin=public_origin,
        assets_output_dir=assets_output_dir,
    )

    summaries: list[dict[str, Any]] = []
    rendered_variants: list[str] = []
    for candidate in result.content_candidates:
        creative = candidate.metadata_json.get("creative")
        if not isinstance(creative, Mapping):
            raise PreviewValidationError("candidate creative metadata is missing")
        variant = _required_text(creative.get("variant_type"), "variant_type")
        rendered_variants.append(variant)
        canonical_html = publisher.html_for(candidate.content_id)
        canonical_sha256 = hashlib.sha256(canonical_html.encode("utf-8")).hexdigest()
        if canonical_sha256 != candidate.artifact_sha256:
            raise PreviewValidationError(
                f"{variant} captured HTML does not match artifact SHA-256"
            )
        content_values_from_rendered_html(
            channel=ContentChannel.EMAIL,
            html_body=canonical_html,
        )
        local_html = _localize_preview_images(
            html_body=canonical_html,
            replacements=replacements,
        )
        verification = _verify_local_preview_html(
            html_body=local_html,
            variant=variant,
            creative=creative,
            output_dir=resolved_output_dir,
        )
        output_name = OUTPUT_NAMES.get(variant)
        if output_name is None:
            raise PreviewValidationError(f"unsupported preview variant {variant}")
        canonical_path = resolved_output_dir / f"{output_name}.production.html"
        local_path = resolved_output_dir / f"{output_name}.local.html"
        canonical_path.write_text(canonical_html, encoding="utf-8")
        local_path.write_text(local_html, encoding="utf-8")
        renderer = creative.get("renderer")
        summaries.append(
            {
                "variant": variant,
                "template_version": (
                    renderer.get("template_version")
                    if isinstance(renderer, Mapping)
                    else None
                ),
                "artifact_status": candidate.artifact_status,
                "image_generation_status": candidate.image_generation_status,
                "canonical_sha256_verified": canonical_sha256,
                "canonical_output": str(canonical_path),
                "local_preview_sha256": hashlib.sha256(
                    local_html.encode("utf-8")
                ).hexdigest(),
                "local_output": str(local_path),
                **verification,
            }
        )

    if tuple(rendered_variants) != EXPECTED_VARIANTS:
        raise PreviewValidationError(
            "candidate order must be editorial, offer_cards, comparison"
        )
    index_path = _write_preview_index(
        output_dir=resolved_output_dir,
        summaries=summaries,
    )
    report = {
        "generation_status": "completed",
        "elapsed_ms_without_external_providers": elapsed_ms,
        "candidate_count": len(result.content_candidates),
        "catalog_hotel_count": len(hotels),
        "external_provider_calls": 0,
        "brand_context": {
            "context_version": bundle.snapshot.context_version,
            "manifest_sha256": bundle.snapshot.manifest_sha256,
            "source_documents_loaded": True,
        },
        "index": str(index_path),
        "candidates": summaries,
    }
    (resolved_output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _preview_request(project_id: str) -> GenerationRequest:
    return GenerationRequest(
        project_id=project_id,
        campaign_id="camp_summer_2026",
        promotion_id="promo_jeju_okinawa_summer",
        analysis_id="analysis_summer_travel",
        content_option_count=3,
        operator_instruction=(
            "20~30대 여름 휴가 고객에게 제주와 오키나와 숙소를 소개한다."
        ),
    )


def _preview_promotion(
    *,
    project_id: str,
    landing_url: str,
    offer_links: tuple[PromotionOfferLink, ...],
) -> PromotionPromptInput:
    return PromotionPromptInput(
        project_id=project_id,
        campaign_id="camp_summer_2026",
        promotion_id="promo_jeju_okinawa_summer",
        channel=ContentChannel.EMAIL,
        goal_metric="booking_conversion_rate",
        goal_target_value="0.030000",
        goal_basis="all_segments",
        message_brief=(
            "여름 휴가를 준비하는 20~30대 사용자의 제주·오키나와 "
            "숙소 예약을 유도하고 인기 여행지와 조기 예약 할인을 강조한다."
        ),
        landing_url=landing_url,
        offer_links=offer_links,
    )


def _preview_target_segment() -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id="analysis_summer_travel",
        promotion_id="promo_jeju_okinawa_summer",
        segment_id="seg_summer_travel_20_30",
        segment_name="여름 휴가를 준비하는 20~30대 여행 관심 고객",
        content_slug="summer_travel_20_30",
        content_brief_json={
            "message_direction": "제주·오키나와 숙소 비교와 예약 유도",
            "keywords": ["여름 휴가", "조기 예약 할인", "숙소 비교"],
        },
        segment_vector_id="segvec_summer_travel_20_30_v1",
        estimated_size=1200,
        priority="high",
        natural_language_query="20~30대 제주 오키나와 여름 숙소 관심 고객",
        generated_sql=None,
        sample_ratio="0.018000",
        source="system_default",
        query_preview_id=None,
    )


def _prepare_local_image_replacements(
    *,
    bundle: LoadedPreviewBundle,
    hotels: Sequence[Mapping[str, Any]],
    public_origin: str,
    assets_output_dir: Path,
) -> dict[str, str]:
    assets_by_id = {
        _required_text(asset.get("asset_id"), "asset_id"): asset
        for asset in _mapping_list(
            bundle.manifest.get("assets"),
            label="brand context assets",
        )
    }
    replacements: dict[str, str] = {}
    for hotel in hotels:
        asset_id = _required_text(hotel.get("asset_id"), "hotel asset_id")
        asset = assets_by_id.get(asset_id)
        if asset is None:
            raise PreviewValidationError(
                f"brand context manifest does not contain asset {asset_id}"
            )
        source_path = _verified_asset_path(bundle.bundle_root, asset)
        filename = f"{_safe_filename(asset_id)}{source_path.suffix.lower()}"
        target_path = assets_output_dir / filename
        shutil.copyfile(source_path, target_path)
        image_path = _required_text(hotel.get("image_path"), "hotel image_path")
        replacements[f"{public_origin}{image_path}"] = f"assets/{filename}"
    return replacements


def _localize_preview_images(
    *,
    html_body: str,
    replacements: Mapping[str, str],
) -> str:
    localized = html_body
    for public_url, local_url in replacements.items():
        localized = localized.replace(
            html.escape(public_url, quote=True),
            html.escape(local_url, quote=True),
        )
    return localized


def _verify_local_preview_html(
    *,
    html_body: str,
    variant: str,
    creative: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    expected_image_count = EXPECTED_LOCAL_IMAGE_COUNTS.get(variant)
    if expected_image_count is None:
        raise PreviewValidationError(f"unsupported preview variant {variant}")
    local_image_urls = [
        value
        for value in re.findall(r'src="([^"]+)"', html_body)
        if value.startswith("assets/")
    ]
    if len(local_image_urls) != expected_image_count:
        raise PreviewValidationError(
            f"{variant} preview expected {expected_image_count} local images, "
            f"found {len(local_image_urls)}"
        )
    for image_url in local_image_urls:
        if not (output_dir / image_url).is_file():
            raise PreviewValidationError(
                f"{variant} preview image does not exist: {image_url}"
            )
    source = creative.get("source")
    required_placeholders = (
        source.get("required_placeholders")
        if isinstance(source, Mapping)
        else None
    )
    if not isinstance(required_placeholders, list) or not all(
        isinstance(value, str) and value in html_body
        for value in required_placeholders
    ):
        raise PreviewValidationError(
            f"{variant} preview is missing required placeholders"
        )
    if f"background:{BRAND_BLUE}" not in html_body:
        raise PreviewValidationError(
            f"{variant} preview is missing the brand-blue CTA"
        )
    if variant == "offer_cards" and html_body.count("숙소 확인하기") != 8:
        raise PreviewValidationError(
            "offer_cards preview must contain eight accommodation CTAs"
        )
    if variant == "comparison" and html_body.count("PICK 0") != 4:
        raise PreviewValidationError(
            "comparison preview must contain four comparison rows"
        )
    if variant == "editorial" and html_body.count("추천:</strong>") != 2:
        raise PreviewValidationError(
            "editorial preview must contain two destination sections"
        )
    return {
        "local_images_verified": len(local_image_urls),
        "required_placeholders_verified": list(required_placeholders),
        "brand_blue_cta_verified": True,
    }


def _write_preview_index(
    *,
    output_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> Path:
    labels = {
        "editorial": "1. 설명형",
        "offer_cards": "2. 8숙소형",
        "comparison": "3. 비교형",
    }
    links = []
    for summary in summaries:
        variant = str(summary["variant"])
        filename = Path(str(summary["local_output"])).name
        links.append(
            "<li>"
            f'<a href="{html.escape(filename, quote=True)}">'
            f"{html.escape(labels[variant])}</a>"
            f" <span>{int(summary['local_images_verified'])}개 이미지 검증</span>"
            "</li>"
        )
    index_body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            "  <title>LoopAd 이메일 3종 로컬 미리보기</title>",
            "  <style>",
            "    body{margin:0;padding:48px 24px;background:#eef3fb;font-family:Arial,sans-serif;color:#10233f}",
            "    main{max-width:720px;margin:0 auto;padding:34px;background:#fff;border-radius:18px;box-shadow:0 12px 40px rgba(16,35,63,.12)}",
            "    h1{margin:0 0 10px;font-size:28px}p{margin:0 0 24px;color:#53657d;line-height:1.6}",
            "    ul{list-style:none;margin:0;padding:0;display:grid;gap:12px}",
            f"    a{{display:block;padding:16px 18px;border-radius:10px;background:{BRAND_BLUE};color:#fff;font-weight:700;text-decoration:none}}",
            "    span{display:block;padding:7px 4px 0;color:#718096;font-size:12px}",
            "  </style>",
            "</head>",
            "<body>",
            "  <main>",
            "    <h1>LoopAd 이메일 3종 로컬 미리보기</h1>",
            "    <p>V2 bundle과 production renderer를 사용했습니다. OpenAI·Gemini·DB·S3 호출은 없습니다.</p>",
            "    <ul>",
            *[f"      {link}" for link in links],
            "    </ul>",
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    index_path = output_dir / "index.html"
    index_path.write_text(index_body, encoding="utf-8")
    return index_path


def _verified_asset_path(
    bundle_root: Path,
    reference: Mapping[str, Any],
) -> Path:
    path = _bundle_object_path(
        bundle_root,
        _required_text(reference.get("s3_key"), "asset s3_key"),
    )
    body = path.read_bytes()
    expected_size = _required_nonnegative_int(
        reference.get("byte_size"),
        "asset byte_size",
    )
    if len(body) != expected_size:
        raise PreviewValidationError(f"asset byte_size does not match: {path}")
    expected_sha256 = _required_text(reference.get("sha256"), "asset sha256")
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise PreviewValidationError(f"asset checksum does not match: {path}")
    return path


def _bundle_object_path(bundle_root: Path, key: str) -> Path:
    path = (bundle_root / key).resolve()
    if not path.is_relative_to(bundle_root) or not path.is_file():
        raise PreviewValidationError(f"bundle object is missing: {key}")
    return path


def _json_mapping(value: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreviewValidationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping):
        raise PreviewValidationError(f"{label} must be a JSON object")
    return parsed


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.casefold()
    values = {
        ".json": "application/json; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }
    content_type = values.get(suffix)
    if content_type is None:
        raise PreviewValidationError(f"unsupported bundle content type: {path}")
    return content_type


def _validated_https_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise PreviewValidationError("landing_url must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        raise PreviewValidationError("landing_url must not contain credentials")
    return f"{parsed.scheme}://{parsed.netloc}"


def _mapping_list(value: object, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise PreviewValidationError(f"{label} must be an object array")
    return list(value)


def _required_text(value: object, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise PreviewValidationError(f"{label} is required")
    return normalized


def _required_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise PreviewValidationError(f"{label} must be a non-negative integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise PreviewValidationError(
            f"{label} must be a non-negative integer"
        ) from exc
    if normalized < 0:
        raise PreviewValidationError(f"{label} must be a non-negative integer")
    return normalized


def _safe_filename(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not normalized:
        raise PreviewValidationError("asset filename is invalid")
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
