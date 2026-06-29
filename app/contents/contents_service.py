import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import status

from app.contents.ai.image_provider import GeneratedImage, ImageProvider
from app.contents.ai.mock_image_provider import MockImageProvider
from app.contents.assets.asset_resolver_service import AssetResolverService
from app.contents.compose.png_exporter_service import BannerRenderScene, PngExporterService
from app.contents.compose.svg_composer_service import SvgComposerService
from app.contents.copy_generation_service import CopyGenerationService
from app.contents.creative_brief_service import CreativeBriefService
from app.contents.errors import ContentGenerationError, not_found
from app.contents.schemas import GenerateContentRequest, GenerateContentResponse
from app.contents.storage.s3_asset_storage_service import AssetStorage, StorageObject
from app.persistence.repository import PostgresRepository


class ContentsService:
    def __init__(
        self,
        *,
        repository: PostgresRepository,
        image_provider: ImageProvider,
        storage: AssetStorage,
        asset_key_prefix: str = "",
        fallback_image_provider: ImageProvider | None = None,
        brief_service: CreativeBriefService | None = None,
        copy_service: CopyGenerationService | None = None,
        asset_resolver: AssetResolverService | None = None,
        svg_composer: SvgComposerService | None = None,
        png_exporter: PngExporterService | None = None,
    ) -> None:
        self.repository = repository
        self.image_provider = image_provider
        self.fallback_image_provider = fallback_image_provider or MockImageProvider()
        self.storage = storage
        self.asset_key_prefix = asset_key_prefix.strip("/")
        self.brief_service = brief_service or CreativeBriefService()
        self.copy_service = copy_service or CopyGenerationService()
        self.asset_resolver = asset_resolver or AssetResolverService()
        self.svg_composer = svg_composer or SvgComposerService()
        self.png_exporter = png_exporter or PngExporterService()

    def generate_content(self, request: GenerateContentRequest) -> GenerateContentResponse:
        try:
            return self._generate_content(request)
        except ContentGenerationError:
            self.repository.rollback()
            raise
        except Exception:
            self.repository.rollback()
            raise

    def _generate_content(self, request: GenerateContentRequest) -> GenerateContentResponse:
        recommendation_action = self.repository.get_recommendation_action_by_result_action(
            recommendation_result_id=request.recommendation_result_id,
            action_id=request.action_id,
        )
        if recommendation_action is None:
            raise not_found(
                "RECOMMENDATION_ACTION_NOT_FOUND",
                (
                    "No recommendation_actions row found for "
                    f"recommendation_result_id={request.recommendation_result_id} "
                    f"and action_id={request.action_id}"
                ),
            )

        action_catalog_item = self.repository.get_active_action_catalog_item(request.action_id)
        if action_catalog_item is None:
            raise not_found(
                "ACTION_CATALOG_NOT_FOUND",
                f"No active action_catalog row found for action_id={request.action_id}",
            )

        target = self.repository.get_content_generation_target(
            recommendation_action_id=recommendation_action.id,
            project_id=recommendation_action.project_id,
        )
        if target is None:
            raise not_found(
                "SEGMENT_AD_MAPPING_NOT_FOUND",
                (
                    "No active segment_ad_mappings row with an active project-matched "
                    f"creative found for recommendation_action_id={recommendation_action.id}"
                ),
            )
        mapping = target.mapping
        ad_creative = target.creative

        existing_url = (getattr(ad_creative, "image_url", None) or "").strip()
        if existing_url and not request.force:
            return GenerateContentResponse(
                creative_id=str(ad_creative.id),
                action_id=request.action_id,
                content_url=existing_url,
                recommendation_action_id=recommendation_action.id,
                mapping_id=mapping.id,
            )

        generation_id = self._new_generation_id()
        creative_brief = self.brief_service.create_brief(
            recommendation_action=recommendation_action,
            action_catalog_item=action_catalog_item,
            ad_creative=ad_creative,
            recommendation_result_id=request.recommendation_result_id,
            generation_id=generation_id,
        )
        copy = self.copy_service.generate_copy(creative_brief)
        assets = self.asset_resolver.resolve_assets()
        background = self._generate_background(creative_brief)
        composition_svg = self._compose_svg(
            background_png=background.body,
            copy=copy,
            assets=assets,
        )
        final_banner_png = self._export_png(
            composition_svg=composition_svg,
            copy=copy,
        )

        keys = self._build_asset_keys(
            recommendation_result_id=request.recommendation_result_id,
            action_id=request.action_id,
            creative_id=str(ad_creative.id),
            generation_id=generation_id,
        )
        urls = {name: self.storage.public_url_for_key(key) for name, key in keys.items()}
        manifest = self._build_manifest(
            generation_id=generation_id,
            recommendation_result_id=request.recommendation_result_id,
            action_id=request.action_id,
            creative_id=str(ad_creative.id),
            urls=urls,
            copy=copy,
            background=background,
        )
        self._upload_assets(
            keys=keys,
            creative_brief=creative_brief,
            copy=copy,
            background=background,
            composition_svg=composition_svg,
            manifest=manifest,
            final_banner_png=final_banner_png,
        )

        content_url = urls["final_banner"]
        self._update_creative_url(ad_creative.id, content_url)
        return GenerateContentResponse(
            creative_id=str(ad_creative.id),
            action_id=request.action_id,
            content_url=content_url,
            recommendation_action_id=recommendation_action.id,
            mapping_id=mapping.id,
        )

    def _generate_background(self, creative_brief: dict[str, object]) -> GeneratedImage:
        try:
            return self.image_provider.generate_background(creative_brief)
        except Exception as primary_exc:
            if self.image_provider.provider_name == self.fallback_image_provider.provider_name:
                raise ContentGenerationError(
                    code="IMAGE_GENERATION_FAILED",
                    message="Image generation failed",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                ) from primary_exc

            try:
                return self.fallback_image_provider.generate_background(creative_brief)
            except Exception as fallback_exc:
                raise ContentGenerationError(
                    code="IMAGE_GENERATION_FAILED",
                    message="Image generation failed and mock fallback also failed",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                ) from fallback_exc

    def _compose_svg(
        self,
        *,
        background_png: bytes,
        copy: dict[str, str],
        assets: dict[str, dict[str, str]],
    ) -> str:
        try:
            return self.svg_composer.compose(
                background_png=background_png,
                copy=copy,
                assets=assets,
            )
        except Exception as exc:
            raise ContentGenerationError(
                code="SVG_COMPOSITION_FAILED",
                message="SVG composition failed",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    def _export_png(self, *, composition_svg: str, copy: dict[str, str]) -> bytes:
        try:
            return self.png_exporter.export(
                composition_svg=composition_svg,
                scene=BannerRenderScene(copy=copy),
            )
        except Exception as exc:
            raise ContentGenerationError(
                code="PNG_EXPORT_FAILED",
                message="PNG export failed",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    def _upload_assets(
        self,
        *,
        keys: dict[str, str],
        creative_brief: dict[str, object],
        copy: dict[str, str],
        background: GeneratedImage,
        composition_svg: str,
        manifest: dict[str, object],
        final_banner_png: bytes,
    ) -> None:
        objects = [
            StorageObject(
                key=keys["creative_brief"],
                body=self._json_bytes(creative_brief),
                content_type="application/json",
            ),
            StorageObject(
                key=keys["copy"],
                body=self._json_bytes(copy),
                content_type="application/json",
            ),
            StorageObject(
                key=keys["background"],
                body=background.body,
                content_type=background.content_type,
            ),
            StorageObject(
                key=keys["composition_svg"],
                body=composition_svg.encode("utf-8"),
                content_type="image/svg+xml",
            ),
            StorageObject(
                key=keys["manifest"],
                body=self._json_bytes(manifest),
                content_type="application/json",
            ),
            StorageObject(
                key=keys["final_banner"],
                body=final_banner_png,
                content_type="image/png",
            ),
        ]
        try:
            self.storage.upload_objects(objects)
        except Exception as exc:
            raise ContentGenerationError(
                code="S3_UPLOAD_FAILED",
                message="Generated asset upload failed",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc

    def _update_creative_url(self, creative_id: int, content_url: str) -> None:
        try:
            updated = self.repository.update_ad_creative_image_url(creative_id, content_url)
            if updated is None:
                raise RuntimeError("ad_creatives row disappeared during update")
            self.repository.commit()
        except Exception as exc:
            raise ContentGenerationError(
                code="AD_CREATIVE_UPDATE_FAILED",
                message="Failed to update ad_creatives.image_url",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    def _build_asset_keys(
        self,
        *,
        recommendation_result_id: int,
        action_id: str,
        creative_id: str,
        generation_id: str,
    ) -> dict[str, str]:
        base_key = (
            f"ad-creatives/recommendation-results/{recommendation_result_id}"
            f"/actions/{action_id}"
            f"/creatives/{creative_id}"
            f"/generations/{generation_id}"
        )
        if self.asset_key_prefix:
            base_key = f"{self.asset_key_prefix}/{base_key}"
        return {
            "creative_brief": f"{base_key}/source/creative-brief.json",
            "copy": f"{base_key}/source/copy.json",
            "background": f"{base_key}/assets/background.png",
            "composition_svg": f"{base_key}/source/composition.svg",
            "manifest": f"{base_key}/source/manifest.json",
            "final_banner": f"{base_key}/final/banner.png",
        }

    def _build_manifest(
        self,
        *,
        generation_id: str,
        recommendation_result_id: int,
        action_id: str,
        creative_id: str,
        urls: dict[str, str],
        copy: dict[str, str],
        background: GeneratedImage,
    ) -> dict[str, object]:
        return {
            "generation_id": generation_id,
            "recommendation_result_id": recommendation_result_id,
            "action_id": action_id,
            "creative_id": creative_id,
            "format": {
                "width": 1200,
                "height": 628,
                "final_type": "image/png",
                "source_type": "image/svg+xml",
            },
            "assets": {
                "creative_brief_url": urls["creative_brief"],
                "copy_url": urls["copy"],
                "background_url": urls["background"],
                "composition_svg_url": urls["composition_svg"],
                "manifest_url": urls["manifest"],
                "final_banner_url": urls["final_banner"],
            },
            "copy": copy,
            "image_provider": {
                "provider": background.provider_name,
                "model": background.model,
            },
        }

    def _new_generation_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"gen_{timestamp}_{uuid4().hex[:8]}"

    def _json_bytes(self, payload: object) -> bytes:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
