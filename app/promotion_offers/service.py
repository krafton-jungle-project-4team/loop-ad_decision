from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import quote

from app.generation.brand_context import BrandContextSnapshot
from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
)
from app.logging import duration_ms, log, log_context_scope, now_ms
from app.promotion_offers.schemas import (
    PromotionOfferCatalogResponse,
    PromotionOfferResponse,
)


STOREFRONT_ORIGIN = "https://demo-shoppingmall.dev.loop-ad.org"
_PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_OFFER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


class PromotionOfferCatalogLoader(Protocol):
    def resolve_snapshot(
        self,
        *,
        project_id: str,
    ) -> BrandContextSnapshot | None: ...

    def load_offer_catalog(
        self,
        *,
        project_id: str,
        snapshot: BrandContextSnapshot,
    ) -> Mapping[str, Any] | None: ...


class PromotionOfferCatalogInvalidProjectId(ValueError):
    """The project cannot be resolved inside the brand-context namespace."""


class PromotionOfferCatalogNotFound(RuntimeError):
    """The project has no current, usable promotion offer catalog."""


class PromotionOfferCatalogUnavailable(RuntimeError):
    """The current catalog could not be read safely."""


class PromotionOfferCatalogService:
    def __init__(self, loader: PromotionOfferCatalogLoader) -> None:
        self._loader = loader

    @log_context_scope
    def list_offers(self, *, project_id: str) -> PromotionOfferCatalogResponse:
        started_at = now_ms()
        log.info("started", {"projectIdLength": len(str(project_id))})
        try:
            project_id = _validated_project_id(project_id)
        except PromotionOfferCatalogInvalidProjectId as exc:
            log.warn("promotion_offer_project_id_invalid", {"err": exc})
            raise
        log.assign_context({"projectId": project_id})
        try:
            snapshot = self._loader.resolve_snapshot(project_id=project_id)
            if snapshot is None:
                log.warn(
                    "promotion_offer_catalog_not_found",
                    {"reason": "brand_context_snapshot_missing"},
                )
                raise PromotionOfferCatalogNotFound
            catalog = self._loader.load_offer_catalog(
                project_id=project_id,
                snapshot=snapshot,
            )
            if catalog is None:
                log.warn(
                    "promotion_offer_catalog_not_found",
                    {"reason": "offer_catalog_missing"},
                )
                raise PromotionOfferCatalogNotFound
        except PromotionOfferCatalogNotFound:
            raise
        except RetryableGenerationError as exc:
            log.warn("promotion_offer_catalog_unavailable", {"err": exc})
            raise PromotionOfferCatalogUnavailable from exc
        except PermanentGenerationError as exc:
            if exc.code == "brand_context_object_missing":
                log.warn(
                    "promotion_offer_catalog_not_found",
                    {"err": exc, "reason": "catalog_object_missing"},
                )
                raise PromotionOfferCatalogNotFound from exc
            log.warn("promotion_offer_catalog_unavailable", {"err": exc})
            raise PromotionOfferCatalogUnavailable from exc
        except ValueError as exc:
            log.warn("promotion_offer_catalog_unavailable", {"err": exc})
            raise PromotionOfferCatalogUnavailable from exc

        catalog_id = _required_text(catalog.get("catalog_id"))
        catalog_version = _required_text(catalog.get("catalog_version"))
        if catalog_id is None or catalog_version is None:
            log.warn(
                "promotion_offer_catalog_invalid",
                {"reason": "catalog_identity_missing"},
            )
            raise PromotionOfferCatalogUnavailable

        offers = _normalised_offers(catalog.get("hotels"))
        response = PromotionOfferCatalogResponse(
            project_id=project_id,
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            offers=offers,
        )
        log.info(
            "completed",
            {
                "catalogId": response.catalog_id,
                "catalogVersion": response.catalog_version,
                "durationMs": duration_ms(started_at),
                "offerCount": len(response.offers),
            },
        )
        return response


def canonical_offer_destination_url(offer_id: str) -> str:
    """Return the canonical demo-storefront detail URL for an offer."""

    normalized_offer_id = _required_text(offer_id)
    if normalized_offer_id is None:
        raise ValueError("offer_id is required")
    return f"{STOREFRONT_ORIGIN}/hotel/{quote(normalized_offer_id, safe='')}"


def public_offer_image_url(image_path: str) -> str:
    """Resolve a verified manifest frontend path against the storefront origin."""

    normalized_path = _required_text(image_path)
    if (
        normalized_path is None
        or not normalized_path.startswith("/")
        or normalized_path.startswith("//")
        or ".." in normalized_path.split("/")
    ):
        raise ValueError("image_path must be a safe absolute frontend path")
    return f"{STOREFRONT_ORIGIN}{normalized_path}"


def _validated_project_id(value: object) -> str:
    project_id = str(value).strip()
    if not _PROJECT_ID_PATTERN.fullmatch(project_id):
        raise PromotionOfferCatalogInvalidProjectId
    return project_id


def _normalised_offers(value: object) -> list[PromotionOfferResponse]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise PromotionOfferCatalogUnavailable

    offers: list[PromotionOfferResponse] = []
    seen_offer_ids: set[str] = set()
    for raw_offer in value:
        if not isinstance(raw_offer, Mapping):
            continue
        offer = _normalised_offer(raw_offer)
        if offer is None or offer.offer_id in seen_offer_ids:
            continue
        seen_offer_ids.add(offer.offer_id)
        offers.append(offer)
    return sorted(
        offers,
        key=lambda offer: (
            offer.hotel_name.casefold(),
            offer.hotel_name,
            offer.offer_id,
        ),
    )


def _normalised_offer(value: Mapping[str, Any]) -> PromotionOfferResponse | None:
    offer_id = _required_text(value.get("offer_id"))
    hotel_name = _required_text(value.get("hotel_name"))
    destination_id = _required_text(value.get("destination_id"))
    currency = _required_text(value.get("currency"))
    sale_price = _nonnegative_int(value.get("sale_price_per_night"))
    original_price = _optional_nonnegative_int(
        value.get("original_price_per_night")
    )
    discount_rate = _optional_nonnegative_int(
        value.get("discount_rate_percent")
    )
    image_path = _required_text(value.get("image_path"))
    if (
        offer_id is None
        or _OFFER_ID_PATTERN.fullmatch(offer_id) is None
        or hotel_name is None
        or destination_id is None
        or currency is None
        or _CURRENCY_PATTERN.fullmatch(currency) is None
        or sale_price is None
        or image_path is None
        or original_price is _INVALID
        or discount_rate is _INVALID
    ):
        return None
    try:
        image_url = public_offer_image_url(image_path)
        destination_url = canonical_offer_destination_url(offer_id)
    except ValueError:
        return None
    return PromotionOfferResponse(
        offer_id=offer_id,
        hotel_name=hotel_name,
        destination_id=destination_id,
        currency=currency,
        sale_price_per_night=sale_price,
        original_price_per_night=original_price,
        discount_rate_percent=discount_rate,
        image_url=image_url,
        destination_url=destination_url,
    )


_INVALID = object()


def _required_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_nonnegative_int(value: object) -> int | None | object:
    if value is None:
        return None
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None else _INVALID
