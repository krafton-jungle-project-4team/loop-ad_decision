from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import quote, urlencode, urlsplit

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
_OFFER_SET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_DEAL_CODE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_STOREFRONT_URL = urlsplit(STOREFRONT_ORIGIN)


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
        offer_set_id: str | None = None,
    ) -> Mapping[str, Any] | None: ...


class PromotionOfferCatalogInvalidProjectId(ValueError):
    """The project cannot be resolved inside the brand-context namespace."""


class PromotionOfferCatalogInvalidOfferSetId(ValueError):
    """The offer set cannot be resolved safely inside the project manifest."""


class PromotionOfferCatalogNotFound(RuntimeError):
    """The project has no current, usable promotion offer catalog."""


class PromotionOfferCatalogUnavailable(RuntimeError):
    """The current catalog could not be read safely."""


class PromotionOfferCatalogService:
    def __init__(self, loader: PromotionOfferCatalogLoader) -> None:
        self._loader = loader

    @log_context_scope
    def list_offers(
        self,
        *,
        project_id: str,
        offer_set_id: str | None = None,
    ) -> PromotionOfferCatalogResponse:
        started_at = now_ms()
        log.info(
            "started",
            {
                "projectIdLength": len(str(project_id)),
                "offerSetIdProvided": offer_set_id is not None,
            },
        )
        try:
            project_id = _validated_project_id(project_id)
        except PromotionOfferCatalogInvalidProjectId as exc:
            log.warn("promotion_offer_project_id_invalid", {"err": exc})
            raise
        try:
            offer_set_id = _validated_offer_set_id(offer_set_id)
        except PromotionOfferCatalogInvalidOfferSetId as exc:
            log.warn("promotion_offer_set_id_invalid", {"err": exc})
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
            if offer_set_id is None:
                catalog = self._loader.load_offer_catalog(
                    project_id=project_id,
                    snapshot=snapshot,
                )
            else:
                catalog = self._loader.load_offer_catalog(
                    project_id=project_id,
                    snapshot=snapshot,
                    offer_set_id=offer_set_id,
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
            if exc.code in {
                "brand_context_object_missing",
                "brand_context_offer_set_unknown",
            }:
                log.warn(
                    "promotion_offer_catalog_not_found",
                    {"err": exc, "reason": exc.code},
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

        selected_offer_set_id = _optional_identifier(
            catalog.get("offer_set_id"),
            pattern=_OFFER_SET_ID_PATTERN,
        )
        if selected_offer_set_id is _INVALID or (
            offer_set_id is not None and selected_offer_set_id != offer_set_id
        ):
            log.warn(
                "promotion_offer_catalog_invalid",
                {"reason": "offer_set_identity_invalid"},
            )
            raise PromotionOfferCatalogUnavailable

        landing_url = _optional_storefront_url(catalog.get("landing_url"))
        if landing_url is _INVALID:
            log.warn(
                "promotion_offer_catalog_invalid",
                {"reason": "landing_url_invalid"},
            )
            raise PromotionOfferCatalogUnavailable

        deal_code = _optional_identifier(
            catalog.get("deal_code"),
            pattern=_DEAL_CODE_PATTERN,
        )
        if deal_code is _INVALID:
            log.warn(
                "promotion_offer_catalog_invalid",
                {"reason": "deal_code_invalid"},
            )
            raise PromotionOfferCatalogUnavailable
        if (
            deal_code is not None
            and landing_url is not None
            and landing_url != canonical_promotion_landing_url(deal_code)
        ):
            log.warn(
                "promotion_offer_catalog_invalid",
                {"reason": "landing_url_price_tier_mismatch"},
            )
            raise PromotionOfferCatalogUnavailable

        offers = _normalised_offers(
            catalog.get("hotels"),
            deal_code=deal_code,
        )
        response = PromotionOfferCatalogResponse(
            project_id=project_id,
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            offer_set_id=selected_offer_set_id,
            landing_url=landing_url,
            offers=offers,
        )
        log.info(
            "completed",
            {
                "catalogId": response.catalog_id,
                "catalogVersion": response.catalog_version,
                "offerSetId": response.offer_set_id,
                "durationMs": duration_ms(started_at),
                "offerCount": len(response.offers),
            },
        )
        return response


def canonical_offer_destination_url(
    offer_id: str,
    deal_code: str | None = None,
) -> str:
    """Return the canonical demo-storefront detail URL for an offer."""

    normalized_offer_id = _required_text(offer_id)
    if normalized_offer_id is None:
        raise ValueError("offer_id is required")
    destination_url = (
        f"{STOREFRONT_ORIGIN}/hotel/{quote(normalized_offer_id, safe='')}"
    )
    normalized_deal_code = _optional_identifier(
        deal_code,
        pattern=_DEAL_CODE_PATTERN,
    )
    if normalized_deal_code is _INVALID:
        raise ValueError("deal_code is invalid")
    if normalized_deal_code is None:
        return destination_url
    return f"{destination_url}?{urlencode({'deal': normalized_deal_code})}"


def canonical_promotion_landing_url(deal_code: str | None = None) -> str:
    """Return the demo-storefront promotion URL for one catalog price tier."""

    normalized_deal_code = _optional_identifier(
        deal_code,
        pattern=_DEAL_CODE_PATTERN,
    )
    if normalized_deal_code is _INVALID:
        raise ValueError("deal_code is invalid")
    landing_url = f"{STOREFRONT_ORIGIN}/search"
    if normalized_deal_code is None:
        return landing_url
    return f"{landing_url}?{urlencode({'deal': normalized_deal_code})}"


def validated_storefront_url(value: object) -> str:
    """Validate and return an HTTPS URL owned by the demo storefront."""

    return _storefront_url(value)


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


def _validated_offer_set_id(value: object) -> str | None:
    if value is None:
        return None
    offer_set_id = str(value).strip()
    if not _OFFER_SET_ID_PATTERN.fullmatch(offer_set_id):
        raise PromotionOfferCatalogInvalidOfferSetId
    return offer_set_id


def _normalised_offers(
    value: object,
    *,
    deal_code: str | None,
) -> list[PromotionOfferResponse]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise PromotionOfferCatalogUnavailable

    offers: list[PromotionOfferResponse] = []
    seen_offer_ids: set[str] = set()
    for raw_offer in value:
        if not isinstance(raw_offer, Mapping):
            continue
        offer = _normalised_offer(raw_offer, deal_code=deal_code)
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


def _normalised_offer(
    value: Mapping[str, Any],
    *,
    deal_code: str | None,
) -> PromotionOfferResponse | None:
    offer_id = _required_text(value.get("offer_id"))
    hotel_name = _required_text(value.get("hotel_name"))
    destination_id = _required_text(value.get("destination_id"))
    currency = _required_text(value.get("currency"))
    sale_price = _nonnegative_int(value.get("sale_price_per_night"))
    original_price = _optional_nonnegative_int(
        value.get("original_price_per_night")
    )
    promotion_price = _optional_nonnegative_int(
        value.get("promotion_price_per_night")
    )
    discount_rate = _optional_nonnegative_int(
        value.get("discount_rate_percent")
    )
    additional_discount_rate = _optional_nonnegative_int(
        value.get("additional_discount_rate_percent")
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
        or promotion_price is _INVALID
        or discount_rate is _INVALID
        or additional_discount_rate is _INVALID
        or not _has_valid_price_order(
            sale_price=sale_price,
            promotion_price=promotion_price,
            original_price=original_price,
        )
    ):
        return None
    try:
        image_url = public_offer_image_url(image_path)
        raw_destination_url = value.get("destination_url")
        expected_destination_url = canonical_offer_destination_url(
            offer_id,
            deal_code=deal_code,
        )
        destination_url = (
            expected_destination_url
            if raw_destination_url is None
            else _storefront_url(raw_destination_url)
        )
        if destination_url != expected_destination_url:
            raise ValueError("catalog destination_url is not canonical")
    except ValueError:
        return None
    return PromotionOfferResponse(
        offer_id=offer_id,
        hotel_name=hotel_name,
        destination_id=destination_id,
        currency=currency,
        sale_price_per_night=sale_price,
        original_price_per_night=original_price,
        promotion_price_per_night=promotion_price,
        discount_rate_percent=discount_rate,
        additional_discount_rate_percent=additional_discount_rate,
        image_url=image_url,
        destination_url=destination_url,
    )


_INVALID = object()


def _has_valid_price_order(
    *,
    sale_price: int,
    promotion_price: int | None,
    original_price: int | None,
) -> bool:
    if promotion_price is not None and promotion_price < sale_price:
        return False
    comparison_price = (
        promotion_price if promotion_price is not None else sale_price
    )
    return original_price is None or original_price >= comparison_price


def _required_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_identifier(
    value: object,
    *,
    pattern: re.Pattern[str],
) -> str | None | object:
    if value is None:
        return None
    text = _required_text(value)
    if text is None or pattern.fullmatch(text) is None:
        return _INVALID
    return text


def _optional_storefront_url(value: object) -> str | None | object:
    if value is None:
        return None
    try:
        return _storefront_url(value)
    except ValueError:
        return _INVALID


def _storefront_url(value: object) -> str:
    url = _required_text(value)
    if url is None or "\\" in url:
        raise ValueError("catalog URL is invalid")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("catalog URL is invalid") from exc
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != _STOREFRONT_URL.hostname
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("catalog URL must use the HTTPS demo storefront origin")
    return url


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_nonnegative_int(value: object) -> int | None | object:
    if value is None:
        return None
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None else _INVALID
