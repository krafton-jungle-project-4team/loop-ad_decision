from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.generation.prompt_builder import PromotionOfferLink


EDITORIAL_VARIANT = "editorial"
OFFER_CARDS_VARIANT = "offer_cards"
COMPARISON_VARIANT = "comparison"
EMAIL_VARIANT_SEQUENCE = (
    EDITORIAL_VARIANT,
    OFFER_CARDS_VARIANT,
    COMPARISON_VARIANT,
)
EDITORIAL_TEMPLATE_VERSION = "email.editorial.v1"
OFFER_CARDS_TEMPLATE_VERSION = "email.offer-cards.v2"
COMPARISON_TEMPLATE_VERSION = "email.comparison.v1"
PRIMARY_REDIRECT_PLACEHOLDER = "{{redirect_url}}"
OPEN_PIXEL_PLACEHOLDER = "{{open_pixel_url}}"
UNSUBSCRIBE_PLACEHOLDER = "{{unsubscribe_url}}"
OFFER_REDIRECT_PATTERN = re.compile(r"^\{\{offer_redirect_url_([1-8])\}\}$")


def build_email_creative_extensions(
    *,
    option_index: int,
    landing_url: str | None,
    offer_links: Sequence[PromotionOfferLink],
    offer_catalog: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not offer_links:
        return {}
    if option_index < 1:
        raise ValueError("email option_index must be at least 1")
    variant_type = EMAIL_VARIANT_SEQUENCE[
        (option_index - 1) % len(EMAIL_VARIANT_SEQUENCE)
    ]
    link_targets: list[dict[str, Any]] = [
        {
            "placeholder": PRIMARY_REDIRECT_PLACEHOLDER,
            "target_type": "promotion",
        }
    ]
    extensions: dict[str, Any] = {
        "variant_type": variant_type,
        "link_targets": link_targets,
    }
    if offer_catalog is None:
        raise ValueError("email variants require a snapshotted offer catalog")
    catalog_hotels = offer_catalog.get("hotels")
    if not isinstance(catalog_hotels, list):
        raise ValueError("offer catalog hotels must be an array")
    hotels_by_id = {
        str(hotel.get("offer_id") or ""): hotel
        for hotel in catalog_hotels
        if isinstance(hotel, Mapping) and str(hotel.get("offer_id") or "")
    }
    image_origin = _public_image_origin(landing_url)
    offers: list[dict[str, Any]] = []
    for index, offer_link in enumerate(offer_links, start=1):
        catalog_hotel = hotels_by_id.get(offer_link.offer_id)
        if catalog_hotel is None:
            raise ValueError(
                f"offer catalog does not contain {offer_link.offer_id}"
            )
        image_path = _required_text(catalog_hotel, "image_path")
        image_url = _public_image_url(image_origin, image_path)
        offer: dict[str, Any] = {
            "offer_id": offer_link.offer_id,
            "hotel_name": _required_text(catalog_hotel, "hotel_name"),
            "destination_id": _required_text(
                catalog_hotel,
                "destination_id",
            ),
            "currency": _required_text(catalog_hotel, "currency"),
            "sale_price_per_night": _required_nonnegative_int(
                catalog_hotel,
                "sale_price_per_night",
            ),
            "original_price_per_night": _optional_nonnegative_int(
                catalog_hotel,
                "original_price_per_night",
            ),
            "discount_rate_percent": _optional_nonnegative_int(
                catalog_hotel,
                "discount_rate_percent",
            ),
            "image_url": image_url,
        }
        if variant_type == OFFER_CARDS_VARIANT:
            placeholder = f"{{{{offer_redirect_url_{index}}}}}"
            link_targets.append(
                {
                    "placeholder": placeholder,
                    "target_type": "offer",
                    "offer_id": offer_link.offer_id,
                    "destination_url": offer_link.destination_url,
                }
            )
            offer["redirect_placeholder"] = placeholder
        offers.append(offer)

    extensions["catalog"] = {
        "catalog_id": _required_text(offer_catalog, "catalog_id"),
        "catalog_version": _required_text(
            offer_catalog,
            "catalog_version",
        ),
    }
    if variant_type == EDITORIAL_VARIANT:
        extensions.update(
            {
                "template_version": EDITORIAL_TEMPLATE_VERSION,
                "featured_offers": _select_destination_offers(
                    offers,
                    per_destination=1,
                    maximum=2,
                ),
            }
        )
    elif variant_type == OFFER_CARDS_VARIANT:
        extensions.update(
            {
                "template_version": OFFER_CARDS_TEMPLATE_VERSION,
                "offers": offers,
            }
        )
    else:
        extensions.update(
            {
                "template_version": COMPARISON_TEMPLATE_VERSION,
                "comparison_offers": _select_destination_offers(
                    offers,
                    per_destination=2,
                    maximum=4,
                ),
            }
        )
    return extensions


def reusable_catalog_image_url(
    creative_extensions: Mapping[str, Any],
) -> str | None:
    """Return a renderer-used catalog image for variants that need no new hero."""

    variant_type = str(creative_extensions.get("variant_type") or "")
    collection_key = {
        OFFER_CARDS_VARIANT: "offers",
        COMPARISON_VARIANT: "comparison_offers",
    }.get(variant_type)
    if collection_key is None:
        return None
    raw_offers = creative_extensions.get(collection_key)
    if not isinstance(raw_offers, list) or not raw_offers:
        raise ValueError(f"{variant_type} requires renderer catalog images")
    first_offer = raw_offers[0]
    if not isinstance(first_offer, Mapping):
        raise ValueError(f"{variant_type} catalog image entry must be an object")
    image_url = str(first_offer.get("image_url") or "").strip()
    parsed = urlsplit(image_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"{variant_type} catalog image URL must be HTTPS")
    return image_url


def _select_destination_offers(
    offers: Sequence[Mapping[str, Any]],
    *,
    per_destination: int,
    maximum: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    destination_counts: dict[str, int] = {}
    selected_offer_ids: set[str] = set()
    for offer in offers:
        destination_id = str(offer.get("destination_id") or "").casefold()
        if destination_counts.get(destination_id, 0) >= per_destination:
            continue
        selected.append(dict(offer))
        selected_offer_ids.add(str(offer.get("offer_id") or ""))
        destination_counts[destination_id] = (
            destination_counts.get(destination_id, 0) + 1
        )
        if len(selected) == maximum:
            return selected
    for offer in offers:
        offer_id = str(offer.get("offer_id") or "")
        if offer_id in selected_offer_ids:
            continue
        selected.append(dict(offer))
        if len(selected) == maximum:
            break
    return selected


def email_required_placeholders(
    content_values: Mapping[str, Any],
) -> tuple[str, ...]:
    raw_targets = content_values.get("link_targets")
    if not isinstance(raw_targets, list):
        return (
            PRIMARY_REDIRECT_PLACEHOLDER,
            OPEN_PIXEL_PLACEHOLDER,
            UNSUBSCRIBE_PLACEHOLDER,
        )
    placeholders: list[str] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping):
            raise ValueError("creative link_targets entries must be objects")
        placeholder = _required_text(raw_target, "placeholder")
        if placeholder != PRIMARY_REDIRECT_PLACEHOLDER and not (
            OFFER_REDIRECT_PATTERN.fullmatch(placeholder)
        ):
            raise ValueError("creative redirect placeholder is invalid")
        placeholders.append(placeholder)
    if not placeholders or placeholders[0] != PRIMARY_REDIRECT_PLACEHOLDER:
        raise ValueError("creative link_targets must start with redirect_url")
    if len(placeholders) != len(set(placeholders)):
        raise ValueError("creative link_targets placeholders must be unique")
    return tuple(
        [
            *placeholders,
            OPEN_PIXEL_PLACEHOLDER,
            UNSUBSCRIBE_PLACEHOLDER,
        ]
    )


def _public_image_origin(landing_url: str | None) -> str:
    parsed = urlsplit(str(landing_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("offer card image origin requires an HTTPS landing_url")
    if parsed.username or parsed.password:
        raise ValueError("offer card image origin must not contain credentials")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _public_image_url(origin: str, path: str) -> str:
    if not path.startswith("/") or path.startswith("//") or ".." in path.split("/"):
        raise ValueError("offer card image_path is invalid")
    return f"{origin}{path}"


def _required_text(value: Mapping[str, Any], key: str) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise ValueError(f"offer catalog requires {key}")
    return text


def _required_nonnegative_int(value: Mapping[str, Any], key: str) -> int:
    raw = value.get(key)
    try:
        number = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"offer catalog {key} must be an integer") from exc
    if number < 0:
        raise ValueError(f"offer catalog {key} must not be negative")
    return number


def _optional_nonnegative_int(
    value: Mapping[str, Any],
    key: str,
) -> int | None:
    if value.get(key) is None:
        return None
    return _required_nonnegative_int(value, key)
