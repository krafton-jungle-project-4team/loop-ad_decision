from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from structlog.testing import capture_logs

from app.generation.brand_context import BrandContextSnapshot
from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
)
from app.promotion_offers.service import (
    PromotionOfferCatalogInvalidOfferSetId,
    PromotionOfferCatalogInvalidProjectId,
    PromotionOfferCatalogNotFound,
    PromotionOfferCatalogService,
    PromotionOfferCatalogUnavailable,
    canonical_offer_destination_url,
    canonical_promotion_landing_url,
    public_offer_image_url,
)


SNAPSHOT = BrandContextSnapshot(
    context_version="v2",
    manifest_key="brand-context/demo_project/manifests/v2/manifest.json",
    manifest_sha256="a" * 64,
    guide_version="v2",
    asset_manifest_version="v2",
    catalog_version="v2",
)


def test_service_returns_sorted_deduplicated_usable_offers() -> None:
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels",
            "catalog_version": "v2",
            "hotels": [
                hotel(
                    offer_id="okinawa-onna-coral-019",
                    hotel_name="Onna Coral Comfort Hotel",
                    image_path="/stayloop/hotels/okinawa-onna-coral-019.jpg",
                ),
                hotel(
                    offer_id="jeju-ocean-breeze-006",
                    hotel_name="Jeju Ocean Breeze Resort",
                    image_path="/stayloop/hotels/jeju-ocean-breeze-006.jpg",
                ),
                hotel(
                    offer_id="jeju-ocean-breeze-006",
                    hotel_name="Duplicate must be ignored",
                    image_path="/stayloop/hotels/duplicate.jpg",
                ),
                hotel(
                    offer_id="missing-image",
                    hotel_name="Missing Image",
                    image_path="",
                ),
                hotel(
                    offer_id="invalid/offer-id",
                    hotel_name="Invalid Offer ID",
                    image_path="/stayloop/hotels/invalid-offer.jpg",
                ),
                {
                    **hotel(
                        offer_id="invalid-currency",
                        hotel_name="Invalid Currency",
                        image_path="/stayloop/hotels/invalid-currency.jpg",
                    ),
                    "currency": "won",
                },
                {"offer_id": "incomplete"},
            ],
        }
    )

    with capture_logs() as logs:
        result = PromotionOfferCatalogService(loader).list_offers(
            project_id="demo_project"
        )

    assert result.project_id == "demo_project"
    assert result.catalog_id == "black-friday-hotels"
    assert result.catalog_version == "v2"
    assert result.offer_set_id is None
    assert result.landing_url is None
    assert [offer.offer_id for offer in result.offers] == [
        "jeju-ocean-breeze-006",
        "okinawa-onna-coral-019",
    ]
    assert str(result.offers[0].image_url) == (
        "https://demo-shoppingmall.dev.loop-ad.org"
        "/stayloop/hotels/jeju-ocean-breeze-006.jpg"
    )
    assert str(result.offers[0].destination_url) == (
        "https://demo-shoppingmall.dev.loop-ad.org"
        "/hotel/jeju-ocean-breeze-006"
    )
    completed = next(record for record in logs if record["event"] == "completed")
    assert completed["catalogId"] == "black-friday-hotels"
    assert completed["offerCount"] == 2
    assert loader.requested_offer_set_ids == [None]


def test_service_selects_offer_set_and_exposes_safe_catalog_urls() -> None:
    destination_url = (
        "https://demo-shoppingmall.dev.loop-ad.org"
        "/hotel/jeju-ocean-breeze-006?deal=summer-lastcall"
    )
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "offer_set_id": "summer-lastcall",
            "deal_code": "summer-lastcall",
            "landing_url": (
                "https://demo-shoppingmall.dev.loop-ad.org"
                "/search?deal=summer-lastcall"
            ),
            "hotels": [
                {
                    **hotel(
                        offer_id="jeju-ocean-breeze-006",
                        hotel_name="Jeju Ocean Breeze Resort",
                        image_path=(
                            "/stayloop/hotels/jeju-ocean-breeze-006.jpg"
                        ),
                    ),
                    "destination_url": destination_url,
                }
            ],
        }
    )

    result = PromotionOfferCatalogService(loader).list_offers(
        project_id="demo_project",
        offer_set_id="summer-lastcall",
    )

    assert result.offer_set_id == "summer-lastcall"
    assert str(result.landing_url) == (
        "https://demo-shoppingmall.dev.loop-ad.org"
        "/search?deal=summer-lastcall"
    )
    assert str(result.offers[0].destination_url) == destination_url
    assert loader.requested_offer_set_ids == ["summer-lastcall"]


def test_service_adds_catalog_deal_to_canonical_offer_url() -> None:
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "offer_set_id": "summer-lastcall",
            "deal_code": "summer-lastcall",
            "hotels": [
                hotel(
                    offer_id="jeju-ocean-breeze-006",
                    hotel_name="Jeju Ocean Breeze Resort",
                    image_path="/stayloop/hotels/jeju-ocean-breeze-006.jpg",
                )
            ],
        }
    )

    result = PromotionOfferCatalogService(loader).list_offers(
        project_id="demo_project",
        offer_set_id="summer-lastcall",
    )

    assert str(result.offers[0].destination_url) == (
        "https://demo-shoppingmall.dev.loop-ad.org"
        "/hotel/jeju-ocean-breeze-006?deal=summer-lastcall"
    )


def test_service_filters_cross_origin_offer_destination_url() -> None:
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "hotels": [
                {
                    **hotel(
                        offer_id="jeju-ocean-breeze-006",
                        hotel_name="Jeju Ocean Breeze Resort",
                        image_path=(
                            "/stayloop/hotels/jeju-ocean-breeze-006.jpg"
                        ),
                    ),
                    "destination_url": "https://untrusted.example/hotel/006",
                }
            ],
        }
    )

    result = PromotionOfferCatalogService(loader).list_offers(
        project_id="demo_project"
    )

    assert result.offers == []


def test_service_filters_same_origin_noncanonical_offer_destination_url() -> None:
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "deal_code": "summer-lastcall",
            "hotels": [
                {
                    **hotel(
                        offer_id="jeju-ocean-breeze-006",
                        hotel_name="Jeju Ocean Breeze Resort",
                        image_path=(
                            "/stayloop/hotels/jeju-ocean-breeze-006.jpg"
                        ),
                    ),
                    "destination_url": (
                        "https://demo-shoppingmall.dev.loop-ad.org/"
                        "hotel/jeju-ocean-breeze-006"
                    ),
                }
            ],
        }
    )

    result = PromotionOfferCatalogService(loader).list_offers(
        project_id="demo_project"
    )

    assert result.offers == []


def test_service_rejects_cross_origin_catalog_landing_url() -> None:
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "landing_url": "https://untrusted.example/search",
            "hotels": [],
        }
    )

    with pytest.raises(PromotionOfferCatalogUnavailable):
        PromotionOfferCatalogService(loader).list_offers(
            project_id="demo_project"
        )


def test_service_rejects_landing_url_for_the_wrong_price_tier() -> None:
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "deal_code": "summer-lastcall",
            "landing_url": "https://demo-shoppingmall.dev.loop-ad.org/search",
            "hotels": [],
        }
    )

    with pytest.raises(PromotionOfferCatalogUnavailable):
        PromotionOfferCatalogService(loader).list_offers(
            project_id="demo_project"
        )


def test_service_allows_an_empty_filtered_offer_list() -> None:
    loader = StubCatalogLoader(
        catalog={
            "catalog_id": "black-friday-hotels",
            "catalog_version": "v2",
            "hotels": [{"offer_id": "incomplete"}],
        }
    )

    result = PromotionOfferCatalogService(loader).list_offers(
        project_id="demo_project"
    )

    assert result.offers == []


def test_service_keeps_legacy_loader_call_compatible_without_offer_set() -> None:
    result = PromotionOfferCatalogService(LegacyCatalogLoader()).list_offers(
        project_id="demo_project"
    )

    assert result.catalog_id == "black-friday-hotels"


@pytest.mark.parametrize("project_id", ["", "Demo_Project", "../demo", "a" * 101])
def test_service_rejects_invalid_project_ids(project_id: str) -> None:
    with pytest.raises(PromotionOfferCatalogInvalidProjectId):
        PromotionOfferCatalogService(StubCatalogLoader()).list_offers(
            project_id=project_id
        )


@pytest.mark.parametrize(
    "offer_set_id",
    ["", "Summer-Lastcall", "../lastcall", "a" * 101],
)
def test_service_rejects_invalid_offer_set_ids(offer_set_id: str) -> None:
    with pytest.raises(PromotionOfferCatalogInvalidOfferSetId):
        PromotionOfferCatalogService(StubCatalogLoader()).list_offers(
            project_id="demo_project",
            offer_set_id=offer_set_id,
        )


def test_service_reports_missing_snapshot_or_catalog_as_not_found() -> None:
    with pytest.raises(PromotionOfferCatalogNotFound):
        PromotionOfferCatalogService(
            StubCatalogLoader(snapshot=None)
        ).list_offers(project_id="demo_project")

    with pytest.raises(PromotionOfferCatalogNotFound):
        PromotionOfferCatalogService(
            StubCatalogLoader(catalog=None)
        ).list_offers(project_id="demo_project")


def test_service_reports_unknown_offer_set_as_not_found() -> None:
    loader = StubCatalogLoader(
        error=PermanentGenerationError(
            code="brand_context_offer_set_unknown",
            safe_message="The requested promotion offer set was unavailable.",
        )
    )

    with pytest.raises(PromotionOfferCatalogNotFound):
        PromotionOfferCatalogService(loader).list_offers(
            project_id="demo_project",
            offer_set_id="summer-unknown",
        )


@pytest.mark.parametrize(
    "error",
    [
        RetryableGenerationError(
            code="brand_context_read_failed",
            safe_message="temporarily unavailable",
        ),
        PermanentGenerationError(
            code="brand_context_catalog_invalid",
            safe_message="invalid catalog",
        ),
    ],
)
def test_service_maps_unreadable_catalogs_to_unavailable(error: Exception) -> None:
    with pytest.raises(PromotionOfferCatalogUnavailable):
        PromotionOfferCatalogService(
            StubCatalogLoader(error=error)
        ).list_offers(project_id="demo_project")


def test_service_maps_missing_catalog_object_to_not_found() -> None:
    with pytest.raises(PromotionOfferCatalogNotFound):
        PromotionOfferCatalogService(
            StubCatalogLoader(
                error=PermanentGenerationError(
                    code="brand_context_object_missing",
                    safe_message="missing",
                )
            )
        ).list_offers(project_id="demo_project")


def test_public_url_helpers_encode_ids_and_reject_unsafe_image_paths() -> None:
    assert canonical_offer_destination_url("hotel/id with space") == (
        "https://demo-shoppingmall.dev.loop-ad.org/hotel/hotel%2Fid%20with%20space"
    )
    assert public_offer_image_url("/assets/hotel.jpg") == (
        "https://demo-shoppingmall.dev.loop-ad.org/assets/hotel.jpg"
    )
    assert canonical_offer_destination_url(
        "jeju-ocean-breeze-006",
        deal_code="summer-lastcall",
    ) == (
        "https://demo-shoppingmall.dev.loop-ad.org"
        "/hotel/jeju-ocean-breeze-006?deal=summer-lastcall"
    )
    assert canonical_promotion_landing_url("summer-lastcall") == (
        "https://demo-shoppingmall.dev.loop-ad.org/"
        "search?deal=summer-lastcall"
    )
    with pytest.raises(ValueError):
        canonical_offer_destination_url(
            "jeju-ocean-breeze-006",
            deal_code="summer&redirect=https://untrusted.example",
        )
    with pytest.raises(ValueError):
        public_offer_image_url("//untrusted.example/hotel.jpg")


def hotel(
    *,
    offer_id: str,
    hotel_name: str,
    image_path: str,
) -> dict[str, Any]:
    return {
        "offer_id": offer_id,
        "hotel_name": hotel_name,
        "destination_id": "jeju",
        "currency": "KRW",
        "sale_price_per_night": 278_000,
        "original_price_per_night": 342_000,
        "discount_rate_percent": 19,
        "image_path": image_path,
    }


class StubCatalogLoader:
    def __init__(
        self,
        *,
        snapshot: BrandContextSnapshot | None = SNAPSHOT,
        catalog: Mapping[str, Any] | None | object = ...,
        error: Exception | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.catalog = (
            {
                "catalog_id": "black-friday-hotels",
                "catalog_version": "v2",
                "hotels": [],
            }
            if catalog is ...
            else catalog
        )
        self.error = error
        self.requested_offer_set_ids: list[str | None] = []

    def resolve_snapshot(self, *, project_id: str) -> BrandContextSnapshot | None:
        assert project_id == "demo_project"
        if self.error is not None:
            raise self.error
        return self.snapshot

    def load_offer_catalog(
        self,
        *,
        project_id: str,
        snapshot: BrandContextSnapshot,
        offer_set_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        assert project_id == "demo_project"
        assert snapshot is SNAPSHOT
        self.requested_offer_set_ids.append(offer_set_id)
        assert self.catalog is None or isinstance(self.catalog, Mapping)
        return self.catalog


class LegacyCatalogLoader:
    def resolve_snapshot(self, *, project_id: str) -> BrandContextSnapshot:
        assert project_id == "demo_project"
        return SNAPSHOT

    def load_offer_catalog(
        self,
        *,
        project_id: str,
        snapshot: BrandContextSnapshot,
    ) -> Mapping[str, Any]:
        assert project_id == "demo_project"
        assert snapshot is SNAPSHOT
        return {
            "catalog_id": "black-friday-hotels",
            "catalog_version": "v2",
            "hotels": [],
        }
