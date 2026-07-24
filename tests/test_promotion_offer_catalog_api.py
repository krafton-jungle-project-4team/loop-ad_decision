from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.promotion_offers.router import get_promotion_offer_catalog_service
from app.promotion_offers.schemas import PromotionOfferCatalogResponse
from app.promotion_offers.service import (
    PromotionOfferCatalogInvalidOfferSetId,
    PromotionOfferCatalogInvalidProjectId,
    PromotionOfferCatalogNotFound,
    PromotionOfferCatalogUnavailable,
)
from tests.config_env import required_env_values


REQUEST_ID = "promotion-offers-request-1"


def test_api_returns_the_decision_promotion_offer_contract() -> None:
    client, headers = make_client(
        ReturningService(
            PromotionOfferCatalogResponse.model_validate(
                {
                    "project_id": "demo_project",
                    "catalog_id": "black-friday-hotels",
                    "catalog_version": "v2",
                    "offer_set_id": "summer-base",
                    "landing_url": (
                        "https://demo-shoppingmall.dev.loop-ad.org/search"
                    ),
                    "offers": [
                        {
                            "offer_id": "jeju-ocean-breeze-006",
                            "hotel_name": "Jeju Ocean Breeze Resort",
                            "destination_id": "jeju",
                            "currency": "KRW",
                            "sale_price_per_night": 278000,
                            "original_price_per_night": 342000,
                            "discount_rate_percent": 19,
                            "image_url": (
                                "https://demo-shoppingmall.dev.loop-ad.org"
                                "/assets/hotels/jeju-ocean-breeze-006.jpg"
                            ),
                            "destination_url": (
                                "https://demo-shoppingmall.dev.loop-ad.org"
                                "/hotel/jeju-ocean-breeze-006"
                            ),
                        }
                    ],
                }
            )
        )
    )

    response = client.get(
        "/decision/v1/projects/demo_project/promotion-offers",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["offers"][0]["offer_id"] == "jeju-ocean-breeze-006"
    assert response.json()["offers"][0]["destination_url"] == (
        "https://demo-shoppingmall.dev.loop-ad.org/hotel/jeju-ocean-breeze-006"
    )
    assert response.json()["offer_set_id"] == "summer-base"
    assert response.json()["landing_url"] == (
        "https://demo-shoppingmall.dev.loop-ad.org/search"
    )


def test_api_passes_optional_offer_set_id_to_service() -> None:
    response_model = PromotionOfferCatalogResponse.model_validate(
        {
            "project_id": "demo_project",
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "offer_set_id": "summer-lastcall",
            "landing_url": (
                "https://demo-shoppingmall.dev.loop-ad.org"
                "/search?deal=summer-lastcall"
            ),
            "offers": [],
        }
    )
    service = OfferSetAwareService(response_model)
    client, headers = make_client(service)

    response = client.get(
        (
            "/decision/v1/projects/demo_project/promotion-offers"
            "?offer_set_id=summer-lastcall"
        ),
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["offer_set_id"] == "summer-lastcall"
    assert service.requested_offer_set_id == "summer-lastcall"


def test_api_requires_the_internal_key() -> None:
    client, _headers = make_client(
        RaisingService(PromotionOfferCatalogNotFound())
    )

    response = client.get(
        "/decision/v1/projects/demo_project/promotion-offers",
        headers={"X-Request-Id": REQUEST_ID},
    )

    assert_error_envelope(
        response,
        status_code=401,
        code="internal_api_key_invalid",
    )


def test_api_rejects_an_incorrect_internal_key() -> None:
    client, headers = make_client(
        RaisingService(PromotionOfferCatalogNotFound())
    )
    headers["X-Loop-Ad-Internal-Key"] = "wrong"
    headers["X-Request-Id"] = REQUEST_ID

    response = client.get(
        "/decision/v1/projects/demo_project/promotion-offers",
        headers=headers,
    )

    assert_error_envelope(
        response,
        status_code=401,
        code="internal_api_key_invalid",
    )


def test_api_returns_the_documented_invalid_project_error_envelope() -> None:
    response = request_with_error(PromotionOfferCatalogInvalidProjectId())

    assert_error_envelope(response, status_code=400, code="project_id_invalid")


def test_api_returns_the_documented_invalid_offer_set_error_envelope() -> None:
    client, headers = make_client(
        OfferSetAwareRaisingService(PromotionOfferCatalogInvalidOfferSetId())
    )

    response = client.get(
        (
            "/decision/v1/projects/demo_project/promotion-offers"
            "?offer_set_id=../lastcall"
        ),
        headers={**headers, "X-Request-Id": REQUEST_ID},
    )

    assert_error_envelope(
        response,
        status_code=400,
        code="offer_set_id_invalid",
    )


def test_api_returns_the_documented_not_found_error_envelope() -> None:
    response = request_with_error(PromotionOfferCatalogNotFound())

    assert_error_envelope(
        response,
        status_code=404,
        code="promotion_offer_catalog_not_found",
    )


def test_api_returns_the_documented_unavailable_error_envelope() -> None:
    response = request_with_error(PromotionOfferCatalogUnavailable())

    assert_error_envelope(
        response,
        status_code=503,
        code="promotion_offer_catalog_unavailable",
    )


def test_generated_error_request_id_matches_the_response_header() -> None:
    client, headers = make_client(
        RaisingService(PromotionOfferCatalogUnavailable())
    )

    response = client.get(
        "/decision/v1/projects/demo_project/promotion-offers",
        headers=headers,
    )

    assert response.json()["requestId"] == response.headers["X-Request-Id"]


def request_with_error(error: Exception):
    client, headers = make_client(RaisingService(error))
    return client.get(
        "/decision/v1/projects/demo_project/promotion-offers",
        headers={**headers, "X-Request-Id": REQUEST_ID},
    )


def assert_error_envelope(response, *, status_code: int, code: str) -> None:
    assert response.status_code == status_code
    assert response.headers["X-Request-Id"] == REQUEST_ID
    payload = response.json()
    assert payload["requestId"] == REQUEST_ID
    assert payload["error"]["statusCode"] == status_code
    assert payload["error"]["code"] == code
    assert payload["error"]["message"]


def make_client(service) -> tuple[TestClient, dict[str, str]]:
    env = valid_env()
    app = create_app(settings=load_settings(env))
    app.dependency_overrides[get_promotion_offer_catalog_service] = lambda: service
    return TestClient(app), {
        "X-Loop-Ad-Internal-Key": env["LOOPAD_INTERNAL_API_KEY"]
    }


def valid_env() -> dict[str, str]:
    values = required_env_values()
    values.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "15432",
        }
    )
    return values


class ReturningService:
    def __init__(self, response: PromotionOfferCatalogResponse) -> None:
        self.response = response

    def list_offers(self, *, project_id: str) -> PromotionOfferCatalogResponse:
        assert project_id == "demo_project"
        return self.response


class RaisingService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def list_offers(self, *, project_id: str) -> PromotionOfferCatalogResponse:
        del project_id
        raise self.error


class OfferSetAwareService:
    def __init__(self, response: PromotionOfferCatalogResponse) -> None:
        self.response = response
        self.requested_offer_set_id: str | None = None

    def list_offers(
        self,
        *,
        project_id: str,
        offer_set_id: str | None = None,
    ) -> PromotionOfferCatalogResponse:
        assert project_id == "demo_project"
        self.requested_offer_set_id = offer_set_id
        return self.response


class OfferSetAwareRaisingService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def list_offers(
        self,
        *,
        project_id: str,
        offer_set_id: str | None = None,
    ) -> PromotionOfferCatalogResponse:
        del project_id, offer_set_id
        raise self.error
