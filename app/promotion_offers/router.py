from __future__ import annotations

import hmac
import uuid

from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import JSONResponse

from app.dependencies import get_settings
from app.generation.brand_context_s3 import S3BrandContextLoader
from app.logging import REQUEST_ID_HEADER
from app.promotion_offers.schemas import (
    PromotionOfferApiError,
    PromotionOfferApiErrorEnvelope,
    PromotionOfferCatalogResponse,
)
from app.promotion_offers.service import (
    PromotionOfferCatalogInvalidOfferSetId,
    PromotionOfferCatalogInvalidProjectId,
    PromotionOfferCatalogNotFound,
    PromotionOfferCatalogService,
    PromotionOfferCatalogUnavailable,
)


router = APIRouter(
    prefix="/decision/v1/projects",
    tags=["promotion-offers"],
)


def get_promotion_offer_catalog_service(
    request: Request,
) -> PromotionOfferCatalogService:
    settings = get_settings(request)
    return PromotionOfferCatalogService(
        S3BrandContextLoader(
            bucket_name=settings.data_storage_bucket,
            base_prefix=settings.brand_context_base_prefix,
        )
    )


@router.get(
    "/{project_id}/promotion-offers",
    response_model=PromotionOfferCatalogResponse,
    status_code=status.HTTP_200_OK,
)
def list_promotion_offers(
    project_id: str,
    request: Request,
    offer_set_id: str | None = None,
    x_loop_ad_internal_key: str | None = Header(
        default=None,
        alias="X-Loop-Ad-Internal-Key",
    ),
    service: PromotionOfferCatalogService = Depends(
        get_promotion_offer_catalog_service
    ),
) -> PromotionOfferCatalogResponse | JSONResponse:
    settings = get_settings(request)
    if not x_loop_ad_internal_key or not hmac.compare_digest(
        x_loop_ad_internal_key,
        settings.internal_api_key,
    ):
        return _error_response(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="internal_api_key_invalid",
            message="internal API key is missing or invalid",
        )
    try:
        if offer_set_id is None:
            return service.list_offers(project_id=project_id)
        return service.list_offers(
            project_id=project_id,
            offer_set_id=offer_set_id,
        )
    except PromotionOfferCatalogInvalidProjectId:
        return _error_response(
            request,
            status_code=status.HTTP_400_BAD_REQUEST,
            code="project_id_invalid",
            message="project_id is invalid",
        )
    except PromotionOfferCatalogInvalidOfferSetId:
        return _error_response(
            request,
            status_code=status.HTTP_400_BAD_REQUEST,
            code="offer_set_id_invalid",
            message="offer_set_id is invalid",
        )
    except PromotionOfferCatalogNotFound:
        return _error_response(
            request,
            status_code=status.HTTP_404_NOT_FOUND,
            code="promotion_offer_catalog_not_found",
            message="promotion offer catalog is not configured",
        )
    except PromotionOfferCatalogUnavailable:
        return _error_response(
            request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="promotion_offer_catalog_unavailable",
            message="promotion offer catalog is temporarily unavailable",
        )


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    request_id = (
        str(getattr(request.state, "request_id", "") or "").strip()
        or str(request.headers.get(REQUEST_ID_HEADER) or "").strip()
        or uuid.uuid4().hex
    )
    envelope = PromotionOfferApiErrorEnvelope(
        request_id=request_id,
        error=PromotionOfferApiError(
            status_code=status_code,
            code=code,
            message=message,
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json", by_alias=True),
        headers={REQUEST_ID_HEADER: request_id},
    )
