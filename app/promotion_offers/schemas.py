from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class PromotionOfferResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    offer_id: str = Field(min_length=1)
    hotel_name: str = Field(min_length=1)
    destination_id: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    sale_price_per_night: int = Field(ge=0)
    original_price_per_night: int | None = Field(default=None, ge=0)
    promotion_price_per_night: int | None = Field(default=None, ge=0)
    discount_rate_percent: int | None = Field(default=None, ge=0)
    additional_discount_rate_percent: int | None = Field(default=None, ge=0)
    image_url: HttpUrl
    destination_url: HttpUrl


class PromotionOfferCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str = Field(min_length=1)
    catalog_id: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    offer_set_id: str | None = Field(default=None, min_length=1)
    landing_url: HttpUrl | None = None
    offers: list[PromotionOfferResponse]


class PromotionOfferApiError(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    status_code: int = Field(alias="statusCode")
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class PromotionOfferApiErrorEnvelope(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    request_id: str = Field(alias="requestId", min_length=1)
    error: PromotionOfferApiError
