from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class FunnelMetricFilters(BaseModel):
    channel: str | None = None
    campaign_id: str | None = None
    age_group: str | None = None
    gender: str | None = None
    device: str | None = None
    category: str | None = None
    product_id: str | None = None
    inventory_status: str | None = None

    model_config = ConfigDict(extra="forbid")


class FunnelMetricRequest(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    filters: FunnelMetricFilters | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("window_start", "window_end")
    @classmethod
    def validate_timezone_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("window_start and window_end must be timezone-aware datetimes")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "FunnelMetricRequest":
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be earlier than window_end")
        return self


class FunnelMetrics(BaseModel):
    product_view_sessions: int
    add_to_cart_sessions: int
    checkout_start_sessions: int
    purchase_sessions: int
    view_to_cart_rate: float | None
    cart_to_checkout_rate: float | None
    checkout_to_purchase_rate: float | None
    view_to_purchase_rate: float | None
    view_to_cart_dropoff_rate: float | None
    cart_to_checkout_dropoff_rate: float | None
    checkout_to_purchase_dropoff_rate: float | None


class FunnelMetricResponse(BaseModel):
    project_id: str
    window_start: datetime
    window_end: datetime
    segment: dict[str, str | None]
    metrics: FunnelMetrics
