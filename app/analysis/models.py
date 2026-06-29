from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal


@dataclass(frozen=True)
class AnalysisWindow:
    analysis_date: date
    timezone: str
    window_start: datetime
    window_end: datetime


@dataclass(frozen=True)
class AnalysisResult:
    segment_count: int = 0
    membership_count: int = 0
    metric_count: int = 0
    anomaly_count: int = 0
    root_cause_count: int = 0
    anomaly_segment_ids: list[int] = field(default_factory=list)
    anomaly_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentAggregate:
    project_id: int
    segment_key: str
    name: str
    dimensions: dict[str, str]
    user_count: int
    session_count: int
    page_view_count: int
    product_view_count: int
    add_to_cart_count: int
    checkout_start_count: int
    purchase_count: int
    ad_impression_count: int
    ad_click_count: int
    revenue: Decimal
    view_to_cart_rate: Decimal | None
    cart_to_checkout_rate: Decimal | None
    checkout_to_purchase_rate: Decimal | None
    view_to_purchase_rate: Decimal | None
    ctr: Decimal | None
    cvr: Decimal | None
    target_view_to_purchase_rate: Decimal = Decimal("0.05")

    @property
    def is_valid_sample(self) -> bool:
        return self.product_view_count >= 100 or self.user_count >= 30


@dataclass(frozen=True)
class StoredSegment:
    id: int
    segment_key: str
