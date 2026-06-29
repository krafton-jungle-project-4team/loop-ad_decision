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


@dataclass(frozen=True)
class UserPrimarySegmentCandidate:
    external_user_id: str
    segment_key: str
    dimensions: dict[str, str]
    confidence: Decimal = Decimal("1.0")


@dataclass(frozen=True)
class BaselineMetrics:
    segment_id: int
    view_to_purchase_rate: Decimal | None


@dataclass(frozen=True)
class SegmentAnomalyCandidate:
    segment_id: int
    metric_name: str
    actual_value: Decimal | None
    expected_value: Decimal | None
    target_value: Decimal
    difference_value: Decimal | None
    difference_rate: Decimal | None
    severity: str
    impact_score: Decimal
    evidence_json: dict[str, object]


@dataclass(frozen=True)
class StoredAnomaly:
    id: int
    segment_id: int


@dataclass(frozen=True)
class RootCauseCandidate:
    anomaly_id: int
    cause_type: str
    cause_key: str
    title: str
    description: str
    confidence_score: Decimal
    impact_score: Decimal
    rank_no: int
    evidence_json: dict[str, object]
