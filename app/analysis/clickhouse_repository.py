from __future__ import annotations

from typing import Any, Protocol
from zoneinfo import ZoneInfo

from app.analysis.metrics import calculate_rate, decimal_or_zero
from app.analysis.models import AnalysisWindow, SegmentAggregate
from app.analysis.segments import (
    build_segment_key,
    build_segment_name,
    is_default_segment_key,
    normalize_dimensions,
)


class ClickHouseQueryResult(Protocol):
    result_rows: list[tuple[Any, ...]]


class ClickHouseClient(Protocol):
    def query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> ClickHouseQueryResult:
        ...


SEGMENT_AGGREGATE_QUERY = """
SELECT
    ifNull(age_group, '') AS age_group,
    ifNull(gender, '') AS gender,
    ifNull(device_type, '') AS device_type,
    if(
        acquisition_channel IS NOT NULL AND acquisition_channel != '',
        acquisition_channel,
        ifNull(utm_source, '')
    ) AS acquisition_channel,
    if(
        primary_category IS NOT NULL AND primary_category != '',
        primary_category,
        ifNull(category, '')
    ) AS primary_category,
    uniqExact(external_user_id) AS user_count,
    uniqExact(session_id) AS session_count,
    countIf(event_name = 'page_view') AS page_view_count,
    countIf(event_name = 'product_view') AS product_view_count,
    countIf(event_name = 'add_to_cart') AS add_to_cart_count,
    countIf(event_name = 'checkout_start') AS checkout_start_count,
    countIf(event_name = 'purchase') AS purchase_count,
    countIf(event_name = 'ad_impression') AS ad_impression_count,
    countIf(event_name = 'ad_click') AS ad_click_count,
    sumIf(ifNull(revenue, 0), event_name = 'purchase') AS revenue
FROM events
WHERE project_id = {project_id:UInt64}
  AND event_time >= parseDateTime64BestEffort({window_start_utc:String}, 3, 'UTC')
  AND event_time < parseDateTime64BestEffort({window_end_utc:String}, 3, 'UTC')
  AND event_name IN (
    'page_view',
    'product_view',
    'add_to_cart',
    'checkout_start',
    'purchase',
    'ad_impression',
    'ad_click'
  )
GROUP BY
    age_group,
    gender,
    device_type,
    acquisition_channel,
    primary_category
HAVING product_view_count >= {min_product_view_count:UInt64}
    OR user_count >= {min_user_count:UInt64}
""".strip()


class ClickHouseAnalysisRepository:
    def __init__(
        self,
        client: ClickHouseClient,
        *,
        min_product_view_count: int = 100,
        min_user_count: int = 30,
    ) -> None:
        self.client = client
        self.min_product_view_count = min_product_view_count
        self.min_user_count = min_user_count

    def fetch_segment_aggregates(
        self,
        project_id: int,
        window: AnalysisWindow,
    ) -> list[SegmentAggregate]:
        parameters = {
            "project_id": project_id,
            "window_start_utc": window.window_start.astimezone(ZoneInfo("UTC")).isoformat(),
            "window_end_utc": window.window_end.astimezone(ZoneInfo("UTC")).isoformat(),
            "min_product_view_count": self.min_product_view_count,
            "min_user_count": self.min_user_count,
        }
        result = self.client.query(SEGMENT_AGGREGATE_QUERY, parameters=parameters)
        aggregates = [
            build_segment_aggregate(project_id=project_id, row=row)
            for row in result.result_rows
        ]
        return [
            aggregate
            for aggregate in aggregates
            if aggregate.is_valid_sample and not is_default_segment_key(aggregate.segment_key)
        ]


def build_segment_aggregate(project_id: int, row: tuple[Any, ...]) -> SegmentAggregate:
    dimensions = normalize_dimensions(
        {
            "age_group": row[0],
            "gender": row[1],
            "device_type": row[2],
            "acquisition_channel": row[3],
            "primary_category": row[4],
        }
    )
    user_count = int(row[5] or 0)
    session_count = int(row[6] or 0)
    page_view_count = int(row[7] or 0)
    product_view_count = int(row[8] or 0)
    add_to_cart_count = int(row[9] or 0)
    checkout_start_count = int(row[10] or 0)
    purchase_count = int(row[11] or 0)
    ad_impression_count = int(row[12] or 0)
    ad_click_count = int(row[13] or 0)
    revenue = decimal_or_zero(row[14])
    return SegmentAggregate(
        project_id=project_id,
        segment_key=build_segment_key(dimensions),
        name=build_segment_name(dimensions),
        dimensions=dimensions,
        user_count=user_count,
        session_count=session_count,
        page_view_count=page_view_count,
        product_view_count=product_view_count,
        add_to_cart_count=add_to_cart_count,
        checkout_start_count=checkout_start_count,
        purchase_count=purchase_count,
        ad_impression_count=ad_impression_count,
        ad_click_count=ad_click_count,
        revenue=revenue,
        view_to_cart_rate=calculate_rate(add_to_cart_count, product_view_count),
        cart_to_checkout_rate=calculate_rate(checkout_start_count, add_to_cart_count),
        checkout_to_purchase_rate=calculate_rate(purchase_count, checkout_start_count),
        view_to_purchase_rate=calculate_rate(purchase_count, product_view_count),
        ctr=calculate_rate(ad_click_count, ad_impression_count),
        cvr=calculate_rate(purchase_count, product_view_count),
    )
