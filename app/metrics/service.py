from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.db.clickhouse import ClickHouseClient
from app.metrics.schemas import FunnelMetricFilters, FunnelMetricRequest, FunnelMetricResponse, FunnelMetrics

KST = ZoneInfo("Asia/Seoul")
FILTER_COLUMNS = {
    "channel": "channel",
    "campaign_id": "campaign_id",
    "age_group": "age_group",
    "gender": "gender",
    "device": "device",
    "category": "category",
    "product_id": "product_id",
    "inventory_status": "inventory_status",
}


def calculate_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_dropoff(rate: float | None) -> float | None:
    if rate is None:
        return None
    return 1 - rate


def normalize_to_event_timezone(value: datetime) -> datetime:
    return value.astimezone(KST)


def format_clickhouse_datetime(value: datetime) -> str:
    return normalize_to_event_timezone(value).isoformat(timespec="milliseconds")


def build_segment(filters: FunnelMetricFilters | None) -> dict[str, str | None]:
    values = filters.model_dump() if filters else {}
    return {filter_key: values.get(filter_key) for filter_key in FILTER_COLUMNS}


def build_funnel_query(filters: FunnelMetricFilters | None) -> tuple[str, dict[str, Any]]:
    where_clauses = [
        "project_id = {project_id:String}",
        "event_time >= parseDateTime64BestEffort({window_start:String}, 3, 'Asia/Seoul')",
        "event_time < parseDateTime64BestEffort({window_end:String}, 3, 'Asia/Seoul')",
        "event_name IN ('product_view', 'add_to_cart', 'checkout_start', 'purchase')",
    ]
    parameters: dict[str, Any] = {}

    if filters is not None:
        for filter_key, filter_value in filters.model_dump(exclude_none=True).items():
            column = FILTER_COLUMNS[filter_key]
            parameter_name = f"filter_{filter_key}"
            where_clauses.append(f"{column} = {{{parameter_name}:String}}")
            parameters[parameter_name] = filter_value

    where_sql = "\n      AND ".join(where_clauses)
    query = f"""
WITH session_funnel AS (
    SELECT
        project_id,
        session_id,
        minIf(toNullable(event_time), event_name = 'product_view') AS product_view_time,
        minIf(toNullable(event_time), event_name = 'add_to_cart') AS add_to_cart_time,
        minIf(toNullable(event_time), event_name = 'checkout_start') AS checkout_start_time,
        minIf(toNullable(event_time), event_name = 'purchase') AS purchase_time
    FROM events
    WHERE {where_sql}
    GROUP BY
        project_id,
        session_id
)
SELECT
    countIf(product_view_time IS NOT NULL) AS product_view_sessions,
    countIf(
        product_view_time IS NOT NULL
        AND add_to_cart_time IS NOT NULL
        AND add_to_cart_time > product_view_time
    ) AS add_to_cart_sessions,
    countIf(
        product_view_time IS NOT NULL
        AND add_to_cart_time IS NOT NULL
        AND checkout_start_time IS NOT NULL
        AND add_to_cart_time > product_view_time
        AND checkout_start_time > add_to_cart_time
    ) AS checkout_start_sessions,
    countIf(
        product_view_time IS NOT NULL
        AND add_to_cart_time IS NOT NULL
        AND checkout_start_time IS NOT NULL
        AND purchase_time IS NOT NULL
        AND add_to_cart_time > product_view_time
        AND checkout_start_time > add_to_cart_time
        AND purchase_time > checkout_start_time
    ) AS purchase_sessions
FROM session_funnel
""".strip()
    return query, parameters


def build_metrics(row: tuple[Any, ...]) -> FunnelMetrics:
    product_view_sessions = int(row[0] or 0)
    add_to_cart_sessions = int(row[1] or 0)
    checkout_start_sessions = int(row[2] or 0)
    purchase_sessions = int(row[3] or 0)

    view_to_cart_rate = calculate_rate(add_to_cart_sessions, product_view_sessions)
    cart_to_checkout_rate = calculate_rate(checkout_start_sessions, add_to_cart_sessions)
    checkout_to_purchase_rate = calculate_rate(purchase_sessions, checkout_start_sessions)
    view_to_purchase_rate = calculate_rate(purchase_sessions, product_view_sessions)

    return FunnelMetrics(
        product_view_sessions=product_view_sessions,
        add_to_cart_sessions=add_to_cart_sessions,
        checkout_start_sessions=checkout_start_sessions,
        purchase_sessions=purchase_sessions,
        view_to_cart_rate=view_to_cart_rate,
        cart_to_checkout_rate=cart_to_checkout_rate,
        checkout_to_purchase_rate=checkout_to_purchase_rate,
        view_to_purchase_rate=view_to_purchase_rate,
        view_to_cart_dropoff_rate=calculate_dropoff(view_to_cart_rate),
        cart_to_checkout_dropoff_rate=calculate_dropoff(cart_to_checkout_rate),
        checkout_to_purchase_dropoff_rate=calculate_dropoff(checkout_to_purchase_rate),
    )


def calculate_funnel_metrics(request: FunnelMetricRequest, client: ClickHouseClient) -> FunnelMetricResponse:
    query, filter_parameters = build_funnel_query(request.filters)
    parameters = {
        "project_id": request.project_id,
        "window_start": format_clickhouse_datetime(request.window_start),
        "window_end": format_clickhouse_datetime(request.window_end),
        **filter_parameters,
    }

    result = client.query(query, parameters=parameters)
    row = result.result_rows[0] if result.result_rows else (0, 0, 0, 0)

    return FunnelMetricResponse(
        project_id=request.project_id,
        window_start=normalize_to_event_timezone(request.window_start),
        window_end=normalize_to_event_timezone(request.window_end),
        segment=build_segment(request.filters),
        metrics=build_metrics(row),
    )
