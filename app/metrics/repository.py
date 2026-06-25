from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.db.clickhouse import ClickHouseClient
from app.metrics.schemas import FunnelMetricFilters

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


class FunnelMetricsRepository:
    def __init__(self, client: ClickHouseClient) -> None:
        self.client = client

    def fetch_funnel_counts(
        self,
        project_id: str,
        window_start: datetime,
        window_end: datetime,
        filters: FunnelMetricFilters | None,
    ) -> tuple[int, int, int, int]:
        query, filter_parameters = build_funnel_query(filters)
        parameters = {
            "project_id": project_id,
            "window_start": format_clickhouse_datetime(window_start),
            "window_end": format_clickhouse_datetime(window_end),
            **filter_parameters,
        }

        result = self.client.query(query, parameters=parameters)
        row = result.result_rows[0] if result.result_rows else (0, 0, 0, 0)
        return (int(row[0] or 0), int(row[1] or 0), int(row[2] or 0), int(row[3] or 0))
