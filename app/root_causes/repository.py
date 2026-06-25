from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.db.clickhouse import ClickHouseClient
from app.metrics.repository import (
    FILTER_COLUMNS,
    FunnelMetricsRepository,
    format_clickhouse_datetime,
)
from app.metrics.schemas import FunnelMetricFilters


@dataclass(frozen=True)
class GroupedFunnelCountsRow:
    dimension_value: str
    current_product_view_sessions: int
    current_add_to_cart_sessions: int
    current_checkout_start_sessions: int
    current_purchase_sessions: int
    baseline_product_view_sessions: int
    baseline_add_to_cart_sessions: int
    baseline_checkout_start_sessions: int
    baseline_purchase_sessions: int


def build_grouped_funnel_counts_query(
    filters: FunnelMetricFilters | None,
    dimension: str,
) -> tuple[str, dict[str, Any]]:
    if dimension not in FILTER_COLUMNS:
        raise ValueError("dimension must be allowlisted")

    dimension_column = FILTER_COLUMNS[dimension]
    current_where_sql, parameters = build_grouped_window_where(
        filters=filters,
        dimension_column=dimension_column,
        start_parameter="current_window_start",
        end_parameter="current_window_end",
    )
    baseline_where_sql, _ = build_grouped_window_where(
        filters=filters,
        dimension_column=dimension_column,
        start_parameter="baseline_start",
        end_parameter="baseline_end",
    )
    query = f"""
WITH current_session_funnel AS (
    SELECT
        session_id,
        {dimension_column} AS dimension_value,
        minIf(toNullable(event_time), event_name = 'product_view') AS product_view_time,
        minIf(toNullable(event_time), event_name = 'add_to_cart') AS add_to_cart_time,
        minIf(toNullable(event_time), event_name = 'checkout_start') AS checkout_start_time,
        minIf(toNullable(event_time), event_name = 'purchase') AS purchase_time
    FROM events
    WHERE {current_where_sql}
    GROUP BY session_id, dimension_value
),
current_counts AS (
    SELECT
        dimension_value,
        countIf(product_view_time IS NOT NULL) AS current_product_view_sessions,
        countIf(
            product_view_time IS NOT NULL
            AND add_to_cart_time IS NOT NULL
            AND add_to_cart_time > product_view_time
        ) AS current_add_to_cart_sessions,
        countIf(
            product_view_time IS NOT NULL
            AND add_to_cart_time IS NOT NULL
            AND checkout_start_time IS NOT NULL
            AND add_to_cart_time > product_view_time
            AND checkout_start_time > add_to_cart_time
        ) AS current_checkout_start_sessions,
        countIf(
            product_view_time IS NOT NULL
            AND add_to_cart_time IS NOT NULL
            AND checkout_start_time IS NOT NULL
            AND purchase_time IS NOT NULL
            AND add_to_cart_time > product_view_time
            AND checkout_start_time > add_to_cart_time
            AND purchase_time > checkout_start_time
        ) AS current_purchase_sessions
    FROM current_session_funnel
    GROUP BY dimension_value
),
baseline_session_funnel AS (
    SELECT
        session_id,
        {dimension_column} AS dimension_value,
        minIf(toNullable(event_time), event_name = 'product_view') AS product_view_time,
        minIf(toNullable(event_time), event_name = 'add_to_cart') AS add_to_cart_time,
        minIf(toNullable(event_time), event_name = 'checkout_start') AS checkout_start_time,
        minIf(toNullable(event_time), event_name = 'purchase') AS purchase_time
    FROM events
    WHERE {baseline_where_sql}
    GROUP BY session_id, dimension_value
),
baseline_counts AS (
    SELECT
        dimension_value,
        countIf(product_view_time IS NOT NULL) AS baseline_product_view_sessions,
        countIf(
            product_view_time IS NOT NULL
            AND add_to_cart_time IS NOT NULL
            AND add_to_cart_time > product_view_time
        ) AS baseline_add_to_cart_sessions,
        countIf(
            product_view_time IS NOT NULL
            AND add_to_cart_time IS NOT NULL
            AND checkout_start_time IS NOT NULL
            AND add_to_cart_time > product_view_time
            AND checkout_start_time > add_to_cart_time
        ) AS baseline_checkout_start_sessions,
        countIf(
            product_view_time IS NOT NULL
            AND add_to_cart_time IS NOT NULL
            AND checkout_start_time IS NOT NULL
            AND purchase_time IS NOT NULL
            AND add_to_cart_time > product_view_time
            AND checkout_start_time > add_to_cart_time
            AND purchase_time > checkout_start_time
        ) AS baseline_purchase_sessions
    FROM baseline_session_funnel
    GROUP BY dimension_value
)
SELECT
    coalesce(current_counts.dimension_value, baseline_counts.dimension_value) AS dimension_value,
    ifNull(current_product_view_sessions, 0) AS current_product_view_sessions,
    ifNull(current_add_to_cart_sessions, 0) AS current_add_to_cart_sessions,
    ifNull(current_checkout_start_sessions, 0) AS current_checkout_start_sessions,
    ifNull(current_purchase_sessions, 0) AS current_purchase_sessions,
    ifNull(baseline_product_view_sessions, 0) AS baseline_product_view_sessions,
    ifNull(baseline_add_to_cart_sessions, 0) AS baseline_add_to_cart_sessions,
    ifNull(baseline_checkout_start_sessions, 0) AS baseline_checkout_start_sessions,
    ifNull(baseline_purchase_sessions, 0) AS baseline_purchase_sessions
FROM current_counts
FULL OUTER JOIN baseline_counts
    ON current_counts.dimension_value = baseline_counts.dimension_value
ORDER BY greatest(
    ifNull(current_product_view_sessions, 0),
    ifNull(baseline_product_view_sessions, 0)
) DESC
LIMIT {{candidate_limit:UInt32}}
""".strip()
    return query, parameters


def build_grouped_window_where(
    *,
    filters: FunnelMetricFilters | None,
    dimension_column: str,
    start_parameter: str,
    end_parameter: str,
) -> tuple[str, dict[str, Any]]:
    where_clauses = [
        "project_id = {project_id:String}",
        f"event_time >= parseDateTime64BestEffort({{{start_parameter}:String}}, 3, 'Asia/Seoul')",
        f"event_time < parseDateTime64BestEffort({{{end_parameter}:String}}, 3, 'Asia/Seoul')",
        "event_name IN ('product_view', 'add_to_cart', 'checkout_start', 'purchase')",
        f"{dimension_column} IS NOT NULL",
        f"{dimension_column} != ''",
    ]
    parameters: dict[str, Any] = {}

    if filters is not None:
        for filter_key, filter_value in filters.model_dump(exclude_none=True).items():
            column = FILTER_COLUMNS[filter_key]
            parameter_name = f"filter_{filter_key}"
            where_clauses.append(f"{column} = {{{parameter_name}:String}}")
            parameters[parameter_name] = filter_value

    where_sql = "\n      AND ".join(where_clauses)
    return where_sql, parameters


class RootCauseRepository:
    def __init__(self, client: ClickHouseClient) -> None:
        self.client = client
        self.metrics_repository = FunnelMetricsRepository(client)

    def fetch_funnel_counts(
        self,
        project_id: str,
        window_start: datetime,
        window_end: datetime,
        filters: FunnelMetricFilters | None,
    ) -> tuple[int, int, int, int]:
        return self.metrics_repository.fetch_funnel_counts(
            project_id=project_id,
            window_start=window_start,
            window_end=window_end,
            filters=filters,
        )

    def fetch_grouped_funnel_counts(
        self,
        project_id: str,
        window_start: datetime,
        window_end: datetime,
        baseline_start: datetime,
        baseline_end: datetime,
        filters: FunnelMetricFilters | None,
        dimension: str,
        limit: int,
    ) -> list[GroupedFunnelCountsRow]:
        query, filter_parameters = build_grouped_funnel_counts_query(filters, dimension)
        parameters = {
            "project_id": project_id,
            "current_window_start": format_clickhouse_datetime(window_start),
            "current_window_end": format_clickhouse_datetime(window_end),
            "baseline_start": format_clickhouse_datetime(baseline_start),
            "baseline_end": format_clickhouse_datetime(baseline_end),
            "candidate_limit": limit,
            **filter_parameters,
        }

        result = self.client.query(query, parameters=parameters)
        return [
            GroupedFunnelCountsRow(
                dimension_value=str(row[0]),
                current_product_view_sessions=int(row[1] or 0),
                current_add_to_cart_sessions=int(row[2] or 0),
                current_checkout_start_sessions=int(row[3] or 0),
                current_purchase_sessions=int(row[4] or 0),
                baseline_product_view_sessions=int(row[5] or 0),
                baseline_add_to_cart_sessions=int(row[6] or 0),
                baseline_checkout_start_sessions=int(row[7] or 0),
                baseline_purchase_sessions=int(row[8] or 0),
            )
            for row in result.result_rows
            if row and row[0] is not None and str(row[0]) != ""
        ]
