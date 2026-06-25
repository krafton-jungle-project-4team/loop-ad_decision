from typing import Any

from app.metrics.repository import FunnelMetricsRepository, build_segment, normalize_to_event_timezone
from app.metrics.schemas import FunnelMetricRequest, FunnelMetricResponse, FunnelMetrics


def calculate_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_dropoff(rate: float | None) -> float | None:
    if rate is None:
        return None
    return 1 - rate


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


def calculate_funnel_metrics(
    request: FunnelMetricRequest,
    repository: FunnelMetricsRepository,
) -> FunnelMetricResponse:
    counts = repository.fetch_funnel_counts(
        project_id=request.project_id,
        window_start=request.window_start,
        window_end=request.window_end,
        filters=request.filters,
    )
    return FunnelMetricResponse(
        project_id=request.project_id,
        window_start=normalize_to_event_timezone(request.window_start),
        window_end=normalize_to_event_timezone(request.window_end),
        segment=build_segment(request.filters),
        metrics=build_metrics(counts),
    )
