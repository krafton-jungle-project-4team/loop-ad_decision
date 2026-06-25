from app.metrics.schemas import FunnelMetricFilters, FunnelMetricRequest
from app.metrics.service import (
    calculate_dropoff,
    calculate_funnel_metrics,
    calculate_rate,
)


class FakeFunnelMetricsRepository:
    def __init__(self) -> None:
        self.project_id: str | None = None
        self.filters: FunnelMetricFilters | None = None

    def fetch_funnel_counts(
        self,
        project_id: str,
        window_start: object,
        window_end: object,
        filters: FunnelMetricFilters | None,
    ) -> tuple[int, int, int, int]:
        self.project_id = project_id
        self.filters = filters
        return (1000, 90, 50, 25)


def test_calculate_rate_returns_none_when_denominator_is_zero() -> None:
    assert calculate_rate(1, 0) is None


def test_calculate_rate_returns_float() -> None:
    assert calculate_rate(90, 1000) == 0.09


def test_calculate_dropoff_returns_none_when_rate_is_none() -> None:
    assert calculate_dropoff(None) is None


def test_calculate_dropoff_returns_inverse_rate() -> None:
    assert calculate_dropoff(0.09) == 0.91


def test_calculate_funnel_metrics_builds_response_from_repository_counts() -> None:
    repository = FakeFunnelMetricsRepository()
    request = FunnelMetricRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T08:00:00+00:00",
        window_end="2026-06-24T09:00:00+00:00",
        filters=FunnelMetricFilters(channel="kakao"),
    )

    response = calculate_funnel_metrics(request, repository)

    assert repository.project_id == "loopad-demo-shop"
    assert repository.filters == request.filters
    assert response.window_start.isoformat() == "2026-06-24T17:00:00+09:00"
    assert response.window_end.isoformat() == "2026-06-24T18:00:00+09:00"
    assert response.segment["channel"] == "kakao"
    assert response.metrics.product_view_sessions == 1000
    assert response.metrics.view_to_purchase_rate == 0.025
