from datetime import datetime
from zoneinfo import ZoneInfo

from app.metrics.schemas import FunnelMetricFilters, FunnelMetricRequest
from app.metrics.service import (
    build_funnel_query,
    calculate_dropoff,
    calculate_funnel_metrics,
    calculate_rate,
)


class FakeResult:
    result_rows = [(1000, 90, 50, 25)]


class FakeClickHouseClient:
    def __init__(self) -> None:
        self.query_text: str | None = None
        self.parameters: dict[str, object] | None = None

    def query(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        self.query_text = query
        self.parameters = parameters
        return FakeResult()


def test_calculate_rate_returns_none_when_denominator_is_zero() -> None:
    assert calculate_rate(1, 0) is None


def test_calculate_rate_returns_float() -> None:
    assert calculate_rate(90, 1000) == 0.09


def test_calculate_dropoff_returns_none_when_rate_is_none() -> None:
    assert calculate_dropoff(None) is None


def test_calculate_dropoff_returns_inverse_rate() -> None:
    assert calculate_dropoff(0.09) == 0.91


def test_build_funnel_query_only_adds_non_null_whitelisted_filters() -> None:
    query, parameters = build_funnel_query(
        FunnelMetricFilters(
            channel="kakao",
            campaign_id=None,
            category="fresh_food",
            product_id=None,
        )
    )

    assert "channel = {filter_channel:String}" in query
    assert "category = {filter_category:String}" in query
    assert "campaign_id = {filter_campaign_id:String}" not in query
    assert "product_id = {filter_product_id:String}" not in query
    assert "event_name IN ('product_view', 'add_to_cart', 'checkout_start', 'purchase')" in query
    assert parameters == {"filter_channel": "kakao", "filter_category": "fresh_food"}


def test_calculate_funnel_metrics_normalizes_window_to_kst() -> None:
    client = FakeClickHouseClient()
    request = FunnelMetricRequest(
        project_id="loopad-demo-shop",
        window_start=datetime(2026, 6, 24, 8, 0, tzinfo=ZoneInfo("UTC")),
        window_end=datetime(2026, 6, 24, 9, 0, tzinfo=ZoneInfo("UTC")),
        filters=FunnelMetricFilters(channel="kakao"),
    )

    response = calculate_funnel_metrics(request, client)

    assert client.parameters is not None
    assert client.parameters["window_start"] == "2026-06-24T17:00:00.000+09:00"
    assert client.parameters["window_end"] == "2026-06-24T18:00:00.000+09:00"
    assert response.window_start.isoformat() == "2026-06-24T17:00:00+09:00"
    assert response.window_end.isoformat() == "2026-06-24T18:00:00+09:00"
    assert response.segment["channel"] == "kakao"
    assert response.metrics.product_view_sessions == 1000
    assert response.metrics.view_to_purchase_rate == 0.025
