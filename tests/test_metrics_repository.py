from datetime import datetime
from zoneinfo import ZoneInfo

from app.metrics.repository import FunnelMetricsRepository, build_funnel_query
from app.metrics.schemas import FunnelMetricFilters


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


def test_fetch_funnel_counts_normalizes_window_to_kst() -> None:
    client = FakeClickHouseClient()
    repository = FunnelMetricsRepository(client)

    counts = repository.fetch_funnel_counts(
        project_id="loopad-demo-shop",
        window_start=datetime(2026, 6, 24, 8, 0, tzinfo=ZoneInfo("UTC")),
        window_end=datetime(2026, 6, 24, 9, 0, tzinfo=ZoneInfo("UTC")),
        filters=FunnelMetricFilters(channel="kakao"),
    )

    assert counts == (1000, 90, 50, 25)
    assert client.parameters is not None
    assert client.parameters["window_start"] == "2026-06-24T17:00:00.000+09:00"
    assert client.parameters["window_end"] == "2026-06-24T18:00:00.000+09:00"
    assert client.parameters["filter_channel"] == "kakao"
