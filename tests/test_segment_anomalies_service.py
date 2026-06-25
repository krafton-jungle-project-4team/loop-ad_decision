from datetime import datetime
from zoneinfo import ZoneInfo

from app.anomalies.schemas import SegmentFunnelAnomalyRequest
from app.anomalies.service import calculate_segment_funnel_anomalies
from app.metrics.schemas import FunnelMetricFilters


class FakeFunnelMetricsRepository:
    def __init__(
        self,
        candidates: list[dict[str, str]],
        counts_by_segment: dict[tuple[tuple[str, str], ...], list[tuple[int, int, int, int]]],
    ) -> None:
        self.candidates = candidates
        self.counts_by_segment = counts_by_segment
        self.segment_calls: list[dict[str, object]] = []
        self.count_calls: list[dict[str, object]] = []

    def fetch_segment_values(
        self,
        project_id: str,
        window_start: object,
        window_end: object,
        base_filters: FunnelMetricFilters | None,
        segment_by: list[str],
        limit: int,
    ) -> list[dict[str, str]]:
        self.segment_calls.append(
            {
                "project_id": project_id,
                "window_start": window_start,
                "window_end": window_end,
                "base_filters": base_filters,
                "segment_by": segment_by,
                "limit": limit,
            }
        )
        return self.candidates

    def fetch_funnel_counts(
        self,
        project_id: str,
        window_start: object,
        window_end: object,
        filters: FunnelMetricFilters | None,
    ) -> tuple[int, int, int, int]:
        key = build_filter_key(filters)
        rows = self.counts_by_segment[key]
        call_count = sum(1 for call in self.count_calls if call["key"] == key)
        self.count_calls.append(
            {
                "key": key,
                "project_id": project_id,
                "window_start": window_start,
                "window_end": window_end,
                "filters": filters,
            }
        )
        return rows[call_count]


def build_filter_key(filters: FunnelMetricFilters | None) -> tuple[tuple[str, str], ...]:
    values = filters.model_dump(exclude_none=True) if filters else {}
    return tuple(sorted(values.items()))


def make_request(**overrides: object) -> SegmentFunnelAnomalyRequest:
    values = {
        "project_id": "loopad-demo-shop",
        "window_start": "2026-06-24T17:00:00+09:00",
        "window_end": "2026-06-24T18:00:00+09:00",
        "segment_by": ["channel"],
    }
    values.update(overrides)
    return SegmentFunnelAnomalyRequest(**values)


def test_calculate_segment_funnel_anomalies_returns_only_warning_and_critical_segments() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[
            {"channel": "kakao"},
            {"channel": "naver"},
            {"channel": "meta"},
        ],
        counts_by_segment={
            (("channel", "kakao"),): [(1000, 90, 50, 25), (1000, 150, 50, 25)],
            (("channel", "naver"),): [(1000, 200, 100, 50), (1000, 200, 100, 50)],
            (("channel", "meta"),): [(10, 0, 0, 0), (10, 0, 0, 0)],
        },
    )

    response = calculate_segment_funnel_anomalies(make_request(), repository)

    assert response.status == "warning"
    assert response.total_segments_discovered == 3
    assert response.total_segments_evaluated == 3
    assert [segment.segment["channel"] for segment in response.segments] == ["kakao"]
    assert response.segments[0].status == "warning"


def test_calculate_segment_funnel_anomalies_returns_volume_only_anomaly() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[{"channel": "kakao"}],
        counts_by_segment={
            (("channel", "kakao"),): [(1000, 200, 100, 40), (2000, 400, 200, 80)],
        },
    )

    response = calculate_segment_funnel_anomalies(make_request(), repository)

    assert response.status == "critical"
    assert len(response.segments) == 1
    assert response.segments[0].anomalies == []
    assert response.segments[0].volume_anomalies
    assert response.segments[0].primary_anomaly is not None
    assert response.segments[0].primary_anomaly.metric == "product_view_sessions"


def test_calculate_segment_funnel_anomalies_can_disable_volume_only_anomalies() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[{"channel": "kakao"}],
        counts_by_segment={
            (("channel", "kakao"),): [(1000, 200, 100, 40), (2000, 400, 200, 80)],
        },
    )

    response = calculate_segment_funnel_anomalies(
        make_request(include_volume_anomalies=False),
        repository,
    )

    assert response.status == "normal"
    assert response.segments == []


def test_calculate_segment_funnel_anomalies_sorts_critical_before_warning() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[
            {"channel": "warning"},
            {"channel": "critical"},
        ],
        counts_by_segment={
            (("channel", "warning"),): [(1000, 90, 50, 25), (1000, 150, 50, 25)],
            (("channel", "critical"),): [(1000, 200, 100, 40), (2000, 400, 200, 80)],
        },
    )

    response = calculate_segment_funnel_anomalies(make_request(), repository)

    assert [segment.segment["channel"] for segment in response.segments] == [
        "critical",
        "warning",
    ]


def test_calculate_segment_funnel_anomalies_sorts_warning_by_relative_drop() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[
            {"channel": "smaller"},
            {"channel": "larger"},
        ],
        counts_by_segment={
            (("channel", "smaller"),): [(1000, 70, 50, 25), (1000, 100, 50, 25)],
            (("channel", "larger"),): [(1000, 60, 50, 25), (1000, 100, 50, 25)],
        },
    )

    response = calculate_segment_funnel_anomalies(
        make_request(
            include_volume_anomalies=False,
            critical_abs_drop=0.90,
            critical_relative_drop=0.90,
        ),
        repository,
    )

    assert [segment.segment["channel"] for segment in response.segments] == [
        "larger",
        "smaller",
    ]


def test_calculate_segment_funnel_anomalies_returns_insufficient_data_without_candidates() -> None:
    repository = FakeFunnelMetricsRepository(candidates=[], counts_by_segment={})

    response = calculate_segment_funnel_anomalies(make_request(), repository)

    assert response.status == "insufficient_data"
    assert response.segments == []
    assert response.summary_message == "Not enough segment data to determine funnel anomalies."


def test_calculate_segment_funnel_anomalies_returns_normal_when_all_candidates_are_normal() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[{"channel": "kakao"}],
        counts_by_segment={
            (("channel", "kakao"),): [(1000, 200, 100, 50), (1000, 200, 100, 50)],
        },
    )

    response = calculate_segment_funnel_anomalies(make_request(), repository)

    assert response.status == "normal"
    assert response.segments == []
    assert response.summary_message == "No segment funnel anomaly detected."


def test_calculate_segment_funnel_anomalies_returns_insufficient_data_when_all_candidates_are_insufficient() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[{"channel": "kakao"}],
        counts_by_segment={
            (("channel", "kakao"),): [(10, 0, 0, 0), (10, 0, 0, 0)],
        },
    )

    response = calculate_segment_funnel_anomalies(make_request(), repository)

    assert response.status == "insufficient_data"
    assert response.segments == []


def test_calculate_segment_funnel_anomalies_merges_base_filters_with_segment_values() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[{"category": "fresh_food"}],
        counts_by_segment={
            (("category", "fresh_food"), ("channel", "kakao")): [
                (1000, 200, 100, 50),
                (1000, 200, 100, 50),
            ],
        },
    )

    response = calculate_segment_funnel_anomalies(
        make_request(
            base_filters=FunnelMetricFilters(channel="kakao"),
            segment_by=["category"],
        ),
        repository,
    )

    assert repository.segment_calls[0]["base_filters"] == FunnelMetricFilters(channel="kakao")
    assert response.base_segment["channel"] == "kakao"
    assert repository.count_calls[0]["filters"] == FunnelMetricFilters(
        channel="kakao",
        category="fresh_food",
    )


def test_calculate_segment_funnel_anomalies_calls_current_then_baseline_for_each_segment() -> None:
    repository = FakeFunnelMetricsRepository(
        candidates=[
            {"channel": "kakao"},
            {"channel": "naver"},
        ],
        counts_by_segment={
            (("channel", "kakao"),): [(1000, 200, 100, 50), (1000, 200, 100, 50)],
            (("channel", "naver"),): [(1000, 200, 100, 50), (1000, 200, 100, 50)],
        },
    )

    calculate_segment_funnel_anomalies(make_request(), repository)

    assert len(repository.count_calls) == 4
    assert repository.count_calls[0]["window_start"] == datetime(
        2026,
        6,
        24,
        17,
        0,
        tzinfo=ZoneInfo("Asia/Seoul"),
    )
    assert repository.count_calls[1]["window_start"] == datetime(
        2026,
        6,
        24,
        16,
        0,
        tzinfo=ZoneInfo("Asia/Seoul"),
    )
    assert repository.count_calls[2]["window_start"] == datetime(
        2026,
        6,
        24,
        17,
        0,
        tzinfo=ZoneInfo("Asia/Seoul"),
    )
    assert repository.count_calls[3]["window_start"] == datetime(
        2026,
        6,
        24,
        16,
        0,
        tzinfo=ZoneInfo("Asia/Seoul"),
    )
