from app.anomalies.schemas import FunnelAnomalyRequest
from app.anomalies.service import calculate_funnel_anomalies
from app.metrics.schemas import FunnelMetricFilters


class FakeFunnelMetricsRepository:
    def __init__(self, rows: list[tuple[int, int, int, int]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def fetch_funnel_counts(
        self,
        project_id: str,
        window_start: object,
        window_end: object,
        filters: FunnelMetricFilters | None,
    ) -> tuple[int, int, int, int]:
        if len(self.calls) >= len(self.rows):
            raise AssertionError("FakeFunnelMetricsRepository received more calls than expected")
        self.calls.append(
            {
                "project_id": project_id,
                "window_start": window_start,
                "window_end": window_end,
                "filters": filters,
            }
        )
        return self.rows[len(self.calls) - 1]


def test_calculate_funnel_anomalies_returns_normal_when_current_matches_baseline() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (1000, 200, 100, 50),
            (1000, 200, 100, 50),
        ]
    )
    request = FunnelAnomalyRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
    )

    response = calculate_funnel_anomalies(request, repository)

    assert response.status == "normal"
    assert response.anomalies == []
    assert [evaluation.severity for evaluation in response.evaluations] == [
        "normal",
        "normal",
        "normal",
        "normal",
    ]


def test_calculate_funnel_anomalies_marks_small_denominator_as_insufficient_data() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (1000, 200, 50, 25),
            (1000, 200, 50, 25),
        ]
    )
    request = FunnelAnomalyRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
        min_sample_size=100,
    )

    response = calculate_funnel_anomalies(request, repository)
    evaluation_by_metric = {evaluation.metric: evaluation for evaluation in response.evaluations}

    assert evaluation_by_metric["checkout_to_purchase_rate"].severity == "insufficient_data"
    assert evaluation_by_metric["checkout_to_purchase_rate"].current_denominator == 50
    assert evaluation_by_metric["checkout_to_purchase_rate"].baseline_denominator == 50
    assert response.status == "normal"


def test_calculate_funnel_anomalies_returns_insufficient_data_when_all_metrics_lack_samples() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (10, 0, 0, 0),
            (10, 0, 0, 0),
        ]
    )
    request = FunnelAnomalyRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
        min_sample_size=100,
    )

    response = calculate_funnel_anomalies(request, repository)

    assert response.status == "insufficient_data"
    assert response.anomalies == []
    assert all(evaluation.severity == "insufficient_data" for evaluation in response.evaluations)
