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
    assert [evaluation.severity for evaluation in response.volume_evaluations] == [
        "normal",
        "normal",
        "normal",
        "normal",
    ]
    assert response.primary_anomaly is None
    assert response.summary_message == "No funnel anomaly detected."


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
    assert all(
        evaluation.severity == "insufficient_data"
        for evaluation in response.volume_evaluations
    )
    assert response.summary_message == "Not enough data to determine funnel anomaly."


def test_calculate_funnel_anomalies_detects_volume_anomaly_when_rates_are_normal() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (1000, 200, 100, 40),
            (2000, 400, 200, 80),
        ]
    )
    request = FunnelAnomalyRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
    )

    response = calculate_funnel_anomalies(request, repository)
    volume_by_metric = {evaluation.metric: evaluation for evaluation in response.volume_evaluations}

    assert response.anomalies == []
    assert all(evaluation.severity == "normal" for evaluation in response.evaluations)
    assert volume_by_metric["purchase_sessions"].severity == "critical"
    assert volume_by_metric["purchase_sessions"].relative_drop == 0.5
    assert response.status == "critical"


def test_calculate_funnel_anomalies_uses_volume_critical_for_response_status() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (1000, 100, 50, 20),
            (2000, 200, 100, 40),
        ]
    )
    request = FunnelAnomalyRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
    )

    response = calculate_funnel_anomalies(request, repository)

    assert response.status == "critical"
    assert response.anomalies == []
    assert response.volume_anomalies


def test_calculate_funnel_anomalies_prefers_volume_critical_over_rate_warning() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (1000, 90, 50, 25),
            (2000, 300, 100, 50),
        ]
    )
    request = FunnelAnomalyRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
    )

    response = calculate_funnel_anomalies(request, repository)

    assert response.status == "critical"
    assert response.primary_anomaly is not None
    assert response.primary_anomaly.metric == "add_to_cart_sessions"
    assert response.primary_anomaly.severity == "critical"
    assert response.primary_anomaly.relative_drop == 0.7


def test_calculate_funnel_anomalies_marks_low_volume_baseline_as_insufficient_data() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (1000, 200, 100, 10),
            (1000, 200, 100, 20),
        ]
    )
    request = FunnelAnomalyRequest(
        project_id="loopad-demo-shop",
        window_start="2026-06-24T17:00:00+09:00",
        window_end="2026-06-24T18:00:00+09:00",
        min_volume_count=30,
    )

    response = calculate_funnel_anomalies(request, repository)
    volume_by_metric = {evaluation.metric: evaluation for evaluation in response.volume_evaluations}

    assert volume_by_metric["purchase_sessions"].severity == "insufficient_data"
    assert volume_by_metric["purchase_sessions"].baseline_value == 20
    assert volume_by_metric["purchase_sessions"].relative_drop == 0.5


def test_calculate_funnel_anomalies_keeps_volume_increase_normal() -> None:
    repository = FakeFunnelMetricsRepository(
        rows=[
            (2000, 400, 200, 100),
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
    assert response.volume_anomalies == []
    assert all(evaluation.severity == "normal" for evaluation in response.volume_evaluations)
