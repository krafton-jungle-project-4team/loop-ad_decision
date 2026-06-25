import pytest

from app.metrics.schemas import FunnelMetricFilters
from app.root_causes.repository import GroupedFunnelCountsRow
from app.root_causes.schemas import RootCauseAnalysisRequest
from app.root_causes.service import calculate_root_causes, resolve_candidate_dimensions


class FakeRootCauseRepository:
    def __init__(
        self,
        funnel_rows: list[tuple[int, int, int, int]],
        grouped_rows_by_dimension: dict[str, list[GroupedFunnelCountsRow]] | None = None,
    ) -> None:
        self.funnel_rows = funnel_rows
        self.grouped_rows_by_dimension = grouped_rows_by_dimension or {}
        self.funnel_calls: list[dict[str, object]] = []
        self.grouped_calls: list[dict[str, object]] = []

    def fetch_funnel_counts(
        self,
        project_id: str,
        window_start: object,
        window_end: object,
        filters: FunnelMetricFilters | None,
    ) -> tuple[int, int, int, int]:
        if len(self.funnel_calls) >= len(self.funnel_rows):
            raise AssertionError("FakeRootCauseRepository received more funnel calls than expected")
        self.funnel_calls.append(
            {
                "project_id": project_id,
                "window_start": window_start,
                "window_end": window_end,
                "filters": filters,
            }
        )
        return self.funnel_rows[len(self.funnel_calls) - 1]

    def fetch_grouped_funnel_counts(
        self,
        project_id: str,
        window_start: object,
        window_end: object,
        baseline_start: object,
        baseline_end: object,
        filters: FunnelMetricFilters | None,
        dimension: str,
        limit: int,
    ) -> list[GroupedFunnelCountsRow]:
        self.grouped_calls.append(
            {
                "project_id": project_id,
                "window_start": window_start,
                "window_end": window_end,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "filters": filters,
                "dimension": dimension,
                "limit": limit,
            }
        )
        return self.grouped_rows_by_dimension.get(dimension, [])


def grouped_row(
    value: str,
    current: tuple[int, int, int, int],
    baseline: tuple[int, int, int, int],
) -> GroupedFunnelCountsRow:
    return GroupedFunnelCountsRow(
        dimension_value=value,
        current_product_view_sessions=current[0],
        current_add_to_cart_sessions=current[1],
        current_checkout_start_sessions=current[2],
        current_purchase_sessions=current[3],
        baseline_product_view_sessions=baseline[0],
        baseline_add_to_cart_sessions=baseline[1],
        baseline_checkout_start_sessions=baseline[2],
        baseline_purchase_sessions=baseline[3],
    )


def root_cause_request(**overrides: object) -> RootCauseAnalysisRequest:
    values = {
        "project_id": "loopad-demo-shop",
        "window_start": "2026-06-24T17:00:00+09:00",
        "window_end": "2026-06-24T18:00:00+09:00",
        "include_volume_anomalies": False,
    }
    values.update(overrides)
    return RootCauseAnalysisRequest(**values)


def test_calculate_root_causes_returns_empty_candidates_without_primary_anomaly() -> None:
    repository = FakeRootCauseRepository(
        funnel_rows=[
            (1000, 200, 100, 50),
            (1000, 200, 100, 50),
        ]
    )
    request = root_cause_request()

    response = calculate_root_causes(request, repository)

    assert response.status == "normal"
    assert response.target_anomaly is None
    assert response.candidates == []
    assert response.total_candidates_evaluated == 0
    assert repository.grouped_calls == []


def test_calculate_root_causes_ranks_inventory_status_critical_first() -> None:
    repository = FakeRootCauseRepository(
        funnel_rows=[
            (1000, 90, 50, 25),
            (1000, 200, 50, 25),
        ],
        grouped_rows_by_dimension={
            "inventory_status": [
                grouped_row("out_of_stock", (300, 15, 5, 2), (300, 150, 20, 10)),
                grouped_row("in_stock", (700, 75, 45, 23), (700, 50, 30, 15)),
            ]
        },
    )
    request = root_cause_request(candidate_dimensions=["inventory_status"])

    response = calculate_root_causes(request, repository)

    assert response.status == "critical"
    assert response.target_anomaly is not None
    assert response.target_anomaly.metric == "view_to_cart_rate"
    assert len(response.candidates) == 1
    candidate = response.candidates[0]
    assert candidate.rank == 1
    assert candidate.dimension == "inventory_status"
    assert candidate.value == "out_of_stock"
    assert candidate.cause_type == "inventory_issue"
    assert candidate.severity == "critical"


def test_calculate_root_causes_calculates_product_drop_and_excess_lost_sessions() -> None:
    repository = FakeRootCauseRepository(
        funnel_rows=[
            (1000, 90, 50, 25),
            (1000, 200, 50, 25),
        ],
        grouped_rows_by_dimension={
            "product_id": [
                grouped_row("sku-1", (200, 20, 8, 4), (200, 80, 20, 10)),
            ]
        },
    )
    request = root_cause_request(candidate_dimensions=["product_id"])

    response = calculate_root_causes(request, repository)

    candidate = response.candidates[0]
    assert candidate.dimension == "product_id"
    assert candidate.metric == "view_to_cart_rate"
    assert candidate.current_value == pytest.approx(0.10)
    assert candidate.baseline_value == pytest.approx(0.40)
    assert candidate.drop_point == pytest.approx(0.30)
    assert candidate.relative_drop == pytest.approx(0.75)
    assert candidate.support_share == pytest.approx(0.20)
    assert candidate.excess_lost_sessions == pytest.approx(60.0)


def test_calculate_root_causes_excludes_filtered_dimensions_from_default_candidates() -> None:
    repository = FakeRootCauseRepository(
        funnel_rows=[
            (1000, 90, 50, 25),
            (1000, 200, 50, 25),
        ]
    )
    request = root_cause_request(filters=FunnelMetricFilters(channel="kakao"))

    response = calculate_root_causes(request, repository)

    called_dimensions = [call["dimension"] for call in repository.grouped_calls]
    assert "channel" not in called_dimensions
    assert "campaign_id" in called_dimensions
    assert response.candidates == []


def test_resolve_candidate_dimensions_rejects_explicit_filtered_dimension() -> None:
    request = RootCauseAnalysisRequest.model_construct(
        filters=FunnelMetricFilters(channel="kakao"),
        candidate_dimensions=["channel"],
    )

    with pytest.raises(ValueError):
        resolve_candidate_dimensions(request)


def test_calculate_root_causes_evaluates_volume_anomaly_candidates() -> None:
    repository = FakeRootCauseRepository(
        funnel_rows=[
            (1000, 100, 50, 20),
            (2000, 200, 100, 40),
        ],
        grouped_rows_by_dimension={
            "channel": [
                grouped_row("paid", (300, 30, 15, 6), (900, 90, 45, 18)),
                grouped_row("organic", (700, 70, 35, 14), (1100, 110, 55, 22)),
            ]
        },
    )
    request = root_cause_request(
        include_volume_anomalies=True,
        candidate_dimensions=["channel"],
    )

    response = calculate_root_causes(request, repository)

    assert response.target_anomaly is not None
    assert response.target_anomaly.metric == "product_view_sessions"
    candidate = response.candidates[0]
    assert candidate.metric == "product_view_sessions"
    assert candidate.funnel_step is None
    assert candidate.cause_type == "traffic_volume_drop"
    assert candidate.current_value == 300
    assert candidate.baseline_value == 900
    assert candidate.relative_drop == pytest.approx(2 / 3)
    assert candidate.support_share == pytest.approx(0.45)
    assert candidate.excess_lost_sessions == pytest.approx(600.0)


def test_calculate_root_causes_excludes_candidates_below_sample_size() -> None:
    repository = FakeRootCauseRepository(
        funnel_rows=[
            (1000, 90, 50, 25),
            (1000, 200, 50, 25),
        ],
        grouped_rows_by_dimension={
            "product_id": [
                grouped_row("small-sku", (50, 1, 0, 0), (50, 25, 2, 1)),
            ]
        },
    )
    request = root_cause_request(candidate_dimensions=["product_id"], min_sample_size=100)

    response = calculate_root_causes(request, repository)

    assert response.total_candidates_evaluated == 1
    assert response.candidates == []


def test_calculate_root_causes_sorts_critical_candidates_before_warning() -> None:
    repository = FakeRootCauseRepository(
        funnel_rows=[
            (1000, 90, 50, 25),
            (1000, 200, 50, 25),
        ],
        grouped_rows_by_dimension={
            "product_id": [
                grouped_row("warning-sku", (900, 180, 50, 25), (900, 270, 50, 25)),
                grouped_row("critical-sku", (100, 10, 0, 0), (100, 40, 0, 0)),
            ]
        },
    )
    request = root_cause_request(
        candidate_dimensions=["product_id"],
        critical_abs_drop=0.20,
    )

    response = calculate_root_causes(request, repository)

    assert [candidate.value for candidate in response.candidates] == [
        "critical-sku",
        "warning-sku",
    ]
    assert [candidate.severity for candidate in response.candidates] == [
        "critical",
        "warning",
    ]
