from __future__ import annotations

import pytest

from app.decision.assignment_selector import (
    ADAPTIVE_MATCHER_STRATEGY,
    ANN_MATCHER_STRATEGY,
    EXACT_MATCHER_STRATEGY,
    MATCHER_SELECTED_EXACT_REASON,
    AssignmentMatcherSelector,
    aggregate_matcher_strategy,
)


def test_production_selector_is_exact_only_for_large_known_workload() -> None:
    selection = AssignmentMatcherSelector().select(
        user_count=10_000,
        segment_count=5_000,
        dimension=64,
        page_size=10_000,
        backend="pgvector",
    )

    assert selection["matcher_strategy"] == EXACT_MATCHER_STRATEGY
    assert selection["use_ann"] is False
    assert selection["workload_size"] == 10_000 * 5_000 * 64
    assert selection["ann_not_applied_reason"] == MATCHER_SELECTED_EXACT_REASON


def test_injected_evidence_region_selects_ann_at_inclusive_boundaries() -> None:
    selector = AssignmentMatcherSelector(
        approved_ann_regions=(evidence_region(),),
        policy_version="test_pgvector_evidence_v1",
    )

    minimum = selector.select(
        user_count=100,
        segment_count=20,
        dimension=64,
        page_size=1_000,
        backend=" PGVECTOR ",
    )
    maximum = selector.select(
        user_count=1_000,
        segment_count=100,
        dimension=64,
        page_size=10_000,
        backend="pgvector",
    )

    assert minimum == {
        "policy_version": "test_pgvector_evidence_v1",
        "backend": "pgvector",
        "matcher_strategy": ANN_MATCHER_STRATEGY,
        "use_ann": True,
        "workload_size": 100 * 20 * 64,
        "ann_not_applied_reason": None,
    }
    assert maximum["matcher_strategy"] == ANN_MATCHER_STRATEGY
    assert maximum["workload_size"] == 1_000 * 100 * 64


@pytest.mark.parametrize(
    ("override", "expected_workload_size"),
    [
        ({"user_count": 99}, 99 * 20 * 64),
        ({"segment_count": 19}, 100 * 19 * 64),
        ({"dimension": 63}, 100 * 20 * 63),
        ({"page_size": 999}, 100 * 20 * 64),
        ({"backend": "faiss"}, 100 * 20 * 64),
    ],
)
def test_outside_any_evidence_axis_selects_exact(
    override: dict[str, int | str],
    expected_workload_size: int,
) -> None:
    selector = AssignmentMatcherSelector(
        approved_ann_regions=(evidence_region(),)
    )
    workload: dict[str, int | str | None] = {
        "user_count": 100,
        "segment_count": 20,
        "dimension": 64,
        "page_size": 1_000,
        "backend": "pgvector",
    }
    workload.update(override)

    selection = selector.select(**workload)  # type: ignore[arg-type]

    assert selection["matcher_strategy"] == EXACT_MATCHER_STRATEGY
    assert selection["use_ann"] is False
    assert selection["workload_size"] == expected_workload_size
    assert selection["ann_not_applied_reason"] == MATCHER_SELECTED_EXACT_REASON


def test_workload_product_must_also_be_inside_evidence_region() -> None:
    region = evidence_region()
    region.update(
        min_user_count=1,
        max_user_count=1_000,
        min_segment_count=1,
        max_segment_count=1_000,
        min_workload_size=1_000_000,
        max_workload_size=2_000_000,
    )
    selector = AssignmentMatcherSelector(approved_ann_regions=(region,))

    selection = selector.select(
        user_count=100,
        segment_count=20,
        dimension=64,
        page_size=1_000,
        backend="pgvector",
    )

    assert 100 * 20 * 64 < region["min_workload_size"]
    assert selection["matcher_strategy"] == EXACT_MATCHER_STRATEGY


@pytest.mark.parametrize(
    "override",
    [
        {"user_count": None},
        {"segment_count": None},
        {"dimension": None},
        {"page_size": None},
        {"backend": None},
        {"user_count": 0},
        {"user_count": 1_001, "page_size": 1_000},
    ],
)
def test_unknown_or_invalid_workload_selects_exact(
    override: dict[str, int | str | None],
) -> None:
    selector = AssignmentMatcherSelector(
        approved_ann_regions=(evidence_region(),)
    )
    workload: dict[str, int | str | None] = {
        "user_count": 100,
        "segment_count": 20,
        "dimension": 64,
        "page_size": 1_000,
        "backend": "pgvector",
    }
    workload.update(override)

    selection = selector.select(**workload)  # type: ignore[arg-type]

    assert selection["matcher_strategy"] == EXACT_MATCHER_STRATEGY
    assert selection["use_ann"] is False
    assert selection["workload_size"] is None
    assert selection["ann_not_applied_reason"] == MATCHER_SELECTED_EXACT_REASON


def test_malformed_evidence_region_is_rejected() -> None:
    region = evidence_region()
    region["min_segment_count"] = 101
    region["max_segment_count"] = 100

    with pytest.raises(ValueError, match="minimum exceeded maximum"):
        AssignmentMatcherSelector(approved_ann_regions=(region,))


@pytest.mark.parametrize(
    ("page_strategies", "expected"),
    [
        ((), EXACT_MATCHER_STRATEGY),
        ((EXACT_MATCHER_STRATEGY,), EXACT_MATCHER_STRATEGY),
        ((ANN_MATCHER_STRATEGY,), ANN_MATCHER_STRATEGY),
        (
            (EXACT_MATCHER_STRATEGY, ANN_MATCHER_STRATEGY),
            ADAPTIVE_MATCHER_STRATEGY,
        ),
    ],
)
def test_aggregate_matcher_strategy_is_truthful(
    page_strategies: tuple[str, ...],
    expected: str,
) -> None:
    assert aggregate_matcher_strategy(page_strategies) == expected


def evidence_region() -> dict[str, str | int]:
    return {
        "backend": "pgvector",
        "min_user_count": 100,
        "max_user_count": 1_000,
        "min_segment_count": 20,
        "max_segment_count": 100,
        "min_dimension": 64,
        "max_dimension": 64,
        "min_page_size": 1_000,
        "max_page_size": 10_000,
        "min_workload_size": 100 * 20 * 64,
        "max_workload_size": 1_000 * 100 * 64,
    }
