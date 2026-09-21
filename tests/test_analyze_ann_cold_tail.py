from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "performance-tests" / "ann-search" / "tools" / "analyze_cold_tail.py"
SPEC = importlib.util.spec_from_file_location("analyze_cold_tail", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(cohort: int, iteration: int, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_type": "funnel_recovery",
        "requested_k": 500,
        "hnsw": {
            "ef_search": 200,
            "iterative_scan": "relaxed_order",
            "max_scan_tuples": 20_000,
        },
        "corpus_user_count": cohort,
        "phase": "confirmation_cold",
        "measured": True,
        "iteration": iteration,
        "measurement_id": f"{cohort}-{iteration}",
        "query_id": "confirmation-query",
        "scenario_id": "confirmation-query",
        "exact_duration_ms": 100.0 + iteration,
        "ann_duration_ms": 20.0 + iteration,
        "hnsw_index_used": True,
        "exact_uses_hnsw": False,
        "temp_spill": False,
        "oom": False,
    }
    payload.update(overrides)
    return payload


def valid_rows() -> list[dict[str, object]]:
    return [
        row(cohort, iteration)
        for cohort in MODULE.TARGET_COHORTS
        for iteration in range(MODULE.EXPECTED_RUNS_PER_COHORT)
    ]


def test_script_can_be_invoked_outside_repository_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--input-dir" in completed.stdout


def test_selection_excludes_other_hnsw_settings() -> None:
    rows = valid_rows()
    rows.append(
        row(
            750_000,
            99,
            measurement_id="wrong-setting",
            hnsw={
                "ef_search": 400,
                "iterative_scan": "relaxed_order",
                "max_scan_tuples": 20_000,
            },
        )
    )
    selected = MODULE.select_and_validate_rows(rows)
    assert len(selected) == 60
    assert all(item["measurement_id"] != "wrong-setting" for item in selected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "expected 30 target rows"),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "iteration": 0}),
            "duplicate iteration",
        ),
        (
            lambda rows: rows.__setitem__(30, {**rows[30], "measurement_id": rows[0]["measurement_id"]}),
            "duplicate measurement_id",
        ),
        (
            lambda rows: rows.__setitem__(30, {**rows[30], "query_id": "different-query"}),
            "mixed query_id/scenario_id",
        ),
    ],
)
def test_validation_rejects_missing_duplicates_and_mixed_queries(mutation, message: str) -> None:
    rows = valid_rows()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        MODULE.select_and_validate_rows(rows)


@pytest.mark.parametrize("bad_latency", [0.0, -1.0, math.inf, math.nan, "20"])
def test_validation_rejects_invalid_latency(bad_latency: object) -> None:
    rows = valid_rows()
    rows[0]["ann_duration_ms"] = bad_latency
    with pytest.raises(ValueError, match="invalid ann_duration_ms"):
        MODULE.select_and_validate_rows(rows)


def test_latency_summary_uses_repository_percentile_and_distinguishes_counts() -> None:
    rows = [
        {
            "exact_duration_ms": exact,
            "ann_duration_ms": ann,
        }
        for exact, ann in zip(
            (10.0, 20.0, 30.0, 40.0, 50.0),
            (5.0, 15.0, 35.0, 45.0, 100.0),
            strict=True,
        )
    ]
    summary = MODULE.latency_summary(rows)
    assert summary["exact_median_ms"] == MODULE.percentile([10, 20, 30, 40, 50], 0.5)
    assert summary["exact_p95_ms"] == pytest.approx(48.0)
    assert summary["ann_p95_ms"] == pytest.approx(89.0)
    assert summary["ann_max_ms"] == 100.0
    assert summary["ann_exceeds_cohort_exact_p95_count"] == 1
    assert summary["ann_slower_than_paired_exact_count"] == 3
