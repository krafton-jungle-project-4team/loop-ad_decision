from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_ann_scale_goal1_measurements import (
    DEFAULT_COHORTS,
    _require_v2_manifest,
    _validate_observations,
    _validate_preflight,
)


def test_goal1_runner_keeps_the_frozen_scale_order() -> None:
    assert DEFAULT_COHORTS == (50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000)


def test_preflight_requires_matching_postgres_clickhouse_and_ready_index() -> None:
    args = SimpleNamespace(scale_series_id="series")
    payload = {
        "corpus_user_count": 50_000,
        "cohort_membership_count": 50_000,
        "cohort_signal_count": 50_000,
        "scale_series_id": "series",
        "hnsw_index": {"indisvalid": True, "indisready": True},
    }
    _validate_preflight(payload, size=50_000, args=args)
    with pytest.raises(RuntimeError, match="ClickHouse membership"):
        _validate_preflight(
            {**payload, "cohort_membership_count": 49_999},
            size=50_000,
            args=args,
        )


def test_goal1_manifest_rejects_non_v2_and_exclusion_context() -> None:
    _require_v2_manifest({"experiment_version": "audience_search.benchmark.v2"})
    with pytest.raises(ValueError, match="benchmark v2"):
        _require_v2_manifest({"experiment_version": "audience_search.benchmark.v1"})
    with pytest.raises(ValueError, match="exclusion"):
        _require_v2_manifest(
            {
                "experiment_version": "audience_search.benchmark.v2",
                "promotion_id": "promotion",
            }
        )


def test_measurement_resume_requires_complete_plan_and_run_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    rows = []
    for plan, requested_k in (
        ("current_runtime", None),
        ("exact_all", None),
        ("filter_first_exact", None),
        ("ann_first", 5_000),
    ):
        for measured, count in ((False, 1), (True, 2)):
            for _ in range(count):
                rows.append(
                    {
                        "experiment_version": "audience_search.benchmark.v2",
                        "scenario_id": "scenario",
                        "corpus_user_count": 50_000,
                        "plan": plan,
                        "requested_k": requested_k,
                        "measured": measured,
                        "score_pass_sample_size": 50_000,
                        "peak_rss_bytes": 1,
                        "temp_spill": False,
                        "oom": False,
                    }
                )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _validate_observations(
        path,
        size=50_000,
        scenario_id="scenario",
        warmups=1,
        repetitions=2,
        confirmation=True,
        requested_k=5_000,
    )
    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in rows
            if row["plan"] != "filter_first_exact"
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="plan coverage"):
        _validate_observations(
            path,
            size=50_000,
            scenario_id="scenario",
            warmups=1,
            repetitions=2,
            confirmation=True,
            requested_k=5_000,
        )
