from __future__ import annotations

import json
from types import SimpleNamespace

from offline_evaluation.ann_search_goal2 import (
    GOAL1_ROOT,
    build_execution_manifest,
    build_policy_coverage_audit,
    build_tuning_cells,
    full_hnsw_grid_payload,
    _build_diagnostic_plan,
    _select_scale_winners,
)
from scripts.run_ann_scale_goal2 import (
    _benchmark_environment,
    _diagnostic_row_identity,
    _final_scale_rows,
    _validate_raw_block,
)


def test_goal2_grid_is_the_preregistered_24_settings() -> None:
    grid = full_hnsw_grid_payload()

    assert len(grid) == 24
    assert {item["ef_search"] for item in grid} == {50, 100, 200, 400}
    assert {item["iterative_scan"] for item in grid} == {
        "strict_order",
        "relaxed_order",
    }
    assert {item["max_scan_tuples"] for item in grid} == {
        20_000,
        50_000,
        100_000,
    }


def test_goal2_execution_manifest_freezes_exact_expected_volume() -> None:
    cells = build_tuning_cells(GOAL1_ROOT)
    manifest = build_execution_manifest(GOAL1_ROOT)

    assert len(cells) == 2_496
    assert len({item.identity for item in cells}) == 2_496
    assert manifest["goal1_union_cell_count"] == 104
    assert manifest["unique_point_count"] == 19
    assert manifest["ann_tuning_cell_count"] == 2_496
    assert manifest["baseline_cell_count"] == 57
    assert manifest["expected_ann_invocation_count"] == 32_448
    assert manifest["expected_baseline_invocation_count"] == 741
    assert manifest["expected_total_invocation_count"] == 33_189
    assert manifest["expected_measured_observation_count"] == 25_530


def test_goal2_policy_coverage_is_not_identifiable_without_five_types() -> None:
    audit = build_policy_coverage_audit(GOAL1_ROOT)

    assert audit["bucket_count"] > 0
    assert audit["five_candidate_type_coverage_bucket_count"] == 0
    assert audit["policy_rule_possible_bucket_count"] == 0
    assert audit["product_confirmation_branch_enabled"] is False
    assert (
        audit["product_policy_track_status"]
        == "not identifiable under current policy dimensions"
    )
    assert all(not item["policy_rule_possible"] for item in audit["buckets"])


def test_goal2_diagnostics_only_register_provisional_or_representative_cells() -> None:
    passed = _gate_row(cell_id="passed", ratio=0.69, prediagnostic=True)
    failed = {
        **_gate_row(cell_id="failed", ratio=0.71, prediagnostic=False),
        "hnsw": {
            "ef_search": 50,
            "iterative_scan": "strict_order",
            "max_scan_tuples": 20_000,
        },
    }

    plan = _build_diagnostic_plan((passed, failed))

    assert [item["cell_id"] for item in plan] == ["failed", "passed"]
    assert {item["reason"] for item in plan} == {
        "provisional_scale_winner",
        "representative_adjacent_failure",
    }


def test_goal2_winner_selection_is_per_k_and_uses_fastest_p95() -> None:
    slower = {
        **_gate_row(cell_id="slow", ratio=0.60, prediagnostic=True),
        "ann_vs_exact_all_passed": True,
        "p95_ms": 80.0,
    }
    faster = {
        **_gate_row(cell_id="fast", ratio=0.65, prediagnostic=True),
        "ann_vs_exact_all_passed": True,
        "p95_ms": 70.0,
    }

    winners = _select_scale_winners((slower, faster))

    assert [item["cell_id"] for item in winners] == ["fast"]


def test_goal2_raw_resume_requires_registered_counts_and_rss(tmp_path) -> None:
    path = tmp_path / "ann.jsonl"
    rows = []
    for measured, count in ((False, 3), (True, 10)):
        for _ in range(count):
            rows.append(
                {
                    "experiment_version": "audience_search.benchmark.v2",
                    "scenario_id": "scenario-t",
                    "corpus_user_count": 50_000,
                    "score_pass_sample_size": 50_000,
                    "peak_rss_bytes": 10,
                    "temp_spill": False,
                    "oom": False,
                    "plan": "ann_first",
                    "requested_k": 5_000,
                    "hnsw": {
                        "ef_search": 100,
                        "iterative_scan": "strict_order",
                        "max_scan_tuples": 20_000,
                    },
                    "measured": measured,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    counts = _validate_raw_block(
        path,
        cohort_size=50_000,
        scenario_id="scenario-t",
        expected_plans={"ann_first"},
        expected_requested_ks={5_000},
        expected_hnsw_count=1,
    )

    assert counts == {"total": 13, "measured": 10, "warmup": 3}


def test_goal2_diagnostic_identity_accepts_nested_hnsw() -> None:
    identity = _diagnostic_row_identity(
        {
            "corpus_user_count": 50_000,
            "scenario_id": "scenario-t",
            "requested_k": 5_000,
            "hnsw": {
                "ef_search": 100,
                "iterative_scan": "strict_order",
                "max_scan_tuples": 20_000,
            },
        }
    )

    assert identity == (
        50_000,
        "scenario-t",
        5_000,
        100,
        "strict_order",
        20_000,
    )


def test_goal2_diagnostic_environment_loads_env_file_and_keeps_overrides(
    tmp_path, monkeypatch
) -> None:
    env_file = tmp_path / "benchmark.env"
    env_file.write_text(
        "LOOPAD_AURORA_USERNAME=file-user\n"
        "LOOPAD_SERVICE_ID=file-service\n"
        "LOOPAD_AURORA_DATABASE=wrong-database\n"
        "LOOPAD_CLICKHOUSE_DATABASE=wrong-clickhouse\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOOPAD_SERVICE_ID", "shell-service")
    args = SimpleNamespace(
        env_file=env_file,
        postgres_prefix="goal2",
        clickhouse_database="goal2-clickhouse",
        postgres_container="goal2-postgres",
    )

    environment = _benchmark_environment(args, 50_000)

    assert environment["LOOPAD_AURORA_USERNAME"] == "file-user"
    assert environment["LOOPAD_SERVICE_ID"] == "shell-service"
    assert environment["LOOPAD_AURORA_DATABASE"] == "goal2_50k"
    assert environment["LOOPAD_CLICKHOUSE_DATABASE"] == "goal2-clickhouse"
    assert environment["ANN_POSTGRES_CONTAINER"] == "goal2-postgres"


def test_goal2_final_scale_rows_do_not_invent_ann_values() -> None:
    rows = _final_scale_rows(
        (
            {
                "plan": "exact_all",
                "corpus_user_count": 50_000,
                "scenario_id": "scenario-t",
                "candidate_type": "funnel_recovery",
                "p50_ms": 10.0,
                "p95_ms": 20.0,
                "p99_ms": 30.0,
            },
        ),
        confirmed_winners=(),
    )

    assert rows[0]["ann_status"] == "no ANN candidate"
    assert rows[0]["ann_p95_ms"] == ""
    assert rows[0]["ann_vs_exact_all_passed"] is False


def _gate_row(
    *, cell_id: str, ratio: float, prediagnostic: bool
) -> dict[str, object]:
    return {
        "experiment_version": "audience_search.benchmark.v2",
        "cell_id": cell_id,
        "corpus_user_count": 50_000,
        "scenario_id": "scenario-t",
        "candidate_type": "funnel_recovery",
        "requested_k": 5_000,
        "hnsw": {
            "ef_search": 100,
            "iterative_scan": "strict_order",
            "max_scan_tuples": 20_000,
        },
        "precision_gate_passed": True,
        "worst_recall_gate_passed": True,
        "wilson_gate_passed": True,
        "oom_gate_passed": True,
        "prediagnostic_scale_gate_passed": prediagnostic,
        "ann_vs_exact_all_passed": False,
        "exact_all_p95_ratio": ratio,
        "p95_ms": ratio * 100,
    }
