from __future__ import annotations

import json
from pathlib import Path

from offline_evaluation.ann_search_goal3 import (
    FULL_HNSW_SETTINGS,
    REQUIRED_K_VALUES,
    append_jsonl_once,
    build_experiment_plan,
    combine_kernel_confirmations,
    crossover_report,
    empty_halving_partitions,
    kernel_final_gate,
    kernel_screen_gate,
    scenario_splits,
    validate_halving_isolation,
)
from scripts.run_ann_scale_goal3 import _unvalidated_fallbacks


def _manifest() -> dict[str, object]:
    scenarios = []
    for index, candidate_type in enumerate(
        (
            "intent_matched",
            "target_destination_affinity",
            "funnel_recovery",
            "benefit_value_seeker",
            "general_destination_explorer",
        )
    ):
        vector = [float(index)] * 64
        scenarios.append(
            {
                "candidate_type": candidate_type,
                "scenario_id": f"{candidate_type}-t",
                "scenario_set": "tuning",
                "query_vector": vector,
            }
        )
        scenarios.append(
            {
                "candidate_type": candidate_type,
                "scenario_id": f"{candidate_type}-c",
                "scenario_set": "confirmation",
                "query_vector": vector if candidate_type == "general_destination_explorer" else [float(index + 1)] * 64,
            }
        )
    return {"scenarios": scenarios}


def _kernel_summary(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "per_query_recall_at_k": 1.0,
        "aggregate_wilson_lower_bound": 0.98,
        "hnsw_index_used": True,
        "exact_uses_hnsw": False,
        "temp_spill": False,
        "oom": False,
        "exact_peak_rss_bytes": 100,
        "ann_peak_rss_bytes": 100,
        "ann_p95_ratio": 0.6,
        "ann_p99_ms": 6.0,
        "exact_p99_ms": 10.0,
        "ann_samples_ms": [4.0] * 300,
        "exact_samples_ms": [10.0] * 300,
    }
    payload.update(overrides)
    return payload


def test_scenario_split_requires_distinct_confirmation_query() -> None:
    splits = {item.candidate_type: item for item in scenario_splits(_manifest())}
    assert splits["intent_matched"].independent_confirmation_query is True
    assert splits["general_destination_explorer"].independent_confirmation_query is False


def test_plan_exhausts_every_required_k_and_hnsw_setting() -> None:
    plan = build_experiment_plan(manifest=_manifest())
    assert len(plan["partitions"]) == 5 * len(REQUIRED_K_VALUES)
    assert plan["local_calibration"]["full_setting_exhaustion"] is True
    assert len(plan["hnsw_full_settings"]) == len(FULL_HNSW_SETTINGS) == 24
    halving = empty_halving_partitions(plan)
    assert validate_halving_isolation(halving)["passed"] is True


def test_kernel_gates_keep_index_and_exact_baseline_requirements_separate() -> None:
    screened = kernel_screen_gate(_kernel_summary())
    assert screened["screening_gate_passed"] is True
    final = kernel_final_gate(_kernel_summary())
    assert final["ann_candidate_retrieval_passed"] is True
    assert kernel_screen_gate(_kernel_summary(exact_uses_hnsw=True))["screening_gate_passed"] is False


def test_append_only_identity_supports_hnsw_object(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    row = {"phase": "screening", "hnsw": {"ef_search": 100}, "value": 1}
    assert append_jsonl_once(path, (row,), identity_fields=("phase", "hnsw")) == (1, 0)
    assert append_jsonl_once(path, (row,), identity_fields=("phase", "hnsw")) == (0, 1)
    assert json.loads(path.read_text(encoding="utf-8")) == row


def test_kernel_final_joins_required_db_cold_companion() -> None:
    identity = {
        "candidate_type": "intent_matched",
        "query_id": "intent_matched-c",
        "corpus_user_count": 1_000_000,
        "requested_k": 1_000,
        "hnsw": {
            "ef_search": 100,
            "iterative_scan": "relaxed_order",
            "max_scan_tuples": 50_000,
        },
    }
    warm = {
        **_kernel_summary(),
        **identity,
        "phase": "confirmation_warm",
        "run_count": 300,
    }
    cold = {
        **_kernel_summary(),
        **identity,
        "phase": "confirmation_cold",
        "run_count": 30,
        "exact_p95_ms": 10.0,
        "ann_p95_ms": 8.0,
    }
    joined = combine_kernel_confirmations((warm,), (cold,))
    assert joined[0]["ann_candidate_retrieval_passed"] is True
    assert joined[0]["cold_confirmation_status"] == "passed"
    missing_cold = combine_kernel_confirmations((warm,), ())
    assert missing_cold[0]["ann_candidate_retrieval_passed"] is False
    assert missing_cold[0]["cold_confirmation_status"] == "required_missing"


def test_crossover_never_mixes_different_hnsw_settings() -> None:
    first = {
        "candidate_type": "intent_matched",
        "requested_k": 1_000,
        "corpus_user_count": 500_000,
        "hnsw": {
            "ef_search": 100,
            "iterative_scan": "relaxed_order",
            "max_scan_tuples": 50_000,
        },
        "db_cold_confirmation_completed": True,
        "ann_candidate_retrieval_passed": False,
    }
    second = {
        **first,
        "corpus_user_count": 1_000_000,
        "hnsw": {
            "ef_search": 200,
            "iterative_scan": "relaxed_order",
            "max_scan_tuples": 50_000,
        },
        "ann_candidate_retrieval_passed": True,
    }
    report = crossover_report((first, second), splits=scenario_splits(_manifest()))
    assert report["ann_candidate_retrieval_valid_region"] == []


def test_unvalidated_fallbacks_match_independent_query_by_runtime_setting() -> None:
    selected_setting = {
        "ef_search": 100,
        "iterative_scan": "relaxed_order",
        "max_scan_tuples": 50_000,
    }
    unselected_setting = {**selected_setting, "max_scan_tuples": 100_000}
    common = {
        "candidate_type": "intent_matched",
        "corpus_user_count": 1_000_000,
        "requested_k": 1_000,
    }
    payload = _unvalidated_fallbacks(
        kernel_screen_rows=(
            {**common, "query_id": "intent-t", "hnsw": selected_setting},
            {**common, "query_id": "intent-t", "hnsw": unselected_setting},
        ),
        kernel_confirmations=(
            {
                **common,
                "query_id": "intent-c",
                "hnsw": selected_setting,
                "ann_candidate_retrieval_passed": True,
                "cold_confirmation_status": "passed",
            },
        ),
        candidates=(),
        pre_results=(),
        full_confirmations=(),
    )
    assert payload["candidate_retrieval_exact_fallbacks"] == [
        {
            **common,
            "query_id": "intent-t",
            "hnsw": unselected_setting,
            "fallback": "exact_top_k",
            "reason": "not_selected_for_final_confirmation",
        }
    ]
    assert payload["full_membership_exact_fallbacks"] == []
