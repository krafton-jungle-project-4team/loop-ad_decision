from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from offline_evaluation.ann_search_benchmark import (
    BenchmarkManifest,
    CellRunConfig,
    _find_plan_values,
    full_hnsw_grid,
)
from offline_evaluation.ann_search_cohort import (
    CohortPreparation,
    _cohort_query,
    _revision_row,
)
from offline_evaluation.ann_search_experiment import (
    BenchmarkObservation,
    BenchmarkPhase,
    CacheMode,
    EXPERIMENT_VERSION,
    HNSW_INDEX_NAME,
    HnswSettings,
    KPolicyParameters,
    SearchPlan,
    ScenarioSet,
    SynthesisConfig,
    bootstrap_percentile_ratio_upper_bound,
    build_current_runtime_report,
    build_phase_report,
    enumerate_unique_candidate_counts,
    select_policy,
    summarize_observations,
    synthesize_policy,
    validate_implementation_fixtures,
    validate_policy_structure,
    wilson_lower_bound,
    write_synthesis_artifacts,
)
from offline_evaluation.ann_search_macro import (
    MacroMode,
    MacroObservation,
    build_macro_report,
)


HNSW = HnswSettings(100, "strict_order", 20_000)


def test_k_policy_does_not_silently_clamp_above_validated_fraction() -> None:
    policy = KPolicyParameters(
        min_candidates=10_000,
        k_safety_factor=1.5,
        max_corpus_fraction=0.10,
    )

    assert policy.proposed_k(
        corpus_user_count=1_000_000,
        expected_member_count=50_000,
    ) == 75_000
    assert policy.proposed_k(
        corpus_user_count=100_000,
        expected_member_count=50_000,
    ) is None


def test_unique_candidate_counts_deduplicate_policy_and_ratio_grids() -> None:
    values = enumerate_unique_candidate_counts(
        corpus_user_count=100_000,
        expected_member_count=5_000,
    )

    assert values == tuple(sorted(set(values)))
    assert {1_000, 5_000, 10_000, 25_000} <= set(values)
    assert all(0 < value <= 100_000 for value in values)


def test_synthesis_emits_direct_runtime_values_and_replayable_fixtures(
    tmp_path: Path,
) -> None:
    observations = benchmark_evidence(include_supporting_modes=True)
    config = SynthesisConfig(required_candidate_types=("intent_matched",))

    result = synthesize_policy(
        observations,
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
        dataset_hash="dataset-hash",
        scenario_set_hash="scenario-hash",
        environment={"postgres": "16", "pgvector": "0.8.0"},
        config=config,
    )

    assert result.policy["score_pass_sample_size"] == 2_000
    ann_rules = [
        rule for rule in result.policy["rules"] if rule["plan"] == "ann_first"
    ]
    assert ann_rules
    ann = ann_rules[-1]["ann"]
    assert ann == {
        "min_candidates": 5_000,
        "k_safety_factor": 1.0,
        "max_corpus_fraction": 0.1,
        "ef_search": 100,
        "iterative_scan": "strict_order",
        "max_scan_tuples": 20_000,
        "validated_recall": 1.0,
        "validated_recall_lower_bound": pytest.approx(
            wilson_lower_bound(successes=2_500, trials=2_500, confidence=0.95)
        ),
    }
    assert result.comparison["candidate"]["ann_min_users"] is not None
    validate_implementation_fixtures(result.policy, result.fixtures)

    decision = select_policy(
        result.policy,
        corpus_user_count=100_000,
        hard_match_user_count=20_000,
        estimated_score_pass_rate=0.25,
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
    )
    assert decision.plan == SearchPlan.ANN_FIRST
    assert decision.requested_k == 5_000
    assert decision.hnsw == HNSW
    below_first_anchor = select_policy(
        result.policy,
        corpus_user_count=49_999,
        hard_match_user_count=10_000,
        estimated_score_pass_rate=0.25,
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
    )
    assert below_first_anchor.plan == SearchPlan.EXACT_ALL
    assert below_first_anchor.fallback_reason == "no_validated_rule"
    assert any("hard_min" in item["fixture_id"] for item in result.fixtures)

    artifacts = write_synthesis_artifacts(result, output_dir=tmp_path)
    assert set(artifacts) == {
        "policy",
        "fixtures",
        "comparison",
        "fallbacks",
        "summary_json",
        "summary_csv",
        "current_runtime",
        "report",
    }
    assert json.loads(artifacts["policy"].read_text())["rules"] == list(
        result.policy["rules"]
    )
    assert "Runtime remains unchanged" in artifacts["report"].read_text()


def test_synthesis_refuses_ann_without_cold_and_exclusion_confirmation() -> None:
    result = synthesize_policy(
        benchmark_evidence(include_supporting_modes=False),
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
        dataset_hash="dataset-hash",
        scenario_set_hash="scenario-hash",
        environment={},
        config=SynthesisConfig(required_candidate_types=("intent_matched",)),
    )

    assert all(rule["plan"] != "ann_first" for rule in result.policy["rules"])


def test_policy_falls_back_for_manifest_drift_and_candidate_cap() -> None:
    policy = {
        "version": EXPERIMENT_VERSION,
        "vector_version": "hotel_behavior.v2",
        "manifest_hash": "manifest-hash",
        "validated_max_users": 1_000_000,
        "rules": [
            {
                "min_users": 1,
                "max_users": 1_000_000,
                "min_hard_match_ratio": 0.0,
                "max_hard_match_ratio": 1.0000001,
                "min_expected_member_ratio": 0.0,
                "max_expected_member_ratio": 1.0000001,
                "plan": "ann_first",
                "ann": {
                    "min_candidates": 10_000,
                    "k_safety_factor": 3.0,
                    "max_corpus_fraction": 0.05,
                    "ef_search": 100,
                    "iterative_scan": "strict_order",
                    "max_scan_tuples": 20_000,
                    "validated_recall": 1.0,
                    "validated_recall_lower_bound": 0.96,
                },
            }
        ],
    }

    mismatch = select_policy(
        policy,
        corpus_user_count=100_000,
        hard_match_user_count=1_000,
        estimated_score_pass_rate=0.1,
        vector_version="hotel_behavior.v2",
        manifest_hash="different",
    )
    capped = select_policy(
        policy,
        corpus_user_count=100_000,
        hard_match_user_count=50_000,
        estimated_score_pass_rate=0.5,
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
    )

    assert mismatch.fallback_reason == "manifest_hash_mismatch"
    assert capped.fallback_reason == "ann_candidate_fraction_exceeded"
    assert mismatch.plan == capped.plan == SearchPlan.EXACT_ALL


def test_live_manifest_and_db_cold_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "project_id": "expedia_ann_benchmark_v1",
                "vector_version": "hotel_behavior.v2",
                "manifest_hash": "hash",
                "campaign_id": "campaign",
                "promotion_id": "promotion",
                "scenarios": [
                    {
                        "scenario_id": "intent_sparse",
                        "candidate_type": "intent_matched",
                        "query_vector": [0.0] * 64,
                        "score_threshold": 0.5,
                        "hard_predicate_keys": ["hotel_product_interest"],
                        "predicate_parameters": {},
                        "compiler_provenance": {
                            "manifest_hash": "hash",
                            "calibration_version": "calibration.v1",
                            "calibration_hash": "calibration-hash",
                            "query_compiler_version": "compiler.v1",
                            "query_compiler_hash": "compiler-hash",
                            "template_id": "hotel.intent_matched.v1",
                            "template_semantic_hash": "template-hash",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = BenchmarkManifest.load(manifest_path)

    assert manifest.promotion_id == "promotion"
    assert manifest.require_scenario("intent_sparse").candidate_type == (
        "intent_matched"
    )
    with pytest.raises(ValueError, match="DB-cold"):
        CellRunConfig(
            phase=BenchmarkPhase.CONFIRMATION,
            cache_mode=CacheMode.DB_COLD,
            warmups=0,
            repetitions=30,
        )
    cold = CellRunConfig(
        phase=BenchmarkPhase.CONFIRMATION,
        cache_mode=CacheMode.DB_COLD,
        warmups=0,
        repetitions=1,
        sample_sizes=(10_000,),
        plans=(SearchPlan.EXACT_ALL,),
    )
    assert cold.plans == (SearchPlan.EXACT_ALL,)
    assert len(full_hnsw_grid()) == 24


def test_explain_parser_finds_nested_index_and_temp_blocks() -> None:
    explain = [
        {
            "Plan": {
                "Node Type": "Limit",
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Index Name": HNSW_INDEX_NAME,
                        "Temp Read Blocks": 0,
                        "Temp Written Blocks": 3,
                    }
                ],
            }
        }
    ]

    assert _find_plan_values(explain, "Index Name") == [HNSW_INDEX_NAME]
    assert _find_plan_values(explain, "Temp Written Blocks") == [3]


def test_phase_reports_select_screening_survivors_and_tuning_winners() -> None:
    evidence = [
        replace(item, phase=BenchmarkPhase.SCREENING)
        for item in benchmark_evidence(include_supporting_modes=False)
    ]
    screening = build_phase_report(evidence, phase=BenchmarkPhase.SCREENING)

    assert screening["next_phase"] == BenchmarkPhase.TUNING.value
    assert screening["retained_cell_count"] > 0
    assert all(cell["requested_k"] == 5_000 for cell in screening["cells"])

    tuning = build_phase_report(
        [replace(item, phase=BenchmarkPhase.TUNING) for item in evidence],
        phase=BenchmarkPhase.TUNING,
    )
    assert tuning["next_phase"] == BenchmarkPhase.CONFIRMATION.value
    assert all(cell["hnsw"] == HNSW.to_dict() for cell in tuning["cells"])


def test_ann_summary_uses_worst_observed_counts_when_hnsw_frontier_varies() -> None:
    base = BenchmarkObservation(
        experiment_version=EXPERIMENT_VERSION,
        phase=BenchmarkPhase.SCREENING,
        measured=True,
        scenario_id="ann-variable-frontier",
        candidate_type="intent_matched",
        corpus_user_count=50_000,
        hard_match_user_count=5_000,
        score_pass_sample_size=50_000,
        estimated_score_pass_rate=0.2,
        actual_score_pass_rate=0.2,
        plan=SearchPlan.ANN_FIRST,
        cache_mode=CacheMode.WARM,
        exclusion_ratio=0.0,
        duration_ms=100.0,
        final_user_count=950,
        exact_positive_count=1_000,
        intersection_count=950,
        requested_k=5_000,
        hnsw=HNSW,
        index_name=HNSW_INDEX_NAME,
        scenario_set=ScenarioSet.TUNING,
    )

    summary = summarize_observations(
        [base, replace(base, final_user_count=900, intersection_count=900)]
    )[0]

    assert summary.final_user_count == 900
    assert summary.intersection_count == 900
    assert summary.precision == 1.0
    assert summary.recall == 0.9


def test_policy_structure_rejects_overlapping_rules_and_runtime_falls_back() -> None:
    rule = {
        "min_users": 1,
        "max_users": 100_000,
        "min_hard_match_ratio": 0.0,
        "max_hard_match_ratio": 0.2,
        "min_expected_member_ratio": 0.0,
        "max_expected_member_ratio": 0.05,
        "plan": "filter_first_exact",
    }
    policy = {
        "version": EXPERIMENT_VERSION,
        "vector_version": "hotel_behavior.v2",
        "manifest_hash": "manifest-hash",
        "validated_max_users": 100_000,
        "rules": [rule, dict(rule)],
    }

    with pytest.raises(ValueError, match="overlap"):
        validate_policy_structure(policy)
    decision = select_policy(
        policy,
        corpus_user_count=50_000,
        hard_match_user_count=1_000,
        estimated_score_pass_rate=0.5,
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
    )
    assert decision.plan == SearchPlan.EXACT_ALL
    assert decision.fallback_reason == "policy_structure_invalid"


def test_cohort_identity_is_stable_and_source_rows_require_valid_64d_vectors() -> None:
    config = CohortPreparation(
        project_id="expedia-ann",
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
        window_start=datetime(2013, 1, 1, tzinfo=UTC),
        window_end=datetime(2015, 1, 1, tzinfo=UTC),
        source_revision_cutoff=datetime(2026, 7, 18, tzinfo=UTC),
        cohort_size=50_000,
        cohort_seed="nested-v1",
    )
    same = replace(config)
    larger = replace(config, cohort_size=100_000)

    assert config.vector_generation_id == same.vector_generation_id
    assert config.vector_generation_id != larger.vector_generation_id
    assert "SHA256" in _cohort_query()
    assert "LIMIT {cohort_size:UInt32}" in _cohort_query()
    parsed = _revision_row(
        (
            "user-1",
            [0.125] * 64,
            "row-1",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        )
    )
    assert parsed[0] == "user-1"
    assert parsed[1].startswith("[")
    with pytest.raises(ValueError, match="64D"):
        _revision_row(
            (
                "user-1",
                [0.0] * 63,
                "row-1",
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
            )
        )


def test_wilson_lower_bound_enforces_positive_set_size() -> None:
    assert wilson_lower_bound(successes=100, trials=100, confidence=0.95) >= 0.95
    assert wilson_lower_bound(successes=50, trials=50, confidence=0.95) < 0.95


def test_bootstrap_latency_gate_and_current_runtime_report_are_deterministic() -> None:
    upper = bootstrap_percentile_ratio_upper_bound(
        [60.0] * 300,
        [100.0] * 300,
        iterations=100,
    )
    assert upper == pytest.approx(0.6)
    summaries = synthesize_policy(
        benchmark_evidence(include_supporting_modes=True),
        vector_version="hotel_behavior.v2",
        manifest_hash="manifest-hash",
        dataset_hash="dataset-hash",
        scenario_set_hash="scenario-hash",
        environment={},
        config=SynthesisConfig(
            required_candidate_types=("intent_matched",),
            bootstrap_iterations=100,
        ),
    ).summaries
    report = build_current_runtime_report(summaries)
    assert report["cell_count"] > 0
    assert report["cells"][0]["max_ann_attempt_count"] == 1
    assert report["cells"][0]["requested_k_histories"] == [[5_000]]


def test_macro_report_requires_zero_error_no_spill_and_no_latency_regression() -> None:
    observations: list[MacroObservation] = []
    for concurrency in (1, 4):
        for mode, duration in (
            (MacroMode.CURRENT_RUNTIME, 100.0),
            (MacroMode.CANDIDATE_POLICY, 70.0),
        ):
            for _ in range(100):
                observations.append(
                    MacroObservation(
                        experiment_version=EXPERIMENT_VERSION,
                        mode=mode,
                        concurrency=concurrency,
                        measured=True,
                        scenario_ids=("a", "b", "c"),
                        duration_ms=duration,
                        selected_plans=("exact", "ann", "exact"),
                        requested_k=(None, 5_000, None),
                        final_user_counts=(10, 20, 30),
                    )
                )
    report = build_macro_report(observations)
    assert report["passed"] is True
    assert all(item["p99_ratio"] == pytest.approx(0.7) for item in report["comparisons"])

    observations[-1] = replace(observations[-1], temp_spill=True)
    spilled = build_macro_report(observations)
    assert spilled["passed"] is False


def benchmark_evidence(
    *,
    include_supporting_modes: bool,
) -> list[BenchmarkObservation]:
    observations: list[BenchmarkObservation] = []
    modes = [(CacheMode.WARM, 0.0)]
    if include_supporting_modes:
        modes.extend(((CacheMode.DB_COLD, 0.0), (CacheMode.WARM, 0.05)))
    for corpus in (50_000, 100_000):
        hard_count = corpus // 5
        exact_count = hard_count // 4
        for sample_size in (2_000, 50_000):
            for cache_mode, exclusion_ratio in modes:
                for plan, duration, memory in (
                    (SearchPlan.CURRENT_RUNTIME, 130.0, 1_100),
                    (SearchPlan.EXACT_ALL, 100.0, 1_000),
                    (SearchPlan.FILTER_FIRST_EXACT, 90.0, 900),
                    (SearchPlan.ANN_FIRST, 60.0, 1_200),
                ):
                    run_count = 30 if cache_mode == CacheMode.DB_COLD else 300
                    for _ in range(run_count):
                        observations.append(
                            BenchmarkObservation(
                                experiment_version=EXPERIMENT_VERSION,
                                phase=BenchmarkPhase.CONFIRMATION,
                                measured=True,
                                scenario_id="intent-sparse",
                                candidate_type="intent_matched",
                                corpus_user_count=corpus,
                                hard_match_user_count=hard_count,
                                score_pass_sample_size=sample_size,
                                estimated_score_pass_rate=0.25,
                                actual_score_pass_rate=0.25,
                                plan=plan,
                                cache_mode=cache_mode,
                                exclusion_ratio=exclusion_ratio,
                                duration_ms=duration,
                                final_user_count=exact_count,
                                exact_positive_count=exact_count,
                                intersection_count=exact_count,
                                requested_k=(
                                    5_000 if plan == SearchPlan.ANN_FIRST else None
                                ),
                                hnsw=HNSW if plan == SearchPlan.ANN_FIRST else None,
                                index_name=(
                                    HNSW_INDEX_NAME
                                    if plan == SearchPlan.ANN_FIRST
                                    else None
                                ),
                                peak_rss_bytes=memory,
                                theoretical_min_k=4_000,
                                scenario_set=ScenarioSet.CONFIRMATION,
                                ann_attempt_count=(
                                    1
                                    if plan == SearchPlan.CURRENT_RUNTIME
                                    else 0
                                ),
                                requested_k_history=(
                                    (5_000,)
                                    if plan == SearchPlan.CURRENT_RUNTIME
                                    else ()
                                ),
                                audited_user_count=(
                                    10_000
                                    if plan == SearchPlan.CURRENT_RUNTIME
                                    else 0
                                ),
                                stage_durations_ms=(
                                    {"search_total": 100.0}
                                    if plan == SearchPlan.CURRENT_RUNTIME
                                    else {}
                                ),
                            )
                        )
    tuning_cells: set[tuple[int, int]] = set()
    for item in list(observations):
        identity = (item.corpus_user_count, item.score_pass_sample_size)
        if (
            item.plan != SearchPlan.EXACT_ALL
            or item.cache_mode != CacheMode.WARM
            or item.exclusion_ratio != 0.0
            or identity in tuning_cells
        ):
            continue
        tuning_cells.add(identity)
        observations.append(
            replace(
                item,
                phase=BenchmarkPhase.TUNING,
                scenario_id="intent-tuning",
                scenario_set=ScenarioSet.TUNING,
            )
        )
    return observations
