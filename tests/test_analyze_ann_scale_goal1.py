from __future__ import annotations

from offline_evaluation.ann_search_experiment import (
    CacheMode,
    HnswSettings,
    RunSummary,
    SearchPlan,
)
from offline_evaluation.ann_search_scale_series import ScenarioSet
from scripts.analyze_ann_scale_goal1 import screening_candidates


def summary(plan: SearchPlan, *, k: int | None = None, p95: float = 100.0) -> RunSummary:
    return RunSummary(
        scenario_id="scenario",
        candidate_type="intent_matched",
        corpus_user_count=50_000,
        hard_match_user_count=5_000,
        score_pass_sample_size=50_000,
        estimated_score_pass_rate=0.1,
        actual_score_pass_rate=0.1,
        plan=plan,
        cache_mode=CacheMode.WARM,
        exclusion_ratio=0.0,
        requested_k=k,
        hnsw=(HnswSettings(100, "strict_order", 20_000) if k else None),
        run_count=10,
        mean_ms=p95,
        p50_ms=p95,
        p95_ms=p95,
        p99_ms=p95,
        final_user_count=500,
        exact_positive_count=500,
        intersection_count=500,
        precision=1.0,
        recall=1.0,
        recall_lower_bound=0.99,
        index_used=True,
        temp_spill=False,
        oom=False,
        peak_rss_bytes=1,
        theoretical_min_k=100,
        scenario_set=ScenarioSet.TUNING,
        duration_samples_ms=(p95,) * 10,
        max_ann_attempt_count=0,
        requested_k_histories=(),
        max_audited_user_count=0,
        exact_fallback_rate=0.0,
        stage_p95_ms={},
    )


def test_screening_separates_exact_all_and_fastest_exact_gates() -> None:
    rows = (
        summary(SearchPlan.EXACT_ALL, p95=100.0),
        summary(SearchPlan.FILTER_FIRST_EXACT, p95=50.0),
        summary(SearchPlan.ANN_FIRST, k=5_000, p95=90.0),
    )
    candidate = screening_candidates(rows)[0]
    assert candidate.scale_survivor is True
    assert candidate.policy_survivor is False
