from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest


np = pytest.importorskip("numpy")

import offline_evaluation.assignment_benchmark as assignment_benchmark  # noqa: E402
from offline_evaluation.assignment_benchmark import (  # noqa: E402
    one_sided_wilson_upper,
    run_benchmark,
    write_benchmark_report,
)
from offline_evaluation.assignment_corpus import generate_synthetic_corpus  # noqa: E402
from offline_evaluation.assignment_matchers import MatcherConfig  # noqa: E402
from offline_evaluation.assignment_shadow import (  # noqa: E402
    SHADOW_WRITE_POLICY,
    run_assignment_shadow,
    write_shadow_report,
)
from scripts.benchmark_segment_assignments import (  # noqa: E402
    _begin_summary,
    _summary,
)


def test_one_sided_wilson_upper_bound() -> None:
    assert one_sided_wilson_upper(0, 100) == pytest.approx(0.0263427, rel=1e-5)
    assert one_sided_wilson_upper(100, 100) == 1.0
    with pytest.raises(ValueError):
        one_sided_wilson_upper(1, 0)


def test_small_benchmark_and_shadow_are_evidence_only(tmp_path, monkeypatch) -> None:
    pytest.importorskip("faiss")
    config = MatcherConfig(
        exact_batch_size=16,
        candidate_k=8,
        hnsw_m=8,
        hnsw_ef_construction=40,
        hnsw_ef_search=16,
        rescue_threshold_band=0.05,
        rescue_margin=0.05,
    )
    corpus = generate_synthetic_corpus(
        user_count=32,
        segment_count=32,
        distribution="clustered",
        random_seed=11,
        git_commit="abc123",
        matcher_config=config.to_dict(),
    )
    with pytest.raises(ValueError, match="at least 2"):
        run_benchmark(corpus, config=config, timing_trial_count=1)
    monkeypatch.setattr(
        assignment_benchmark,
        "_resource_peak_rss_bytes",
        lambda: 10**15,
    )
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    report = run_benchmark(corpus, config=config, warmup_user_count=4)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert report.format_version == "loopad.assignment-benchmark.v5"
    assert report.requested_warmup_user_count == 4
    assert report.warmup_user_count == 4
    assert report.timing_trial_count == 3
    assert report.timing_trial_schedule == (
        ("exact", "fixed", "adaptive"),
        ("fixed", "adaptive", "exact"),
        ("adaptive", "exact", "fixed"),
    )
    assert report.corpus_manifest["git_commit"] == "abc123"
    assert report.benchmark_code_identity["git_commit"]
    assert len(report.benchmark_code_identity["benchmark_source_sha256"]) == 64
    assert (
        "offline_evaluation/assignment_benchmark.py"
        in report.benchmark_code_identity["benchmark_source_files_sha256"]
    )
    assert report.faiss_flat_crosscheck == "passed"
    assert [value.matcher for value in report.matchers] == [
        "exact_numpy_batched_matmul",
        "fixed_faiss_hnsw_exact_candidate_rerank",
        "adaptive_faiss_hnsw_exact_rescue",
    ]
    assert report.matchers[0].agreement.assignment_agreement == 1.0
    assert report.matchers[0].agreement.oracle_fallback_count >= 0
    assert report.matchers[0].agreement.candidate_fallback_count >= 0
    assert report.matchers[2].agreement.post_rescue_segment_agreement >= report.matchers[2].agreement.pre_rescue_segment_agreement
    assert report.matchers[0].timing.latency_sample_count == 6
    assert report.matchers[0].timing.end_to_end_sample_count == 3
    assert len(report.matchers[0].timing.trial_samples) == 3
    assert report.matchers[0].timing.percentile_method == "numpy_linear"
    exact_trial_e2e_ms = [
        sample["end_to_end_seconds"] * 1000.0
        for sample in report.matchers[0].timing.trial_samples
    ]
    assert report.matchers[0].timing.end_to_end_p50_ms == pytest.approx(
        np.percentile(exact_trial_e2e_ms, 50, method="linear")
    )
    assert report.matchers[0].timing.end_to_end_p95_ms == pytest.approx(
        np.percentile(exact_trial_e2e_ms, 95, method="linear")
    )
    assert report.matchers[0].timing.end_to_end_p95_ms >= (
        report.matchers[0].timing.end_to_end_p50_ms
    )
    assert report.matchers[0].timing.output_assembly_seconds >= 0.0
    assert "excludes spawn startup" in (
        report.matchers[0].timing.end_to_end_definition
    )
    assert all(result.timing.peak_rss_bytes < 10**15 for result in report.matchers)
    assert all(
        result.execution_environment["process_role"] == "matcher_spawn_child"
        for result in report.matchers
    )
    assert all(
        result.execution_environment["matcher_process_isolation"]
        == "fresh_spawn_child_per_matcher_trial"
        for result in report.matchers
    )
    assert all(
        result.execution_environment["process_id"]
        != report.execution_environment["process_id"]
        for result in report.matchers
    )
    assert all(
        result.execution_environment["timing_trial_process_count"] == 3
        and len(result.execution_environment["trial_process_ids"]) == 3
        and len(set(result.execution_environment["trial_process_ids"])) == 3
        for result in report.matchers
    )
    assert (
        report.matchers[1].execution_environment["thread_settings"][
            "faiss_omp_threads"
        ]
        == config.faiss_threads
    )
    assert all(
        matcher.execution_environment["thread_settings"][name]
        == str(config.blas_threads)
        for matcher in report.matchers
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    )
    assert "numpy_blas_build" in (
        report.matchers[0].execution_environment["thread_settings"]
    )
    for matcher in report.matchers:
        settings = matcher.execution_environment["thread_settings"]
        for pool in settings["runtime_threadpools"]:
            if pool["user_api"] == "blas":
                assert pool["num_threads"] == config.blas_threads
        if (
            settings["numpy_blas_build"]
            and settings["numpy_blas_build"].get("name") == "accelerate"
        ):
            assert (
                settings["blas_control_verification"]
                == "environment_only_unverified"
            )
    assert any(
        pool["num_threads"] == config.faiss_threads
        for pool in report.matchers[1].execution_environment["thread_settings"][
            "runtime_threadpools"
        ]
        if pool["user_api"] == "openmp"
    )

    benchmark_path = tmp_path / "benchmark.json"
    write_benchmark_report(report, benchmark_path)
    assert json.loads(benchmark_path.read_text())["corpus_manifest"]["dimension"] == 64

    shadow = run_assignment_shadow(corpus, config=config, max_mismatch_examples=5)
    assert shadow.format_version == "loopad.assignment-shadow.v3"
    assert shadow.write_policy == SHADOW_WRITE_POLICY
    assert shadow.corpus_manifest["corpus_sha256"] == corpus.manifest.corpus_sha256
    assert shadow.benchmark_code_identity["benchmark_source_sha256"]
    assert shadow.execution_environment["process_role"] == "shadow_single_process"
    assert (
        shadow.execution_environment["matcher_process_isolation"]
        == "single_process_exact_fixed_adaptive_sequential"
    )
    for pool in shadow.execution_environment["thread_settings"][
        "runtime_threadpools"
    ]:
        if pool["user_api"] == "blas":
            assert pool["num_threads"] == config.blas_threads
    assert shadow.mismatch_example_limit == 5
    assert shadow.mismatch_user_count == 0
    assert shadow.mismatch_examples_truncated is False
    shadow_path = tmp_path / "shadow.json"
    write_shadow_report(shadow, shadow_path)
    payload = shadow_path.read_text(encoding="utf-8")
    assert "synthetic-user" not in payload
    assert "write_policy" in payload
    assert "benchmark_code_identity" in payload

    summary = _summary(
        [
            (
                corpus.manifest.distribution,
                corpus.manifest.user_count,
                corpus.manifest.segment_count,
                report,
                benchmark_path,
            )
        ],
        mode="provided_corpus",
        config=config,
    )
    assert summary["mode"] == "provided_corpus"
    assert summary["status"] == "complete"
    assert summary["crossover_observations"] == []

    stale_summary_path = tmp_path / "stale-summary.json"
    stale_summary_path.write_text('{"status":"complete"}\n', encoding="utf-8")
    _begin_summary(stale_summary_path, mode="full", overwrite=True)
    assert json.loads(stale_summary_path.read_text(encoding="utf-8")) == {
        "format_version": "loopad.assignment-benchmark-summary.v2",
        "mode": "full",
        "note": (
            "The runner replaces this marker only after every case and optional "
            "shadow report completes. Do not use it as benchmark evidence."
        ),
        "status": "incomplete",
    }

    def with_end_to_end_times(
        segment_count: int,
        exact_seconds: float,
        fixed_seconds: float,
        adaptive_seconds: float,
        *,
        p95_seconds: tuple[float, float, float] | None = None,
    ):
        median_seconds = (exact_seconds, fixed_seconds, adaptive_seconds)
        p95_values = p95_seconds or median_seconds
        return replace(
            report,
            corpus_manifest={
                **report.corpus_manifest,
                "segment_count": segment_count,
            },
            matchers=tuple(
                replace(
                    matcher,
                    timing=replace(
                        matcher.timing,
                        end_to_end_seconds=seconds,
                        end_to_end_p50_ms=seconds * 1000.0,
                        end_to_end_p95_ms=p95_seconds_value * 1000.0,
                    ),
                )
                for matcher, seconds, p95_seconds_value in zip(
                    report.matchers,
                    median_seconds,
                    p95_values,
                    strict=True,
                )
            ),
        )

    matrix_summary = _summary(
        [
            (
                "clustered",
                32,
                32,
                with_end_to_end_times(32, 1.0, 2.0, 3.0),
                tmp_path / "s32.json",
            ),
            (
                "clustered",
                32,
                64,
                with_end_to_end_times(64, 1.0, 0.5, 0.4),
                tmp_path / "s64.json",
            ),
        ],
        mode="smoke",
        config=config,
    )
    assert [
        crossover["observed_bracketed_crossover_segment_count"]
        for crossover in matrix_summary["crossover_observations"]
    ] == [64, 64]
    assert all(
        crossover["evaluated_segment_counts"] == [32, 64]
        for crossover in matrix_summary["crossover_observations"]
    )
    assert all(
        crossover["status"] == "observed_bracketed_trial_p95"
        for crossover in matrix_summary["crossover_observations"]
    )

    unbracketed_summary = _summary(
        [
            (
                "clustered",
                32,
                32,
                with_end_to_end_times(32, 1.0, 0.5, 0.4),
                tmp_path / "unbracketed-s32.json",
            )
        ],
        mode="smoke",
        config=config,
    )
    assert all(
        crossover["minimum_observed_faster_segment_count"] == 32
        and crossover["observed_bracketed_crossover_segment_count"] is None
        and crossover["status"] == "unbracketed_faster_at_minimum"
        for crossover in unbracketed_summary["crossover_observations"]
    )

    p95_governs_summary = _summary(
        [
            (
                "clustered",
                32,
                32,
                with_end_to_end_times(
                    32,
                    1.0,
                    0.5,
                    0.4,
                    p95_seconds=(1.0, 2.0, 3.0),
                ),
                tmp_path / "p95-governs-s32.json",
            )
        ],
        mode="smoke",
        config=config,
    )
    assert all(
        crossover["status"] == "not_observed"
        and crossover["timing_comparison_basis"]
        == "end_to_end_p95_ms_across_trials"
        for crossover in p95_governs_summary["crossover_observations"]
    )

    non_monotonic_summary = _summary(
        [
            (
                "clustered",
                32,
                segment_count,
                with_end_to_end_times(
                    segment_count,
                    1.0,
                    candidate_seconds,
                    candidate_seconds,
                ),
                tmp_path / f"non-monotonic-s{segment_count}.json",
            )
            for segment_count, candidate_seconds in (
                (16, 0.5),
                (32, 2.0),
                (64, 0.5),
            )
        ],
        mode="smoke",
        config=config,
    )
    assert all(
        crossover["status"] == "non_monotonic_or_unstable"
        and crossover["observed_bracketed_crossover_segment_count"] is None
        for crossover in non_monotonic_summary["crossover_observations"]
    )

    def fail_if_compared(*_args, **_kwargs):
        raise AssertionError("zero-example shadow must not scan mismatches")

    monkeypatch.setattr(
        "offline_evaluation.assignment_shadow._assignment_equal",
        fail_if_compared,
    )
    empty_examples = run_assignment_shadow(
        corpus,
        config=config,
        max_mismatch_examples=0,
    )
    assert empty_examples.mismatch_examples == ()
    assert empty_examples.mismatch_example_limit == 0
