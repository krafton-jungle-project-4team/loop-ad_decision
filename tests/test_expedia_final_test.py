from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from offline_evaluation.expedia_backtest import (
    ExpediaBacktestConfig,
    ExpediaBacktestError,
    ExpediaBacktestScenario,
    ExpediaFutureBookingUsers,
    ExpediaSourceStats,
)
from offline_evaluation.expedia_final_test import (
    build_sealed_final_test_manifest,
    load_sealed_final_test_manifest,
    reserve_sealed_final_test_execution,
    run_sealed_final_test,
    verify_sealed_final_test_runtime,
    write_sealed_final_test_artifacts,
    write_sealed_final_test_manifest,
)
from offline_evaluation.sealed_execution import (
    mark_execution_failure,
    mark_outcomes_opened,
)
from app.analysis.repositories import RawEventUserSignalRecord
from app.analysis.segment_performance import (
    CANDIDATE_TYPE_SUPPORT_CONTRACT_VERSION,
    MODEL_FEATURE_NAMES,
    LogisticSegmentPerformanceModel,
    write_segment_performance_model,
)


class FakeSealedFinalTestRepository:
    def __init__(self) -> None:
        self.future_call_count = 0
        self.outcome_events: list[str] = []
        self.scenario_calls: list[dict[str, object]] = []

    def source_stats(self) -> ExpediaSourceStats:
        return source_stats()

    def source_checksum(self) -> str:
        return "123:456"

    def list_scenarios(self, **kwargs):
        self.scenario_calls.append(kwargs)
        cutoff = kwargs["cutoff"]
        excluded = set(kwargs["excluded_destination_ids"])
        limit = int(kwargs["limit"])
        if not excluded:
            destination_ids = [100, 101][:limit]
        else:
            base = 9000 + cutoff.month * 10
            destination_ids = [
                value
                for value in range(base, base + limit + 5)
                if value not in excluded
            ][:limit]
        return [
            ExpediaBacktestScenario(
                scenario_id=f"{cutoff:%Y%m%d}_destination_{destination_id}",
                cutoff=cutoff,
                target_destination_id=destination_id,
                historical_user_count=50,
                historical_event_count=120,
            )
            for destination_id in destination_ids
        ]

    def list_user_profiles(self, **kwargs):
        return [
            profile("1", destination_match_count=3, hotel_search_count=3),
            profile("2", destination_match_count=2, hotel_search_count=2),
            profile("3", destination_match_count=1, booking_start_count=2),
            profile("4", destination_match_count=1, booking_start_count=1),
            profile("5", destination_match_count=1, deal_event_count=2),
            profile("6", destination_match_count=1, deal_event_count=1),
        ]

    def future_booking_users(self, **kwargs):
        self.future_call_count += 1
        self.outcome_events.append("future_query")
        return ExpediaFutureBookingUsers(
            any_booking_user_ids=frozenset(
                {"expedia-user-1", "expedia-user-3", "expedia-user-5"}
            ),
            contextual_booking_user_ids=frozenset(
                {"expedia-user-1", "expedia-user-3"}
            ),
        )


def test_seal_selects_unseen_destinations_without_reading_future_outcomes(
    tmp_path,
) -> None:
    repository = FakeSealedFinalTestRepository()
    model_path, model = model_fixture(tmp_path)

    manifest = build_manifest(repository, model_path, model)

    assert repository.future_call_count == 0
    assert manifest.development_validation["excluded_destination_ids"] == [100, 101]
    final_scenarios = manifest.final_test["scenarios"]
    final_destination_ids = [
        scenario["target_destination_id"] for scenario in final_scenarios
    ]
    assert len(final_destination_ids) == 4
    assert len(set(final_destination_ids)) == 4
    assert not {100, 101}.intersection(final_destination_ids)
    assert manifest.final_test["selection_uses_future_outcomes"] is False
    candidate_support = manifest.model["candidate_type_support"]
    assert candidate_support["training_example_counts"][
        "promotion_responsive"
    ] == 0
    assert "promotion_responsive" not in candidate_support[
        "supported_candidate_types"
    ]


def test_manifest_integrity_rejects_changed_scenario(tmp_path) -> None:
    repository = FakeSealedFinalTestRepository()
    model_path, model = model_fixture(tmp_path)
    manifest = build_manifest(repository, model_path, model)
    manifest_path = tmp_path / "sealed.json"
    write_sealed_final_test_manifest(manifest, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["final_test"]["scenarios"][0]["target_destination_id"] = 999999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check failed"):
        load_sealed_final_test_manifest(manifest_path)


def test_runtime_verification_rejects_changed_code_or_source(tmp_path) -> None:
    repository = FakeSealedFinalTestRepository()
    model_path, model = model_fixture(tmp_path)
    manifest = build_manifest(repository, model_path, model)

    verify_sealed_final_test_runtime(
        manifest,
        source_table="expedia_hotel_events",
        source_stats=source_stats(),
        source_checksum="123:456",
        model_path=model_path,
        model=model,
        code_commit="commit-1",
        code_tree="tree-1",
    )
    with pytest.raises(ValueError, match="source data changed"):
        verify_sealed_final_test_runtime(
            manifest,
            source_table="expedia_hotel_events",
            source_stats=source_stats(),
            source_checksum="changed",
            model_path=model_path,
            model=model,
            code_commit="commit-1",
            code_tree="tree-1",
        )


def test_runtime_verification_rejects_changed_candidate_support(tmp_path) -> None:
    repository = FakeSealedFinalTestRepository()
    model_path, model = model_fixture(tmp_path)
    manifest = build_manifest(repository, model_path, model)
    changed_metadata = dict(model.training_metadata)
    changed_counts = dict(
        changed_metadata["training_candidate_type_example_counts"]
    )
    changed_counts["promotion_responsive"] = 1
    changed_metadata["training_candidate_type_example_counts"] = changed_counts
    changed_metadata["supported_candidate_types"] = [
        *changed_metadata["supported_candidate_types"],
        "promotion_responsive",
    ]
    changed_model = replace(model, training_metadata=changed_metadata)

    with pytest.raises(ValueError, match="candidate type support changed"):
        verify_sealed_final_test_runtime(
            manifest,
            source_table="expedia_hotel_events",
            source_stats=source_stats(),
            source_checksum="123:456",
            model_path=model_path,
            model=changed_model,
            code_commit="commit-1",
            code_tree="tree-1",
        )


def test_expedia_execution_blocks_retry_after_outcomes_are_opened(tmp_path) -> None:
    repository = FakeSealedFinalTestRepository()
    model_path, model = model_fixture(tmp_path)
    manifest = build_manifest(repository, model_path, model)
    manifest_path = tmp_path / "sealed.json"
    write_sealed_final_test_manifest(manifest, manifest_path)

    execution = reserve_sealed_final_test_execution(
        manifest_path,
        manifest,
        code_commit="commit-1",
        output_dir=tmp_path / "result",
    )
    mark_outcomes_opened(execution)
    mark_execution_failure(execution, RuntimeError("evaluation failed"))

    with pytest.raises(ExpediaBacktestError, match="repeat the final test"):
        reserve_sealed_final_test_execution(
            manifest_path,
            manifest,
            code_commit="commit-1",
            output_dir=tmp_path / "result",
            resume_execution_id=execution.execution_id,
        )


def test_final_test_uses_fixed_scenarios_and_writes_verdict(tmp_path) -> None:
    repository = FakeSealedFinalTestRepository()
    model_path, model = model_fixture(tmp_path)
    manifest = build_manifest(repository, model_path, model)

    result = run_sealed_final_test(
        repository,
        manifest=manifest,
        model=model,
        on_outcomes_opened=lambda: repository.outcome_events.append(
            "outcomes_opened"
        ),
    )
    artifacts = write_sealed_final_test_artifacts(
        result,
        manifest=manifest,
        output_dir=tmp_path / "result",
        source_stats=source_stats(),
    )

    assert repository.future_call_count == 4
    assert repository.outcome_events == [
        "outcomes_opened",
        "future_query",
        "future_query",
        "future_query",
        "future_query",
    ]
    assert result.metrics["candidate_result_count"] > 0
    assert "all_candidate_brier_skill_score" in result.metrics
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    assert summary["manifest_id"] == manifest.manifest_id
    assert summary["verdict"] == "inconclusive"
    assert summary["passed"] is False
    assert "pairwise_rank_accuracy" in summary["criteria_results"]
    assert "rank_three_beats_baseline_rate" in summary["criteria_results"]
    assert "새로운 연도" in artifacts["report"].read_text(encoding="utf-8")


def build_manifest(
    repository: FakeSealedFinalTestRepository,
    model_path,
    model,
):
    return build_sealed_final_test_manifest(
        repository,
        source_table="expedia_hotel_events",
        source_stats=source_stats(),
        source_checksum="123:456",
        model_path=model_path,
        model=model,
        config=ExpediaBacktestConfig(
            max_scenarios_per_cutoff=2,
            max_suggested_segments=3,
            min_scenario_users=20,
            user_sample_modulo=1,
        ),
        development_cutoffs=(
            datetime(2014, 1, 1, tzinfo=UTC),
            datetime(2014, 2, 1, tzinfo=UTC),
        ),
        final_cutoffs=(
            datetime(2014, 7, 1, tzinfo=UTC),
            datetime(2014, 8, 1, tzinfo=UTC),
        ),
        development_scenarios_per_cutoff=2,
        code_commit="commit-1",
        code_tree="tree-1",
        created_at=datetime(2026, 7, 10, tzinfo=UTC),
    )


def model_fixture(tmp_path):
    model = LogisticSegmentPerformanceModel(
        intercept=-2.5,
        coefficients=(0.0,) * len(MODEL_FEATURE_NAMES),
        feature_means=(0.0,) * len(MODEL_FEATURE_NAMES),
        feature_scales=(1.0,) * len(MODEL_FEATURE_NAMES),
        training_metadata={
            "training_end_cutoff": "2013-12-01T00:00:00+00:00",
            "training_target": "future_contextual_booking_rate",
            "training_contextual_booking_observation_rate": 0.04,
            "candidate_type_support_contract_version": (
                CANDIDATE_TYPE_SUPPORT_CONTRACT_VERSION
            ),
            "training_candidate_type_example_counts": {
                "intent_matched": 12,
                "target_destination_affinity": 12,
                "funnel_recovery": 12,
                "benefit_value_seeker": 12,
                "promotion_responsive": 0,
                "general_destination_explorer": 0,
            },
            "training_candidate_type_user_observation_counts": {
                "intent_matched": 120,
                "target_destination_affinity": 120,
                "funnel_recovery": 120,
                "benefit_value_seeker": 120,
                "promotion_responsive": 0,
                "general_destination_explorer": 0,
            },
            "supported_candidate_types": [
                "intent_matched",
                "target_destination_affinity",
                "funnel_recovery",
                "benefit_value_seeker",
            ],
        },
    )
    model_path = tmp_path / "model.json"
    write_segment_performance_model(model, model_path)
    return model_path, model


def source_stats() -> ExpediaSourceStats:
    return ExpediaSourceStats(
        row_count=1000,
        user_count=100,
        booking_row_count=20,
        first_event_at=datetime(2013, 1, 1, tzinfo=UTC),
        last_event_at=datetime(2014, 12, 31, 23, 59, tzinfo=UTC),
    )


def profile(
    suffix: str,
    *,
    hotel_search_count: int = 0,
    booking_start_count: int = 0,
    deal_event_count: int = 0,
    destination_match_count: int = 0,
) -> RawEventUserSignalRecord:
    event_count = max(
        1,
        hotel_search_count + booking_start_count + deal_event_count,
    )
    return RawEventUserSignalRecord(
        project_id="expedia_backtest",
        user_id=f"expedia-user-{suffix}",
        event_count=event_count,
        hotel_search_count=hotel_search_count,
        hotel_click_count=0,
        hotel_detail_view_count=hotel_search_count,
        promotion_impression_count=0,
        promotion_click_count=0,
        campaign_redirect_click_count=0,
        campaign_landing_count=0,
        booking_start_count=booking_start_count,
        booking_complete_count=0,
        booking_cancel_count=0,
        deal_event_count=deal_event_count,
        free_cancellation_count=deal_event_count,
        breakfast_included_count=0,
        price_event_count=0,
        avg_price=0.0,
        destination_values=(),
        checkin_dates=(),
        hotel_market_values=(),
        hotel_cluster_values=(),
        age_group_values=(),
        gender_values=(),
        preferred_category_values=(),
        destination_match_count=destination_match_count,
        season_match_count=0,
    )
