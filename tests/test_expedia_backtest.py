from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from app.analysis.audience_selection import load_audience_selection_policy
from offline_evaluation.expedia_backtest import (
    ClickHouseExpediaBacktestRepository,
    EXPEDIA_TRAIN_COLUMNS,
    ExpediaBacktestConfig,
    ExpediaBacktestError,
    ExpediaBacktestScenario,
    ExpediaSourceStats,
    ExpediaFutureBookingUsers,
    ExpediaSegmentBacktestService,
    run_temporal_holdout_backtest,
    monthly_cutoffs,
    summarize_backtest,
    validate_table_identifier,
    validate_train_csv_header,
    write_backtest_artifacts,
    write_temporal_holdout_artifacts,
)
from app.analysis.repositories import RawEventUserSignalRecord


class FakeExpediaBacktestRepository:
    def __init__(self) -> None:
        self.cutoff = datetime(2014, 7, 1, tzinfo=UTC)
        self.scenario = ExpediaBacktestScenario(
            scenario_id="20140701_destination_8250",
            cutoff=self.cutoff,
            target_destination_id=8250,
            historical_user_count=6,
            historical_event_count=20,
        )
        self.profiles = [
            profile(
                "1",
                hotel_search_count=3,
                hotel_detail_view_count=2,
                destination_values=("8250",),
                destination_match_count=3,
                booking_complete_count=1,
            ),
            profile(
                "2",
                hotel_search_count=2,
                hotel_detail_view_count=2,
                destination_values=("8250",),
                destination_match_count=2,
            ),
            profile(
                "3",
                booking_start_count=2,
                destination_match_count=1,
            ),
            profile(
                "4",
                booking_start_count=1,
                destination_match_count=1,
            ),
            profile(
                "5",
                deal_event_count=2,
                free_cancellation_count=1,
                destination_match_count=1,
            ),
            profile(
                "6",
                deal_event_count=1,
                breakfast_included_count=1,
                destination_match_count=1,
            ),
        ]
        self.calls: list[tuple[str, dict[str, object]]] = []

    def source_stats(self):  # pragma: no cover - not used by the service
        raise NotImplementedError

    def list_scenarios(self, **kwargs):
        self.calls.append(("list_scenarios", kwargs))
        cutoff = kwargs["cutoff"]
        return [
            replace(
                self.scenario,
                cutoff=cutoff,
                scenario_id=f"{cutoff:%Y%m%d}_destination_8250",
            )
        ]

    def list_user_profiles(self, **kwargs):
        self.calls.append(("list_user_profiles", kwargs))
        return self.profiles

    def future_booking_users(self, **kwargs):
        self.calls.append(("future_booking_users", kwargs))
        return ExpediaFutureBookingUsers(
            any_booking_user_ids=frozenset(
                {"expedia-user-1", "expedia-user-3", "expedia-user-5"}
            ),
            contextual_booking_user_ids=frozenset(
                {"expedia-user-1", "expedia-user-3"}
            ),
        )


def test_backtest_separates_observation_and_future_windows() -> None:
    repository = FakeExpediaBacktestRepository()
    service = ExpediaSegmentBacktestService(
        repository,
        config=ExpediaBacktestConfig(
            lookback_days=90,
            outcome_days=30,
            max_scenarios_per_cutoff=1,
            min_scenario_users=2,
            user_sample_modulo=1,
        ),
    )

    run = service.run([repository.cutoff])

    assert run.results
    scenario_call = next(payload for name, payload in repository.calls if name == "list_scenarios")
    profile_call = next(
        payload for name, payload in repository.calls if name == "list_user_profiles"
    )
    future_call = next(
        payload for name, payload in repository.calls if name == "future_booking_users"
    )
    expected_start = datetime(2014, 4, 2, tzinfo=UTC)
    assert scenario_call["observation_start"] == expected_start
    assert scenario_call["cutoff"] == repository.cutoff
    assert profile_call["observation_start"] == expected_start
    assert profile_call["cutoff"] == repository.cutoff
    assert future_call["outcome_end"] == datetime(2014, 7, 31, tzinfo=UTC)
    assert future_call["scenario"].cutoff == repository.cutoff


def test_backtest_runs_only_explicitly_sealed_scenarios() -> None:
    repository = FakeExpediaBacktestRepository()
    second = replace(
        repository.scenario,
        scenario_id="20140701_destination_8267",
        target_destination_id=8267,
    )
    service = ExpediaSegmentBacktestService(
        repository,
        config=ExpediaBacktestConfig(
            max_scenarios_per_cutoff=3,
            min_scenario_users=2,
            user_sample_modulo=1,
        ),
    )

    run = service.run_scenarios([repository.scenario, second])

    assert run.results
    assert not any(name == "list_scenarios" for name, _ in repository.calls)
    evaluated_scenarios = {
        payload["scenario"].scenario_id
        for name, payload in repository.calls
        if name == "future_booking_users"
    }
    assert evaluated_scenarios == {
        "20140701_destination_8250",
        "20140701_destination_8267",
    }


def test_backtest_calculates_user_level_future_rates_and_lift() -> None:
    repository = FakeExpediaBacktestRepository()
    run = ExpediaSegmentBacktestService(
        repository,
        config=ExpediaBacktestConfig(
            max_scenarios_per_cutoff=1,
            min_scenario_users=2,
            user_sample_modulo=1,
        ),
    ).run([repository.cutoff])

    assert len(run.results) >= 2
    assert all(result.total_eligible_user_count == 6 for result in run.results)
    assert all(
        result.baseline_contextual_conversion_rate == pytest.approx(2 / 6)
        for result in run.results
    )
    intent_result = next(
        result for result in run.results if result.candidate_type == "intent_matched"
    )
    assert intent_result.sample_size == 2
    assert intent_result.contextual_booking_user_count == 1
    assert intent_result.actual_contextual_conversion_rate == pytest.approx(0.5)
    assert intent_result.absolute_lift_percentage_points == pytest.approx(
        (0.5 - 2 / 6) * 100
    )
    assert intent_result.calibration_error_percentage_points >= 0


def test_backtest_summary_measures_candidate_portfolio_against_outcomes() -> None:
    repository = FakeExpediaBacktestRepository()
    run = ExpediaSegmentBacktestService(
        repository,
        config=ExpediaBacktestConfig(
            max_scenarios_per_cutoff=1,
            min_scenario_users=2,
            user_sample_modulo=1,
        ),
    ).run([repository.cutoff])

    summary = summarize_backtest(run)

    assert summary["scenario_count"] == 1
    assert summary["candidate_result_count"] == len(run.results)
    assert summary["portfolio_candidate_result_count"] == len(run.results)
    assert 0 <= summary["portfolio_candidate_beats_baseline_rate"] <= 1
    assert summary[
        "portfolio_scenario_any_candidate_beats_baseline_rate"
    ] == 1.0
    assert summary[
        "portfolio_scenario_all_candidates_beat_baseline_rate"
    ] == 0.0
    # Stored order diagnostics remain available for offline debugging only.
    assert 0 <= summary["rank_one_beats_baseline_rate"] <= 1
    assert summary["rank_one_is_best_rate"] == 0.0
    assert summary["rank_one_tied_best_rate"] == 1.0
    assert summary["pairwise_rank_accuracy"] == 1.0
    assert summary["pairwise_rank_tie_rate"] == pytest.approx(1 / 3)
    assert summary["rank_two_beats_baseline_rate"] == 1.0
    assert summary["rank_three_beats_baseline_rate"] == 0.0


def test_backtest_summary_excludes_scenarios_without_future_context_outcomes() -> None:
    repository = FakeExpediaBacktestRepository()

    def no_future_bookings(**kwargs):
        return ExpediaFutureBookingUsers(
            any_booking_user_ids=frozenset(),
            contextual_booking_user_ids=frozenset(),
        )

    repository.future_booking_users = no_future_bookings  # type: ignore[method-assign]
    run = ExpediaSegmentBacktestService(
        repository,
        config=ExpediaBacktestConfig(
            max_scenarios_per_cutoff=1,
            min_scenario_users=2,
            user_sample_modulo=1,
        ),
    ).run([repository.cutoff])

    summary = summarize_backtest(run)

    assert summary["scenario_count"] == 1
    assert summary["evaluable_scenario_count"] == 0
    assert summary["unevaluable_scenario_count"] == 1
    assert summary["rank_one_is_best_rate"] is None


def test_temporal_holdout_trains_on_2013_and_predicts_2014() -> None:
    repository = FakeExpediaBacktestRepository()
    config = ExpediaBacktestConfig(
        max_scenarios_per_cutoff=1,
        min_scenario_users=2,
        user_sample_modulo=1,
    )

    temporal_run = run_temporal_holdout_backtest(
        repository,
        config=config,
        training_cutoffs=[
            datetime(2013, 9, 1, tzinfo=UTC),
            datetime(2013, 11, 1, tzinfo=UTC),
        ],
        holdout_cutoffs=[datetime(2014, 1, 1, tzinfo=UTC)],
    )

    assert temporal_run.training_run.results
    assert temporal_run.holdout_run.results
    assert temporal_run.calibration_model.training_metadata["target"] == (
        "future_contextual_booking_rate"
    )
    assert temporal_run.calibration_model.training_metadata[
        "candidate_training_scope"
    ] == "all_eligible_candidate_types"
    assert temporal_run.calibration_model.training_metadata[
        "training_dataset"
    ] == "expedia_hotel_recommendations_train"
    assert temporal_run.calibration_model.training_metadata[
        "applicability_scope"
    ] == "destination_specific_hotel_booking_promotions"
    assert temporal_run.calibration_model.training_metadata[
        "training_example_count"
    ] > len(temporal_run.training_run.results)
    candidate_type_counts = temporal_run.calibration_model.training_metadata[
        "training_candidate_type_example_counts"
    ]
    assert sum(candidate_type_counts.values()) == (
        temporal_run.calibration_model.training_metadata[
            "training_example_count"
        ]
    )
    supported_candidate_types = {
        candidate_type
        for candidate_type, count in candidate_type_counts.items()
        if count > 0
    }
    assert {
        result.candidate_type for result in temporal_run.holdout_run.results
    } <= supported_candidate_types
    assert all(
        result.cutoff.year == 2013
        for result in temporal_run.training_run.results
    )
    assert all(
        result.cutoff.year == 2014
        for result in temporal_run.holdout_run.results
    )
    assert all(
        result.prediction_method == "temporal_holdout_logistic_calibration"
        for result in temporal_run.holdout_run.results
    )
    assert {
        outcome.selection_ratio
        for outcome in temporal_run.training_run.audience_selection_outcomes
    } == {0.2, 0.4, 0.6, 0.8, 1.0}
    assert temporal_run.audience_selection_evaluation.artifact["provenance"][
        "final_test"
    ] == "not_run"
    holdout_summary = summarize_backtest(temporal_run.holdout_run)
    assert holdout_summary["all_candidate_brier_score"] >= 0


def test_temporal_artifacts_label_2014_as_development_validation(
    tmp_path: Path,
) -> None:
    repository = FakeExpediaBacktestRepository()
    config = ExpediaBacktestConfig(
        max_scenarios_per_cutoff=1,
        min_scenario_users=2,
        user_sample_modulo=1,
    )
    temporal_run = run_temporal_holdout_backtest(
        repository,
        config=config,
        training_cutoffs=[
            datetime(2013, 9, 1, tzinfo=UTC),
            datetime(2013, 11, 1, tzinfo=UTC),
        ],
        holdout_cutoffs=[datetime(2014, 1, 1, tzinfo=UTC)],
    )

    artifacts = write_temporal_holdout_artifacts(
        temporal_run,
        output_dir=tmp_path,
        source_stats=source_stats(),
        config=config,
    )

    summary = artifacts["summary"].read_text(encoding="utf-8")
    report = artifacts["report"].read_text(encoding="utf-8")
    assert '"development_validation": "2014"' in summary
    assert '"final_test": "not_run"' in summary
    assert "개발 검증" in report
    assert "최종 일반화 성능" in report
    assert (tmp_path / "development-validation-2014" / "results.csv").exists()
    assert artifacts["audience_selection_policy"].exists()
    assert artifacts["audience_selection_development_results"].exists()
    assert artifacts["audience_selection_validation_results"].exists()
    policy = load_audience_selection_policy(
        artifacts["audience_selection_policy"]
    )
    assert policy.policy_version.startswith(
        "expedia-booking-audience-selection."
    )


def test_backtest_writes_csv_json_and_markdown_artifacts(tmp_path: Path) -> None:
    repository = FakeExpediaBacktestRepository()
    config = ExpediaBacktestConfig(
        max_scenarios_per_cutoff=1,
        min_scenario_users=2,
        user_sample_modulo=1,
    )
    run = ExpediaSegmentBacktestService(repository, config=config).run(
        [repository.cutoff]
    )

    artifacts = write_backtest_artifacts(
        run,
        output_dir=tmp_path,
        source_stats=source_stats(),
        config=config,
    )

    assert artifacts["results"].read_text(encoding="utf-8").startswith("cutoff,")
    assert '"scenario_count": 1' in artifacts["summary"].read_text(
        encoding="utf-8"
    )
    report = artifacts["report"].read_text(encoding="utf-8")
    assert "추천 후보 평균 실제 전환율" in report
    assert "Rank 1" not in report
    assert "광고의 인과적 증분 효과" in report


def test_train_csv_validation_rejects_unlabeled_test_file(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    train_path.write_text(",".join(EXPEDIA_TRAIN_COLUMNS) + "\n", encoding="utf-8")
    validate_train_csv_header(train_path)

    test_path = tmp_path / "test.csv"
    test_path.write_text(
        "id,date_time,user_id,srch_destination_id\n",
        encoding="utf-8",
    )

    with pytest.raises(ExpediaBacktestError, match="requires train.csv"):
        validate_train_csv_header(test_path)


def test_monthly_cutoffs_are_inclusive() -> None:
    assert monthly_cutoffs(date(2014, 10, 1), date(2014, 12, 1)) == [
        datetime(2014, 10, 1, tzinfo=UTC),
        datetime(2014, 11, 1, tzinfo=UTC),
        datetime(2014, 12, 1, tzinfo=UTC),
    ]


def test_clickhouse_table_identifier_rejects_sql_fragments() -> None:
    assert validate_table_identifier("expedia_hotel_events") == "expedia_hotel_events"
    with pytest.raises(ValueError, match="invalid ClickHouse table"):
        validate_table_identifier("expedia_hotel_events; DROP TABLE raw_events")


def test_future_booking_query_does_not_shadow_source_user_id() -> None:
    client = FakeClickHouseClient(
        [
            {
                "backtest_user_id": "expedia-user-1",
                "contextual_booking": 1,
            },
            {
                "backtest_user_id": "expedia-user-2",
                "contextual_booking": 0,
            },
        ]
    )
    repository = ClickHouseExpediaBacktestRepository(client)
    scenario = ExpediaBacktestScenario(
        scenario_id="20140701_destination_8250",
        cutoff=datetime(2014, 7, 1, tzinfo=UTC),
        target_destination_id=8250,
        historical_user_count=2,
        historical_event_count=4,
    )

    future = repository.future_booking_users(
        scenario=scenario,
        outcome_end=datetime(2014, 7, 31, tzinfo=UTC),
        eligible_user_ids=("expedia-user-1", "expedia-user-2"),
    )

    assert future.any_booking_user_ids == {"expedia-user-1", "expedia-user-2"}
    assert future.contextual_booking_user_ids == {"expedia-user-1"}
    assert "AS backtest_user_id" in client.queries[0]
    assert client.parameters[0]["user_ids"] == [1, 2]


def test_profile_query_does_not_invent_unsupported_benefit_properties() -> None:
    client = FakeClickHouseClient([])
    repository = ClickHouseExpediaBacktestRepository(client)
    scenario = ExpediaBacktestScenario(
        scenario_id="20140701_destination_8250",
        cutoff=datetime(2014, 7, 1, tzinfo=UTC),
        target_destination_id=8250,
        historical_user_count=20,
        historical_event_count=40,
    )

    repository.list_user_profiles(
        scenario=scenario,
        observation_start=datetime(2014, 4, 2, tzinfo=UTC),
        cutoff=scenario.cutoff,
        limit=1000,
        user_sample_modulo=1,
        user_sample_remainder=0,
    )

    query = client.queries[0]
    assert "countIf(\n            is_package = 1\n        ) AS deal_event_count" in query
    assert "toUInt64(0) AS free_cancellation_count" in query
    assert "toUInt64(0) AS breakfast_included_count" in query


def test_scenario_query_excludes_development_destinations() -> None:
    client = FakeClickHouseClient(
        [
            {
                "target_destination_id": 9001,
                "historical_user_count": 30,
                "historical_event_count": 80,
            }
        ]
    )
    repository = ClickHouseExpediaBacktestRepository(client)

    scenarios = repository.list_scenarios(
        observation_start=datetime(2014, 4, 2, tzinfo=UTC),
        cutoff=datetime(2014, 7, 1, tzinfo=UTC),
        limit=3,
        min_users=20,
        user_sample_modulo=1,
        user_sample_remainder=0,
        season=None,
        excluded_destination_ids=(8250, 8267),
    )

    assert [scenario.target_destination_id for scenario in scenarios] == [9001]
    assert client.parameters[0]["excluded_destination_ids"] == [8250, 8267]
    assert "NOT has" in client.queries[0]


def test_source_checksum_combines_sum_and_xor() -> None:
    client = FakeClickHouseClient(
        [{"checksum_sum": "1234", "checksum_xor": "5678"}]
    )
    repository = ClickHouseExpediaBacktestRepository(client)

    assert repository.source_checksum() == "1234:5678"
    assert "sumWithOverflow" in client.queries[0]
    assert "groupBitXor" in client.queries[0]


def profile(
    suffix: str,
    *,
    hotel_search_count: int = 0,
    hotel_detail_view_count: int = 0,
    booking_start_count: int = 0,
    booking_complete_count: int = 0,
    deal_event_count: int = 0,
    free_cancellation_count: int = 0,
    breakfast_included_count: int = 0,
    destination_values: tuple[str, ...] = (),
    destination_match_count: int = 0,
) -> RawEventUserSignalRecord:
    event_count = max(
        1,
        hotel_search_count
        + hotel_detail_view_count
        + booking_start_count
        + booking_complete_count
        + deal_event_count,
    )
    return RawEventUserSignalRecord(
        project_id="expedia_backtest",
        user_id=f"expedia-user-{suffix}",
        event_count=event_count,
        hotel_search_count=hotel_search_count,
        hotel_click_count=0,
        hotel_detail_view_count=hotel_detail_view_count,
        promotion_impression_count=0,
        promotion_click_count=0,
        campaign_redirect_click_count=0,
        campaign_landing_count=0,
        booking_start_count=booking_start_count,
        booking_complete_count=booking_complete_count,
        booking_cancel_count=0,
        deal_event_count=deal_event_count,
        free_cancellation_count=free_cancellation_count,
        breakfast_included_count=breakfast_included_count,
        price_event_count=0,
        avg_price=0.0,
        destination_values=destination_values,
        checkin_dates=(),
        hotel_market_values=(),
        hotel_cluster_values=(),
        age_group_values=(),
        gender_values=(),
        preferred_category_values=(),
        destination_match_count=destination_match_count,
        season_match_count=0,
    )


def source_stats() -> ExpediaSourceStats:
    return ExpediaSourceStats(
        row_count=100,
        user_count=6,
        booking_row_count=10,
        first_event_at=datetime(2013, 1, 1, tzinfo=UTC),
        last_event_at=datetime(2014, 12, 31, 23, 59, 59, tzinfo=UTC),
    )


class FakeQueryResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def named_results(self):
        return iter(self._rows)


class FakeClickHouseClient:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.queries: list[str] = []
        self.parameters: list[dict[str, object]] = []

    def query(self, query: str, parameters=None):
        self.queries.append(query)
        self.parameters.append(dict(parameters or {}))
        return FakeQueryResult(self.rows)

    def command(self, command: str, parameters=None):  # pragma: no cover
        raise NotImplementedError

    def raw_insert(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError
