from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.analysis.raw_event_segments import PromotionIntent
from app.analysis.repositories import PromotionRecord, RawEventUserSignalRecord
from app.analysis.segment_performance import (
    ContextualBookingHeuristicPredictor,
    build_segment_performance_predictor,
)
from offline_evaluation.audience_selection import (
    AudienceSelectionEvaluationConfig,
)
from offline_evaluation.external_audience_selection import (
    build_external_audience_selection_diagnostic,
    evaluate_external_audience_selection_cases,
    write_external_audience_selection_diagnostic_artifacts,
)
from offline_evaluation.external_backtest import (
    ExternalBacktestConfig,
    ExternalDatasetManifest,
    ExternalEvaluationCase,
    run_external_backtest,
    write_external_backtest_artifacts,
)
from offline_evaluation.external_datasets import (
    EXTERNAL_SEALED_FINAL_ROLE,
    ExternalAdapterConfig,
    load_airbnb_dataset,
    load_booking_com_dataset,
    load_booking_com_final_dataset,
    load_synerise_dataset,
)


def test_external_backtest_uses_production_candidates_and_future_outcomes(
    tmp_path: Path,
) -> None:
    profiles = (
        _profile("u1", search=3, starts=2, completes=0, destination_match=2),
        _profile("u2", search=2, starts=1, completes=0, destination_match=1),
        _profile("u3", search=1, prices=2, destination_match=1),
        _profile("u4", search=1),
    )
    case = _case(profiles=profiles, positive_user_ids=frozenset({"u1", "u2"}))

    run = run_external_backtest(
        (case,),
        config=ExternalBacktestConfig(
            max_suggested_segments=3,
            min_sample_size=1,
        ),
        performance_predictor=ContextualBookingHeuristicPredictor(),
    )

    assert run.results
    candidate_types = {result.candidate_type for result in run.results}
    assert "funnel_recovery" in candidate_types
    assert len(candidate_types) >= 2
    assert all(result.outcome_name == "future_target_rate" for result in run.results)
    assert run.summary["scenario_count"] == 1
    assert run.summary["candidate_result_count"] == len(run.results)
    assert run.summary["prediction_error_comparable"] is False
    assert (
        run.summary["mean_absolute_prediction_error_percentage_points"] is None
    )

    manifest = ExternalDatasetManifest(
        dataset_id="fixture",
        source_version="fixture.v1",
        evaluation_design="temporal_holdout",
        outcome_name="future_target_rate",
        supports_temporal_holdout=True,
        supported_claims=("fixture claim",),
        unsupported_claims=("causal lift",),
        signal_mappings={},
        source_files=(),
    )
    paths = write_external_backtest_artifacts(
        run,
        manifest=manifest,
        output_dir=tmp_path / "artifacts",
        model_metadata=ContextualBookingHeuristicPredictor().metadata(),
    )
    assert paths["results"].is_file()
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["metrics"]["scenario_count"] == 1


def test_external_backtest_uses_booking_model_candidate_support_contract() -> None:
    profiles = (
        _profile("u1", search=3, starts=2, completes=0, destination_match=2),
        _profile("u2", search=2, starts=1, completes=0, destination_match=1),
        _profile("u3", search=1, prices=2, destination_match=1),
        _profile("u4", search=1, destination_match=1),
    )
    case = _case(profiles=profiles, positive_user_ids=frozenset({"u1", "u2"}))
    predictor = build_segment_performance_predictor()

    run = run_external_backtest(
        (case,),
        config=ExternalBacktestConfig(
            max_suggested_segments=3,
            min_sample_size=1,
        ),
        performance_predictor=predictor,
    )

    supported_candidate_types = set(
        predictor.metadata()["supported_candidate_types"]
    )
    assert run.results
    assert {result.candidate_type for result in run.results} <= (
        supported_candidate_types
    )


def test_external_development_diagnostic_compares_audience_selection_ratios(
    tmp_path: Path,
) -> None:
    profiles = (
        _profile("u1", search=3, starts=2, destination_match=2),
        _profile("u2", search=2, starts=1, destination_match=1),
        _profile("u3", search=1, prices=2, destination_match=1),
        _profile("u4", search=1),
    )
    case = _case(profiles=profiles, positive_user_ids=frozenset({"u1", "u2"}))
    backtest_config = ExternalBacktestConfig(
        max_suggested_segments=3,
        min_sample_size=1,
    )
    selection_config = AudienceSelectionEvaluationConfig(
        ratios=(0.5, 1.0),
        minimum_selected_user_count=1,
        minimum_positive_capture_rate=0.5,
    )
    predictor = ContextualBookingHeuristicPredictor()

    outcomes = evaluate_external_audience_selection_cases(
        (case,),
        evaluation_key="development-fixture",
        backtest_config=backtest_config,
        selection_config=selection_config,
        performance_predictor=predictor,
    )

    assert {outcome.selection_ratio for outcome in outcomes} == {0.5, 1.0}
    limited = [
        outcome
        for outcome in outcomes
        if outcome.selection_ratio == 0.5 and outcome.policy_applied
    ]
    assert limited
    assert all(
        outcome.selected_user_count < outcome.matching_user_count
        for outcome in limited
    )

    diagnostic = build_external_audience_selection_diagnostic(
        dataset_id="fixture",
        outcome_name=case.outcome_name,
        evaluation_designs=(case.evaluation_design,),
        outcomes=outcomes,
        selection_config=selection_config,
        current_runtime_ratio=0.5,
    )
    manifest = ExternalDatasetManifest(
        dataset_id="fixture",
        source_version="fixture.v1",
        evaluation_design=case.evaluation_design,
        outcome_name=case.outcome_name,
        supports_temporal_holdout=True,
        supported_claims=("audience selection robustness",),
        unsupported_claims=("causal lift",),
        signal_mappings={},
        source_files=(),
    )
    artifacts = write_external_audience_selection_diagnostic_artifacts(
        diagnostic,
        manifest=manifest,
        model_metadata=predictor.metadata(),
        output_dir=tmp_path / "audience-selection",
    )

    payload = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    contract = payload["diagnostic"]
    assert contract["updates_model_parameters"] is False
    assert contract["updates_runtime_policy"] is False
    assert contract["accesses_sealed_final_partition"] is False
    assert contract["ratio_grid"] == [0.5, 1.0]
    assert contract["ratio_summaries"][0]["candidate_type_count"] >= 1
    assert "Rank pairwise 정확도" in artifacts["report"].read_text(
        encoding="utf-8"
    )
    assert not (artifacts["summary"].parent / "audience_selection_policy_v1.json").exists()


def test_booking_adapter_hides_last_trip_city_from_profiles(tmp_path: Path) -> None:
    source_dir = tmp_path / "booking"
    source_dir.mkdir()
    rows = [
        _booking_row("trip-1", "2016-01-01", "10"),
        _booking_row("trip-1", "2016-01-02", "10"),
        _booking_row("trip-1", "2016-01-03", "99"),
        _booking_row("trip-2", "2016-01-01", "10"),
        _booking_row("trip-2", "2016-01-02", "20"),
        _booking_row("trip-2", "2016-01-03", "10"),
        _booking_row("trip-3", "2016-01-01", "10"),
        _booking_row("trip-3", "2016-01-02", "30"),
        _booking_row("trip-3", "2016-01-03", "10"),
    ]
    _write_csv(source_dir / "train_set.csv", rows)

    bundle = load_booking_com_dataset(
        source_dir,
        config=_adapter_config(profile_pool_limit=10, max_scenarios=1),
    )

    case = bundle.cases[0]
    trip_one = next(
        profile for profile in case.profiles if profile.user_id == "booking-trip-trip-1"
    )
    assert "99" not in trip_one.destination_values
    assert "booking-trip-trip-1" not in case.positive_user_ids
    assert case.target_value == "10"
    assert bundle.manifest.evaluation_design == "within_trip_sequential_holdout"
    assert len(bundle.manifest.source_files) == 1


def test_booking_official_final_adapter_excludes_placeholder_city(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "booking-final"
    source_dir.mkdir()
    _write_csv(
        source_dir / "test_set.csv",
        [
            _booking_row("trip-1", "2016-01-01", "10"),
            _booking_row("trip-1", "2016-01-02", "20"),
            _booking_row("trip-1", "2016-01-03", "0"),
            _booking_row("trip-2", "2016-01-01", "10"),
            _booking_row("trip-2", "2016-01-02", "30"),
            _booking_row("trip-2", "2016-01-03", "0"),
        ],
    )
    _write_csv(
        source_dir / "ground_truth.csv",
        [
            {"utrip_id": "trip-1", "city_id": "10", "hotel_country": "A"},
            {"utrip_id": "trip-2", "city_id": "99", "hotel_country": "B"},
        ],
    )
    config = ExternalAdapterConfig(
        profile_pool_limit=10,
        max_scenarios=1,
        min_scenario_users=1,
        sample_modulo=1,
        sample_remainder=0,
        sample_remainders=(0,),
        evaluation_role=EXTERNAL_SEALED_FINAL_ROLE,
        include_checksum=False,
    )

    bundle = load_booking_com_final_dataset(source_dir, config=config)

    case = bundle.cases[0]
    trip_one = next(
        profile
        for profile in case.profiles
        if profile.user_id == "booking-final-trip-trip-1"
    )
    assert "0" not in trip_one.destination_values
    assert "booking-final-trip-trip-1" in case.positive_user_ids
    assert bundle.manifest.evaluation_role == EXTERNAL_SEALED_FINAL_ROLE
    assert bundle.manifest.evaluation_design == "official_test_ground_truth_holdout"


def test_airbnb_adapter_excludes_booking_actions_and_destination_label(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "airbnb"
    source_dir.mkdir()
    _write_zip_csv(
        source_dir / "train_users_2.csv.zip",
        "train_users_2.csv",
        [
            {
                "id": "u1",
                "country_destination": "US",
                "age": "31",
                "gender": "MALE",
                "affiliate_channel": "seo",
            },
            {
                "id": "u2",
                "country_destination": "NDF",
                "age": "29",
                "gender": "FEMALE",
                "affiliate_channel": "direct",
            },
        ],
    )
    _write_zip_csv(
        source_dir / "sessions.csv.zip",
        "sessions.csv",
        [
            {
                "user_id": "u1",
                "action": "search_results",
                "action_type": "click",
                "action_detail": "view_search_results",
                "device_type": "Mac Desktop",
                "secs_elapsed": "10",
            },
            {
                "user_id": "u1",
                "action": "booking",
                "action_type": "booking_request",
                "action_detail": "",
                "device_type": "Mac Desktop",
                "secs_elapsed": "20",
            },
            {
                "user_id": "u2",
                "action": "lookup",
                "action_type": "view",
                "action_detail": "p3",
                "device_type": "Windows Desktop",
                "secs_elapsed": "30",
            },
        ],
    )

    bundle = load_airbnb_dataset(
        source_dir,
        config=_adapter_config(profile_pool_limit=10, max_scenarios=1),
    )

    case = bundle.cases[0]
    user_one = next(
        profile for profile in case.profiles if profile.user_id == "airbnb-user-u1"
    )
    assert user_one.event_count == 1
    assert user_one.booking_start_count == 0
    assert user_one.destination_values == ()
    assert "airbnb-user-u1" in case.positive_user_ids
    assert bundle.manifest.supports_temporal_holdout is False


def test_synerise_adapter_keeps_future_purchase_out_of_observation_profiles(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "synerise"
    source_dir.mkdir()
    _write_parquet(
        source_dir / "search_query.parquet",
        [
            {"client_id": 1, "timestamp": "2022-11-01 00:00:00", "query": "q"},
            {"client_id": 2, "timestamp": "2022-11-01 00:00:00", "query": "q"},
        ],
    )
    _write_parquet(
        source_dir / "add_to_cart.parquet",
        [
            {"client_id": 1, "timestamp": "2022-11-02 00:00:00", "sku": 100},
            {"client_id": 2, "timestamp": "2022-11-02 00:00:00", "sku": 100},
        ],
    )
    _write_parquet(
        source_dir / "remove_from_cart.parquet",
        [{"client_id": 2, "timestamp": "2022-11-03 00:00:00", "sku": 100}],
    )
    _write_parquet(
        source_dir / "product_buy.parquet",
        [
            {"client_id": 1, "timestamp": "2022-11-12 00:00:00", "sku": 100},
            {"client_id": 2, "timestamp": "2022-11-04 00:00:00", "sku": 200},
            {"client_id": 3, "timestamp": "2022-11-12 00:00:00", "sku": 100},
        ],
    )
    _write_parquet(
        source_dir / "product_properties.parquet",
        [
            {"sku": 100, "category": 10, "price": 20, "name": "a"},
            {"sku": 200, "category": 20, "price": 30, "name": "b"},
        ],
    )

    bundle = load_synerise_dataset(
        source_dir,
        config=ExternalAdapterConfig(
            profile_pool_limit=10,
            max_scenarios=1,
            min_scenario_users=1,
            sample_modulo=1,
            sample_remainder=0,
            include_checksum=False,
            cutoff=datetime(2022, 11, 10, tzinfo=UTC),
            lookback_days=30,
            outcome_days=10,
        ),
    )

    case = bundle.cases[0]
    client_one = next(
        profile
        for profile in case.profiles
        if profile.user_id == "synerise-client-1"
    )
    assert client_one.booking_complete_count == 0
    assert "synerise-client-1" in case.positive_user_ids
    assert all(
        profile.user_id != "synerise-client-3" for profile in case.profiles
    )
    assert case.target_value == "10"
    assert bundle.manifest.supports_temporal_holdout is True


def _case(
    *,
    profiles: tuple[RawEventUserSignalRecord, ...],
    positive_user_ids: frozenset[str],
) -> ExternalEvaluationCase:
    promotion = PromotionRecord(
        project_id="fixture",
        campaign_id="campaign",
        promotion_id="promotion",
        channel="email",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.1"),
        goal_basis="all_segments",
        min_sample_size=1,
        landing_url="https://example.test",
        message_brief="target hotel discount",
    )
    intent = PromotionIntent(
        summary="target promotion",
        product="hotel",
        season=(),
        destinations=("target",),
        benefits=("discount",),
        audience_hints=(),
        channel="email",
        goal_metric="booking_conversion_rate",
        funnel_goal="booking_complete",
        desired_behaviors=("recent_destination_search", "price_sensitive"),
        explicit_conditions=("target", "discount"),
        source="test",
    )
    return ExternalEvaluationCase(
        dataset_id="fixture",
        scenario_id="fixture-target",
        target_value="target",
        target_label="target",
        outcome_name="future_target_rate",
        evaluation_design="temporal_holdout",
        profiles=profiles,
        positive_user_ids=positive_user_ids,
        promotion=promotion,
        intent=intent,
    )


def _profile(
    user_id: str,
    *,
    search: int = 0,
    starts: int = 0,
    completes: int = 0,
    prices: int = 0,
    destination_match: int = 0,
) -> RawEventUserSignalRecord:
    return RawEventUserSignalRecord(
        project_id="fixture",
        user_id=user_id,
        event_count=search + starts + completes + prices,
        hotel_search_count=search,
        hotel_click_count=0,
        hotel_detail_view_count=0,
        promotion_impression_count=0,
        promotion_click_count=0,
        campaign_redirect_click_count=0,
        campaign_landing_count=0,
        booking_start_count=starts,
        booking_complete_count=completes,
        booking_cancel_count=0,
        deal_event_count=0,
        free_cancellation_count=0,
        breakfast_included_count=0,
        price_event_count=prices,
        avg_price=10.0 if prices else 0.0,
        destination_values=("target",) if destination_match else (),
        checkin_dates=(),
        hotel_market_values=(),
        hotel_cluster_values=(),
        age_group_values=(),
        gender_values=(),
        preferred_category_values=(),
        destination_match_count=destination_match,
        season_match_count=0,
    )


def _adapter_config(
    *,
    profile_pool_limit: int,
    max_scenarios: int,
) -> ExternalAdapterConfig:
    return ExternalAdapterConfig(
        profile_pool_limit=profile_pool_limit,
        max_scenarios=max_scenarios,
        min_scenario_users=1,
        sample_modulo=1,
        sample_remainder=0,
        include_checksum=False,
    )


def _booking_row(trip_id: str, checkin: str, city_id: str) -> dict[str, str]:
    return {
        "user_id": trip_id,
        "checkin": checkin,
        "checkout": checkin,
        "city_id": city_id,
        "device_class": "desktop",
        "affiliate_id": "1",
        "booker_country": "A",
        "hotel_country": "B",
        "utrip_id": trip_id,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_zip_csv(
    path: Path,
    member: str,
    rows: list[dict[str, str]],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, buffer.getvalue())


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)
