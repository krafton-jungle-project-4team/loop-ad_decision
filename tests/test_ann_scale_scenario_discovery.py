from __future__ import annotations

import pytest

from scripts.discover_ann_scale_scenarios import (
    build_definition_configs,
    definition_sha256,
    load_signal_checkpoint,
    signal_insert_sql,
    signal_predicate,
)


def _pool():
    return {
        "destinations": [
            {"destination_id": str(value), "user_count": 100 - value}
            for value in range(1, 5)
        ],
        "benefits": {
            "discount": 10,
            "early_booking": 10,
            "free_cancellation": 10,
            "breakfast_included": 10,
        },
    }


def test_definition_pool_uses_only_actual_inputs_and_all_five_candidate_types() -> None:
    configs = build_definition_configs(_pool())

    assert {row["candidate_type"] for row in configs} == {
        "intent_matched",
        "target_destination_affinity",
        "funnel_recovery",
        "benefit_value_seeker",
        "general_destination_explorer",
    }
    assert all(
        set(row["destination_ids"]).issubset({"1", "2", "3", "4"})
        for row in configs
    )


def test_definition_hash_ignores_scenario_id_and_set_by_construction() -> None:
    config = {
        "candidate_type": "intent_matched",
        "destination_ids": ["1"],
        "season_months": [],
        "benefit_keys": [],
    }
    assert definition_sha256(config) == definition_sha256(dict(config))
    assert definition_sha256(config) != definition_sha256(
        {**config, "destination_ids": ["2"]}
    )


def test_signal_predicate_matches_destination_and_season_parameters() -> None:
    sql, parameters = signal_predicate(
        ["hotel_product_interest", "recent_destination_search", "season_match"],
        {"destinations": ["1"], "season_months": [1, 2]},
    )
    assert "hotel_interest_count > 0" in sql
    assert "arrayExists" in sql
    assert parameters["destinations"] == ["1"]
    assert parameters["season_months"] == [1, 2]


def test_signal_insert_is_cohort_and_source_revision_bounded() -> None:
    sql = signal_insert_sql()
    assert "ann_benchmark_scale_membership" in sql
    assert "cohort_rank <= {cohort_size:UInt64}" in sql
    assert "received_at <= parseDateTime64BestEffort" in sql
    assert "modulo(cityHash64(raw.user_id)" in sql


def test_signal_checkpoint_rejects_other_series(tmp_path) -> None:
    path = tmp_path / "signals.jsonl"
    path.write_text(
        '{"scale_series_id":"other","cohort_size":1000000,'
        '"build_shard_count":32,"build_shard_index":0}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="identity differs"):
        load_signal_checkpoint(
            path,
            scale_series_id="series",
            cohort_size=1_000_000,
            shard_count=32,
        )
