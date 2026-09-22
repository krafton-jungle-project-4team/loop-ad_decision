#!/usr/bin/env python3
"""Discover production-valid ANN scale scenarios from frozen Expedia data."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audience_contract import SEGMENT_AUDIENCE_CONTRACT  # noqa: E402
from app.analysis.behavior_manifest import (  # noqa: E402
    clickhouse_canonical_destination_sql,
    load_behavior_manifest,
)
from app.analysis.segment_audience_templates import (  # noqa: E402
    RegisteredSegmentAudienceBinder,
)
from app.analysis.semantic_selection import (  # noqa: E402
    compile_registered_segment_audience,
)
from app.config import load_settings  # noqa: E402
from app.db import create_clickhouse_client  # noqa: E402
from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    AppendOnlyJsonl,
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_EXPERIMENT_VERSION,
    ScenarioCensusRecord,
    ScenarioSet,
    cohort_signal_predicate,
    select_scenario_pairs,
)


SIGNAL_RELATION = "ann_benchmark_user_signals"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--vector-version", default="hotel_behavior.v2")
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--scale-series-id", required=True)
    parser.add_argument("--cohort-size", type=positive_int, default=1_000_000)
    parser.add_argument("--build-shard-count", type=positive_int, default=32)
    parser.add_argument("--reference-sample-seed", required=True)
    parser.add_argument("--reference-sample-size", type=positive_int, default=50_000)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--source-revision-cutoff", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-local-clickhouse-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_local_clickhouse_write:
        raise ValueError("scenario discovery requires --confirm-local-clickhouse-write")
    load_dotenv(args.env_file, override=False)
    settings = load_settings()
    host = (urlparse(settings.clickhouse_url).hostname or "").lower()
    if host not in LOCAL_HOSTS:
        raise ValueError("scenario discovery only accepts local ClickHouse")
    window_start = parse_utc(args.window_start)
    window_end = parse_utc(args.window_end)
    revision_cutoff = parse_utc(args.source_revision_cutoff)
    client = create_clickhouse_client(settings)
    try:
        prepare_signals(
            client,
            args=args,
            window_start=window_start,
            window_end=window_end,
            revision_cutoff=revision_cutoff,
        )
        pool = actual_input_pool(client, args=args)
        write_immutable_json(args.output_dir / "actual-input-pool.json", pool)
        configs = build_definition_configs(pool)
        definitions, scenarios = compile_definitions(
            configs,
            project_id=args.project_id,
            vector_version=args.vector_version,
            manifest_hash=args.manifest_hash,
        )
        write_immutable_json(
            args.output_dir / "scenario-definitions.json",
            {
                "experiment_version": SCALE_EXPERIMENT_VERSION,
                "source": "frozen_expedia_actual_inputs",
                "scenarios": definitions,
            },
        )
        write_immutable_json(
            args.output_dir / "scenario-manifest.json",
            {
                "experiment_version": SCALE_EXPERIMENT_VERSION,
                "project_id": args.project_id,
                "vector_version": args.vector_version,
                "manifest_hash": args.manifest_hash,
                "scenarios": scenarios,
            },
        )
        records = census_scenarios(
            client,
            args=args,
            scenarios=scenarios,
        )
        pairs = select_scenario_pairs(records)
        write_immutable_json(
            args.output_dir / "scenario-pairs.json",
            {
                "experiment_version": SCALE_EXPERIMENT_VERSION,
                "census_cohort_size": args.cohort_size,
                "pairs": [pair.to_dict() for pair in pairs],
                "validated_pair_count": sum(pair.validated for pair in pairs),
                "fallback_count": sum(not pair.validated for pair in pairs),
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "signal_user_count": pool["signal_user_count"],
                    "logical_definition_count": len(configs),
                    "compiled_scenario_count": len(scenarios),
                    "census_record_count": len(records),
                    "validated_pair_count": sum(pair.validated for pair in pairs),
                    "fallback_count": sum(not pair.validated for pair in pairs),
                },
                indent=2,
            )
        )
        return 0
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def prepare_signals(
    client: Any,
    *,
    args: argparse.Namespace,
    window_start: datetime,
    window_end: datetime,
    revision_cutoff: datetime,
) -> None:
    client.command(signal_create_sql())
    checkpoint_path = args.output_dir / "signal-build-shards.jsonl"
    checkpoint = AppendOnlyJsonl(
        checkpoint_path,
        identity_fields=("build_shard_count", "build_shard_index"),
    )
    completed = load_signal_checkpoint(
        checkpoint_path,
        scale_series_id=args.scale_series_id,
        cohort_size=args.cohort_size,
        shard_count=args.build_shard_count,
    )
    base_parameters = {
        "scale_series_id": args.scale_series_id,
        "cohort_size": args.cohort_size,
        "build_shard_count": args.build_shard_count,
        "project_id": args.project_id,
        "vector_version": args.vector_version,
        "window_start": clickhouse_time(window_start),
        "window_end": clickhouse_time(window_end),
        "source_revision_cutoff": revision_cutoff.isoformat(),
    }
    for shard_index in range(args.build_shard_count):
        parameters = {**base_parameters, "build_shard_index": shard_index}
        expected = scalar(
            client.query(signal_expected_count_sql(), parameters=parameters)
        )
        observed = scalar(
            client.query(signal_actual_count_sql(), parameters=parameters)
        )
        if shard_index in completed:
            if observed != expected:
                raise RuntimeError("completed signal shard differs from membership")
            continue
        status = "validated_existing_after_interruption"
        if observed == 0:
            client.command(
                signal_insert_sql(),
                parameters=parameters,
                settings={"max_threads": 4},
            )
            observed = scalar(
                client.query(signal_actual_count_sql(), parameters=parameters)
            )
            status = "insert_command_succeeded"
        if observed != expected:
            raise RuntimeError(
                f"signal shard {shard_index} expected {expected}, observed {observed}"
            )
        checkpoint.append(
            (
                {
                    "experiment_version": SCALE_EXPERIMENT_VERSION,
                    "scale_series_id": args.scale_series_id,
                    "cohort_size": args.cohort_size,
                    "build_shard_count": args.build_shard_count,
                    "build_shard_index": shard_index,
                    "user_count": observed,
                    "status": status,
                },
            )
        )
        print(
            json.dumps(
                {
                    "signal_build_shard": shard_index,
                    "user_count": observed,
                    "status": status,
                }
            ),
            flush=True,
        )
    total = scalar(
        client.query(
            f"SELECT count() FROM {SIGNAL_RELATION} "
            "WHERE scale_series_id={scale_series_id:String}",
            parameters={"scale_series_id": args.scale_series_id},
        )
    )
    if total != args.cohort_size:
        raise RuntimeError("signal relation is not complete for the census cohort")


def signal_create_sql() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {SIGNAL_RELATION} (
        scale_series_id String,
        cohort_rank UInt64,
        user_id String,
        hotel_interest_count UInt64,
        booking_start_count UInt64,
        booking_complete_count UInt64,
        promotion_response_count UInt64,
        destination_count UInt64,
        destination_values Array(String),
        checkin_months Array(UInt8),
        deal_count UInt64,
        price_count UInt64,
        free_cancellation_count UInt64,
        breakfast_count UInt64,
        loaded_at DateTime64(6, 'UTC') DEFAULT now64(6)
    )
    ENGINE = MergeTree
    ORDER BY (scale_series_id, cohort_rank, user_id)
    """


def signal_expected_count_sql() -> str:
    return """
    SELECT count()
    FROM ann_benchmark_scale_membership
    WHERE scale_series_id = {scale_series_id:String}
      AND cohort_rank <= {cohort_size:UInt64}
      AND modulo(cityHash64(user_id), {build_shard_count:UInt64})
          = {build_shard_index:UInt64}
    """


def signal_actual_count_sql() -> str:
    return f"""
    SELECT count()
    FROM {SIGNAL_RELATION}
    WHERE scale_series_id = {{scale_series_id:String}}
      AND cohort_rank <= {{cohort_size:UInt64}}
      AND modulo(cityHash64(user_id), {{build_shard_count:UInt64}})
          = {{build_shard_index:UInt64}}
    """


def signal_insert_sql() -> str:
    destination = clickhouse_canonical_destination_sql(
        """
        coalesce(
            nullIf(JSONExtractString(properties_json, 'destination_id'), ''),
            nullIf(JSONExtractString(properties_json, 'destination_name'), ''),
            nullIf(JSONExtractString(properties_json, 'hotel_city'), ''),
            ''
        )
        """.strip()
    )
    return f"""
    INSERT INTO {SIGNAL_RELATION} (
        scale_series_id, cohort_rank, user_id, hotel_interest_count,
        booking_start_count, booking_complete_count, promotion_response_count,
        destination_count, destination_values, checkin_months, deal_count,
        price_count, free_cancellation_count, breakfast_count
    )
    SELECT
        {{scale_series_id:String}},
        any(membership.cohort_rank),
        raw.user_id,
        countIf(raw.event_name IN (
            'hotel_search','hotel_click','hotel_detail_view'
        )),
        countIf(raw.event_name = 'booking_start'),
        countIf(raw.event_name = 'booking_complete'),
        countIf(raw.event_name IN ('promotion_click','campaign_landing')),
        uniqExactIf(
            {destination},
            raw.event_name IN ('hotel_search','hotel_click','hotel_detail_view')
            AND {destination} != ''
        ),
        groupArrayIf({destination}, {destination} != ''),
        groupArrayIf(
            toUInt8(toMonth(parseDateTimeBestEffortOrNull(
                JSONExtractString(raw.properties_json, 'checkin_date')
            ))),
            parseDateTimeBestEffortOrNull(
                JSONExtractString(raw.properties_json, 'checkin_date')
            ) IS NOT NULL
        ),
        countIf(toUInt8OrZero(JSONExtractString(
            raw.properties_json, 'deal'
        )) = 1),
        countIf(nullIf(JSONExtractString(
            raw.properties_json, 'price'
        ), '') IS NOT NULL),
        countIf(toUInt8OrZero(JSONExtractString(
            raw.properties_json, 'free_cancellation'
        )) = 1),
        countIf(toUInt8OrZero(JSONExtractString(
            raw.properties_json, 'breakfast_included'
        )) = 1)
    FROM raw_events AS raw
    INNER JOIN ann_benchmark_scale_membership AS membership
        ON membership.scale_series_id = {{scale_series_id:String}}
       AND membership.cohort_rank <= {{cohort_size:UInt64}}
       AND membership.user_id = raw.user_id
    WHERE raw.project_id = {{project_id:String}}
      AND raw.validation_status = 'valid'
      AND raw.received_at <= parseDateTime64BestEffort(
          {{source_revision_cutoff:String}}, 6, 'UTC'
      )
      AND raw.event_time >= toDateTime64(
          parseDateTimeBestEffort({{window_start:String}}), 3, 'UTC'
      )
      AND raw.event_time < toDateTime64(
          parseDateTimeBestEffort({{window_end:String}}), 3, 'UTC'
      )
      AND modulo(cityHash64(raw.user_id), {{build_shard_count:UInt64}})
          = {{build_shard_index:UInt64}}
    GROUP BY raw.user_id
    """


def load_signal_checkpoint(
    path: Path,
    *,
    scale_series_id: str,
    cohort_size: int,
    shard_count: int,
) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise RuntimeError(f"blank signal checkpoint row {line_number}")
        row = json.loads(line)
        if (
            row.get("scale_series_id") != scale_series_id
            or int(row.get("cohort_size", -1)) != cohort_size
            or int(row.get("build_shard_count", -1)) != shard_count
        ):
            raise RuntimeError("signal checkpoint identity differs from this run")
        shard = int(row["build_shard_index"])
        if shard in completed:
            raise RuntimeError("signal checkpoint contains duplicate shard")
        completed.add(shard)
    return completed


def actual_input_pool(client: Any, *, args: argparse.Namespace) -> Mapping[str, Any]:
    destinations = named_rows(
        client.query(
            f"""
            SELECT destination_id, uniqExact(user_id) AS user_count
            FROM
            (
                SELECT
                    user_id,
                    arrayJoin(arrayDistinct(destination_values)) AS destination_id
                FROM {SIGNAL_RELATION}
                WHERE scale_series_id = {{scale_series_id:String}}
            )
            WHERE destination_id != ''
            GROUP BY destination_id
            ORDER BY user_count DESC, destination_id
            LIMIT 16
            """,
            parameters={"scale_series_id": args.scale_series_id},
        )
    )
    months = named_rows(
        client.query(
            f"""
            SELECT month, uniqExact(user_id) AS user_count
            FROM
            (
                SELECT user_id, arrayJoin(arrayDistinct(checkin_months)) AS month
                FROM {SIGNAL_RELATION}
                WHERE scale_series_id = {{scale_series_id:String}}
            )
            GROUP BY month
            ORDER BY month
            """,
            parameters={"scale_series_id": args.scale_series_id},
        )
    )
    benefits = named_rows(
        client.query(
            f"""
            SELECT
                countIf(deal_count + price_count > 0) AS discount,
                countIf(deal_count + price_count > 0) AS early_booking,
                countIf(free_cancellation_count > 0) AS free_cancellation,
                countIf(breakfast_count > 0) AS breakfast_included
            FROM {SIGNAL_RELATION}
            WHERE scale_series_id = {{scale_series_id:String}}
            """,
            parameters={"scale_series_id": args.scale_series_id},
        )
    )
    signal_user_count = scalar(
        client.query(
            f"SELECT count() FROM {SIGNAL_RELATION} "
            "WHERE scale_series_id={scale_series_id:String}",
            parameters={"scale_series_id": args.scale_series_id},
        )
    )
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "scale_series_id": args.scale_series_id,
        "cohort_size": args.cohort_size,
        "signal_user_count": signal_user_count,
        "destinations": [
            {
                "destination_id": str(row["destination_id"]),
                "user_count": int(row["user_count"]),
            }
            for row in destinations
        ],
        "months": [
            {"month": int(row["month"]), "user_count": int(row["user_count"])}
            for row in months
        ],
        "benefits": {
            key: int(value)
            for key, value in (benefits[0] if benefits else {}).items()
        },
    }


def build_definition_configs(pool: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    destinations = [
        str(row["destination_id"])
        for row in pool["destinations"][:4]
    ]
    if len(destinations) < 4:
        raise RuntimeError("fewer than four actual destinations are available")
    manifest = load_behavior_manifest()
    supported_benefits = [
        str(value) for value in manifest["intent_benefit_query_dimensions"]
        if int(pool["benefits"].get(str(value), 0)) > 0
    ]
    season_groups = ([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
    configs: list[Mapping[str, Any]] = []

    def add(candidate_type: str, *, destinations_: Sequence[str] = (),
            months_: Sequence[int] = (), benefits_: Sequence[str] = ()) -> None:
        config = {
            "candidate_type": candidate_type,
            "destination_ids": list(destinations_),
            "season_months": list(months_),
            "benefit_keys": list(benefits_),
        }
        if config not in configs:
            configs.append(config)

    add("intent_matched")
    for destination in destinations:
        add("intent_matched", destinations_=[destination])
    add("intent_matched", destinations_=destinations[:2])
    add("intent_matched", destinations_=destinations)
    for months in season_groups:
        add("intent_matched", months_=months)
    add("intent_matched", destinations_=[destinations[0]], months_=season_groups[0])
    add("intent_matched", destinations_=[destinations[1]], months_=season_groups[1])

    for destination in destinations:
        add("target_destination_affinity", destinations_=[destination])
    add("target_destination_affinity", destinations_=destinations[:2])
    add("target_destination_affinity", destinations_=destinations)

    add("funnel_recovery")
    for destination in destinations:
        add("funnel_recovery", destinations_=[destination])
    add("funnel_recovery", destinations_=destinations[:2])
    add("funnel_recovery", destinations_=destinations)

    add("benefit_value_seeker")
    for benefit in supported_benefits:
        add("benefit_value_seeker", benefits_=[benefit])
    for destination in destinations[:2]:
        add("benefit_value_seeker", destinations_=[destination])
    for destination, benefit in zip(destinations, supported_benefits, strict=False):
        add(
            "benefit_value_seeker",
            destinations_=[destination],
            benefits_=[benefit],
        )
    add("general_destination_explorer")
    return tuple(configs)


def compile_definitions(
    configs: Sequence[Mapping[str, Any]],
    *,
    project_id: str,
    vector_version: str,
    manifest_hash: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    binder = RegisteredSegmentAudienceBinder()
    definitions: list[Mapping[str, Any]] = []
    scenarios: list[Mapping[str, Any]] = []
    for index, config in enumerate(configs, start=1):
        definition_hash = definition_sha256(config)
        for scenario_set in ("tuning", "confirmation"):
            scenario_id = (
                f"v2-{str(config['candidate_type']).replace('_', '-')}-{index:02d}-"
                f"{'t' if scenario_set == 'tuning' else 'c'}"
            )
            binding = binder.bind(
                candidate_type=str(config["candidate_type"]),
                destination_ids=[str(value) for value in config["destination_ids"]],
                season_months=[int(value) for value in config["season_months"]],
                benefit_keys=[str(value) for value in config["benefit_keys"]],
            )
            compiled = compile_registered_segment_audience(
                segment_id=scenario_id,
                rule_json={
                    "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
                    "segment_audience_spec": dict(binding),
                },
            )
            if compiled.manifest_hash != manifest_hash:
                raise RuntimeError("production compiler manifest differs from vectors")
            definition = {
                "scenario_id": scenario_id,
                "scenario_set": scenario_set,
                **dict(config),
                "definition_sha256": definition_hash,
            }
            definitions.append(definition)
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_set": scenario_set,
                    "candidate_type": str(config["candidate_type"]),
                    "query_vector": list(compiled.query_vector),
                    "score_threshold": compiled.score_threshold,
                    "hard_predicate_keys": list(compiled.hard_predicate_keys),
                    "predicate_parameters": {
                        key: list(value)
                        for key, value in compiled.predicate_parameters.items()
                    },
                    "compiler_provenance": {
                        "manifest_hash": compiled.manifest_hash,
                        "calibration_version": compiled.calibration_version,
                        "calibration_hash": compiled.calibration_hash,
                        "query_compiler_version": compiled.query_compiler_version,
                        "query_compiler_hash": compiled.query_compiler_hash,
                        "template_id": compiled.template_id,
                        "template_semantic_hash": compiled.template_semantic_hash,
                    },
                    "definition_sha256": definition_hash,
                    "project_id": project_id,
                    "vector_version": vector_version,
                }
            )
    return definitions, scenarios


def census_scenarios(
    client: Any,
    *,
    args: argparse.Namespace,
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[ScenarioCensusRecord, ...]:
    path = args.output_dir / "scenario-census.jsonl"
    output = AppendOnlyJsonl(path, identity_fields=("scenario_id",))
    unique: dict[str, Mapping[str, Any]] = {}
    for scenario in scenarios:
        unique.setdefault(str(scenario["definition_sha256"]), scenario)
    measured: dict[str, tuple[int, int, int, int]] = {}
    for definition_hash, scenario in unique.items():
        predicate, parameters = cohort_signal_predicate(
            scenario["hard_predicate_keys"],
            scenario["predicate_parameters"],
        )
        common = {
            "scale_series_id": args.scale_series_id,
            "project_id": args.project_id,
            "vector_version": args.vector_version,
            "cohort_size": args.cohort_size,
            "query_vector": [float(value) for value in scenario["query_vector"]],
            "score_threshold": float(scenario["score_threshold"]),
            **parameters,
        }
        full = named_rows(
            client.query(
                full_census_sql(predicate),
                parameters=common,
                settings={"max_threads": 4},
            )
        )[0]
        sample = named_rows(
            client.query(
                sample_census_sql(predicate),
                parameters={
                    **common,
                    "reference_sample_seed": args.reference_sample_seed,
                    "reference_sample_size": args.reference_sample_size,
                },
                settings={"max_threads": 4},
            )
        )[0]
        measured[definition_hash] = (
            int(full["hard_count"]),
            int(full["exact_positive_count"]),
            int(sample["sampled_count"]),
            int(sample["passed_count"]),
        )
        print(
            json.dumps(
                {
                    "census_definition": definition_hash[:12],
                    "candidate_type": scenario["candidate_type"],
                    "hard_count": int(full["hard_count"]),
                    "exact_positive_count": int(full["exact_positive_count"]),
                }
            ),
            flush=True,
        )

    rows: list[Mapping[str, Any]] = []
    records: list[ScenarioCensusRecord] = []
    for scenario in scenarios:
        definition_hash = str(scenario["definition_sha256"])
        hard_count, exact_positive, sampled_count, passed_count = measured[
            definition_hash
        ]
        pass_rate = passed_count / sampled_count if sampled_count else 0.0
        record = ScenarioCensusRecord(
            scenario_id=str(scenario["scenario_id"]),
            definition_sha256=definition_hash,
            candidate_type=str(scenario["candidate_type"]),
            scenario_set=ScenarioSet(str(scenario["scenario_set"])),
            corpus_user_count=args.cohort_size,
            hard_match_user_count=hard_count,
            estimated_member_count=hard_count * pass_rate,
            exact_positive_count=exact_positive,
        )
        records.append(record)
        rows.append(
            {
                **record.to_dict(),
                "sampled_count": sampled_count,
                "sample_passed_count": passed_count,
                "estimated_score_pass_rate": pass_rate,
            }
        )
    output.append(rows)
    return tuple(records)


def signal_predicate(
    keys: Sequence[str],
    raw_parameters: Mapping[str, Sequence[Any]],
) -> tuple[str, Mapping[str, Any]]:
    """Backward-compatible entry point for discovery unit tests."""
    return cohort_signal_predicate(keys, raw_parameters)


def full_census_sql(predicate: str) -> str:
    return f"""
    SELECT
        countIf({predicate}) AS hard_count,
        countIf(
            ({predicate}) AND
            1 - cosineDistance(
                vectors.vector_values,
                {{query_vector:Array(Float32)}}
            ) >= {{score_threshold:Float64}}
        ) AS exact_positive_count
    FROM {SIGNAL_RELATION} AS signals
    INNER JOIN user_behavior_vectors AS vectors USING (user_id)
    WHERE signals.scale_series_id = {{scale_series_id:String}}
      AND signals.cohort_rank <= {{cohort_size:UInt64}}
      AND vectors.project_id = {{project_id:String}}
      AND vectors.vector_version = {{vector_version:String}}
    """


def sample_census_sql(predicate: str) -> str:
    return f"""
    SELECT
        count() AS sampled_count,
        countIf(score >= {{score_threshold:Float64}}) AS passed_count
    FROM
    (
        SELECT
            1 - cosineDistance(
                vectors.vector_values,
                {{query_vector:Array(Float32)}}
            ) AS score
        FROM {SIGNAL_RELATION} AS signals
        INNER JOIN user_behavior_vectors AS vectors USING (user_id)
        WHERE signals.scale_series_id = {{scale_series_id:String}}
          AND signals.cohort_rank <= {{cohort_size:UInt64}}
          AND vectors.project_id = {{project_id:String}}
          AND vectors.vector_version = {{vector_version:String}}
          AND ({predicate})
        ORDER BY
            hex(SHA256(concat(
                {{reference_sample_seed:String}}, '|', signals.user_id
            ))),
            signals.user_id
        LIMIT {{reference_sample_size:UInt32}}
    )
    """


def definition_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def named_rows(result: Any) -> list[Mapping[str, Any]]:
    return list(result.named_results())


def scalar(result: Any) -> int:
    rows = getattr(result, "result_rows", None)
    if not rows:
        return 0
    return int(rows[0][0])


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(UTC)


def clickhouse_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
