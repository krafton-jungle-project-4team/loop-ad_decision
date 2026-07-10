from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from app.analysis.raw_event_segments import (
    PromotionIntent,
    compile_raw_event_intent,
    generate_raw_event_segment_definitions,
)
from app.analysis.repositories import PromotionRecord, RawEventUserSignalRecord
from app.analysis.segment_performance import (
    MODEL_CANDIDATE_TYPES,
    CalibrationTrainingExample,
    LogisticSegmentPerformanceModel,
    SegmentPerformanceFeatures,
    SegmentPerformancePredictor,
    fit_logistic_segment_performance_model,
    write_segment_performance_model,
)
from app.logging import duration_ms, log, log_context_scope, now_ms


EXPEDIA_TRAIN_COLUMNS = (
    "date_time",
    "site_name",
    "posa_continent",
    "user_location_country",
    "user_location_region",
    "user_location_city",
    "orig_destination_distance",
    "user_id",
    "is_mobile",
    "is_package",
    "channel",
    "srch_ci",
    "srch_co",
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_rm_cnt",
    "srch_destination_id",
    "srch_destination_type_id",
    "is_booking",
    "cnt",
    "hotel_continent",
    "hotel_country",
    "hotel_market",
    "hotel_cluster",
)

SEASON_MONTHS: Mapping[str, tuple[int, ...]] = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "fall": (9, 10, 11),
    "winter": (12, 1, 2),
}

_TABLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPEDIA_USER_PREFIX = "expedia-user-"


class ExpediaBacktestError(RuntimeError):
    pass


class ClickHouseQueryClient(Protocol):
    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        ...

    def command(
        self,
        command: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        ...

    def raw_insert(
        self,
        table: str,
        column_names: Sequence[str] | None = None,
        insert_block: Any = None,
        settings: Mapping[str, Any] | None = None,
        fmt: str | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class ExpediaSourceStats:
    row_count: int
    user_count: int
    booking_row_count: int
    first_event_at: datetime | None
    last_event_at: datetime | None

    @property
    def booking_row_rate(self) -> float:
        return _safe_rate(self.booking_row_count, self.row_count)


@dataclass(frozen=True, slots=True)
class ExpediaBacktestScenario:
    scenario_id: str
    cutoff: datetime
    target_destination_id: int
    historical_user_count: int
    historical_event_count: int
    season: str | None = None

    @property
    def season_months(self) -> tuple[int, ...]:
        if self.season is None:
            return ()
        return SEASON_MONTHS[self.season]


@dataclass(frozen=True, slots=True)
class ExpediaFutureBookingUsers:
    any_booking_user_ids: frozenset[str]
    contextual_booking_user_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ExpediaBacktestConfig:
    lookback_days: int = 90
    outcome_days: int = 30
    max_scenarios_per_cutoff: int = 3
    max_suggested_segments: int = 3
    min_sample_size: int = 2
    profile_pool_limit: int = 1000
    min_scenario_users: int = 20
    user_sample_modulo: int = 20
    user_sample_remainder: int = 0
    season: str | None = None
    excluded_destination_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.lookback_days <= 0 or self.outcome_days <= 0:
            raise ValueError("lookback_days and outcome_days must be positive")
        if self.max_scenarios_per_cutoff <= 0:
            raise ValueError("max_scenarios_per_cutoff must be positive")
        if self.max_suggested_segments <= 0:
            raise ValueError("max_suggested_segments must be positive")
        if self.min_sample_size <= 0 or self.profile_pool_limit <= 0:
            raise ValueError("sample and profile limits must be positive")
        if self.user_sample_modulo <= 0:
            raise ValueError("user_sample_modulo must be positive")
        if not 0 <= self.user_sample_remainder < self.user_sample_modulo:
            raise ValueError(
                "user_sample_remainder must be smaller than user_sample_modulo"
            )
        if self.season is not None and self.season not in SEASON_MONTHS:
            raise ValueError(f"unsupported season: {self.season}")
        if any(value <= 0 for value in self.excluded_destination_ids):
            raise ValueError("excluded destination IDs must be positive")
        if len(set(self.excluded_destination_ids)) != len(
            self.excluded_destination_ids
        ):
            raise ValueError("excluded destination IDs must be unique")


@dataclass(frozen=True, slots=True)
class ExpediaBacktestResult:
    cutoff: datetime
    observation_start: datetime
    outcome_end: datetime
    scenario_id: str
    target_destination_id: int
    season: str | None
    rank: int
    segment_id: str
    candidate_type: str
    rank_role: str
    sample_size: int
    total_eligible_user_count: int
    matching_profile_count: int
    performance_features: Mapping[str, Any]
    prediction_method: str
    prediction_model_version: str
    predicted_conversion_rate: float
    actual_contextual_conversion_rate: float
    actual_any_conversion_rate: float
    baseline_contextual_conversion_rate: float
    baseline_any_conversion_rate: float
    contextual_booking_user_count: int
    any_booking_user_count: int
    absolute_lift_percentage_points: float
    relative_lift: float | None
    calibration_error_percentage_points: float
    recommendation_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cutoff"] = self.cutoff.isoformat()
        payload["observation_start"] = self.observation_start.isoformat()
        payload["outcome_end"] = self.outcome_end.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class ExpediaBacktestSkippedScenario:
    cutoff: datetime
    scenario_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "cutoff": self.cutoff.isoformat(),
            "scenario_id": self.scenario_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ExpediaBacktestRun:
    results: tuple[ExpediaBacktestResult, ...]
    skipped_scenarios: tuple[ExpediaBacktestSkippedScenario, ...]


@dataclass(frozen=True, slots=True)
class ExpediaTemporalHoldoutRun:
    training_run: ExpediaBacktestRun
    holdout_run: ExpediaBacktestRun
    calibration_model: LogisticSegmentPerformanceModel


class ExpediaBacktestRepository(Protocol):
    def source_stats(self) -> ExpediaSourceStats:
        ...

    def source_checksum(self) -> str:
        ...

    def list_scenarios(
        self,
        *,
        observation_start: datetime,
        cutoff: datetime,
        limit: int,
        min_users: int,
        user_sample_modulo: int,
        user_sample_remainder: int,
        season: str | None,
        excluded_destination_ids: Sequence[int],
    ) -> list[ExpediaBacktestScenario]:
        ...

    def list_user_profiles(
        self,
        *,
        scenario: ExpediaBacktestScenario,
        observation_start: datetime,
        cutoff: datetime,
        limit: int,
        user_sample_modulo: int,
        user_sample_remainder: int,
    ) -> list[RawEventUserSignalRecord]:
        ...

    def future_booking_users(
        self,
        *,
        scenario: ExpediaBacktestScenario,
        outcome_end: datetime,
        eligible_user_ids: Sequence[str],
    ) -> ExpediaFutureBookingUsers:
        ...


class ClickHouseExpediaBacktestRepository:
    def __init__(
        self,
        client: ClickHouseQueryClient,
        *,
        source_table: str = "expedia_hotel_events",
        project_id: str = "expedia_backtest",
    ) -> None:
        self._client = client
        self._source_table = validate_table_identifier(source_table)
        self._project_id = project_id

    def ensure_source_table(self) -> None:
        self._client.command(_create_source_table_sql(self._source_table))

    def load_train_csv(self, train_csv: Path, *, replace: bool = False) -> int:
        validate_train_csv_header(train_csv)
        self.ensure_source_table()
        current_count = self.source_stats().row_count
        if current_count > 0 and not replace:
            return current_count
        if current_count > 0:
            self._client.command(f"TRUNCATE TABLE {self._source_table}")
        with train_csv.open("rb") as source:
            self._client.raw_insert(
                self._source_table,
                insert_block=source,
                fmt="CSVWithNames",
                settings={
                    "input_format_csv_empty_as_default": 1,
                    "date_time_input_format": "best_effort",
                },
            )
        return self.source_stats().row_count

    def source_stats(self) -> ExpediaSourceStats:
        result = self._client.query(
            f"""
            SELECT
                count() AS row_count,
                countDistinct(user_id) AS user_count,
                countIf(is_booking = 1) AS booking_row_count,
                minOrNull(date_time) AS first_event_at,
                maxOrNull(date_time) AS last_event_at
            FROM {self._source_table}
            """
        )
        rows = _clickhouse_rows(result)
        if not rows:
            return ExpediaSourceStats(0, 0, 0, None, None)
        row = rows[0]
        return ExpediaSourceStats(
            row_count=int(_row_value(row, "row_count", 0) or 0),
            user_count=int(_row_value(row, "user_count", 1) or 0),
            booking_row_count=int(_row_value(row, "booking_row_count", 2) or 0),
            first_event_at=_as_utc_datetime(_row_value(row, "first_event_at", 3)),
            last_event_at=_as_utc_datetime(_row_value(row, "last_event_at", 4)),
        )

    def source_checksum(self) -> str:
        row_hash = "cityHash64(concatWithSeparator('|', " + ", ".join(
            (
                "toString(date_time)",
                "toString(user_id)",
                "toString(srch_destination_id)",
                "toString(is_booking)",
                "toString(cnt)",
                "toString(hotel_market)",
                "toString(hotel_cluster)",
            )
        ) + "))"
        result = self._client.query(
            f"""
            SELECT
                toString(sumWithOverflow({row_hash})) AS checksum_sum,
                toString(groupBitXor({row_hash})) AS checksum_xor
            FROM {self._source_table}
            """
        )
        rows = _clickhouse_rows(result)
        if not rows:
            return "0:0"
        row = rows[0]
        return (
            f"{_row_value(row, 'checksum_sum', 0) or '0'}:"
            f"{_row_value(row, 'checksum_xor', 1) or '0'}"
        )

    def list_scenarios(
        self,
        *,
        observation_start: datetime,
        cutoff: datetime,
        limit: int,
        min_users: int,
        user_sample_modulo: int,
        user_sample_remainder: int,
        season: str | None,
        excluded_destination_ids: Sequence[int] = (),
    ) -> list[ExpediaBacktestScenario]:
        result = self._client.query(
            f"""
            SELECT
                toUInt64(srch_destination_id) AS target_destination_id,
                countDistinct(user_id) AS historical_user_count,
                count() AS historical_event_count
            FROM {self._source_table}
            WHERE date_time >= toDateTime64(
                    parseDateTimeBestEffort({{observation_start:String}}), 3, 'UTC'
                  )
              AND date_time < toDateTime64(
                    parseDateTimeBestEffort({{cutoff:String}}), 3, 'UTC'
                  )
              AND srch_destination_id > 0
              AND modulo(cityHash64(toString(user_id)), {{sample_modulo:UInt64}})
                    = {{sample_remainder:UInt64}}
              AND NOT has(
                    {{excluded_destination_ids:Array(UInt64)}},
                    toUInt64(srch_destination_id)
                  )
            GROUP BY srch_destination_id
            HAVING historical_user_count >= {{min_users:UInt64}}
            ORDER BY historical_user_count DESC, historical_event_count DESC,
                     target_destination_id ASC
            LIMIT {{limit:UInt32}}
            """,
            parameters={
                "observation_start": _clickhouse_datetime(observation_start),
                "cutoff": _clickhouse_datetime(cutoff),
                "sample_modulo": user_sample_modulo,
                "sample_remainder": user_sample_remainder,
                "min_users": min_users,
                "limit": limit,
                "excluded_destination_ids": list(excluded_destination_ids),
            },
        )
        return [
            ExpediaBacktestScenario(
                scenario_id=_scenario_id(
                    cutoff=cutoff,
                    destination_id=int(_row_value(row, "target_destination_id", 0)),
                    season=season,
                ),
                cutoff=cutoff,
                target_destination_id=int(
                    _row_value(row, "target_destination_id", 0)
                ),
                historical_user_count=int(
                    _row_value(row, "historical_user_count", 1)
                ),
                historical_event_count=int(
                    _row_value(row, "historical_event_count", 2)
                ),
                season=season,
            )
            for row in _clickhouse_rows(result)
        ]

    def list_user_profiles(
        self,
        *,
        scenario: ExpediaBacktestScenario,
        observation_start: datetime,
        cutoff: datetime,
        limit: int,
        user_sample_modulo: int,
        user_sample_remainder: int,
    ) -> list[RawEventUserSignalRecord]:
        result = self._client.query(
            _profile_query(self._source_table),
            parameters={
                "project_id": self._project_id,
                "observation_start": _clickhouse_datetime(observation_start),
                "cutoff": _clickhouse_datetime(cutoff),
                "target_destination_id": scenario.target_destination_id,
                "season_months": list(scenario.season_months),
                "sample_modulo": user_sample_modulo,
                "sample_remainder": user_sample_remainder,
                "limit": limit,
            },
        )
        return [_profile_from_row(row) for row in _clickhouse_rows(result)]

    def future_booking_users(
        self,
        *,
        scenario: ExpediaBacktestScenario,
        outcome_end: datetime,
        eligible_user_ids: Sequence[str],
    ) -> ExpediaFutureBookingUsers:
        source_user_ids = [_source_user_id(user_id) for user_id in eligible_user_ids]
        if not source_user_ids:
            return ExpediaFutureBookingUsers(frozenset(), frozenset())
        result = self._client.query(
            f"""
            SELECT
                concat('{_EXPEDIA_USER_PREFIX}', toString(user_id))
                    AS backtest_user_id,
                max(
                    srch_destination_id = {{target_destination_id:UInt64}}
                    AND (
                        empty({{season_months:Array(UInt8)}})
                        OR (
                            NOT isNull(srch_ci)
                            AND toMonth(assumeNotNull(srch_ci))
                                IN {{season_months:Array(UInt8)}}
                        )
                    )
                ) AS contextual_booking
            FROM {self._source_table}
            WHERE date_time >= toDateTime64(
                    parseDateTimeBestEffort({{cutoff:String}}), 3, 'UTC'
                  )
              AND date_time < toDateTime64(
                    parseDateTimeBestEffort({{outcome_end:String}}), 3, 'UTC'
                  )
              AND is_booking = 1
              AND user_id IN {{user_ids:Array(UInt64)}}
            GROUP BY user_id
            """,
            parameters={
                "cutoff": _clickhouse_datetime(scenario.cutoff),
                "outcome_end": _clickhouse_datetime(outcome_end),
                "target_destination_id": scenario.target_destination_id,
                "season_months": list(scenario.season_months),
                "user_ids": source_user_ids,
            },
        )
        any_booking: set[str] = set()
        contextual_booking: set[str] = set()
        for row in _clickhouse_rows(result):
            user_id = str(_row_value(row, "backtest_user_id", 0))
            any_booking.add(user_id)
            if bool(_row_value(row, "contextual_booking", 1)):
                contextual_booking.add(user_id)
        return ExpediaFutureBookingUsers(
            any_booking_user_ids=frozenset(any_booking),
            contextual_booking_user_ids=frozenset(contextual_booking),
        )


class ExpediaSegmentBacktestService:
    def __init__(
        self,
        repository: ExpediaBacktestRepository,
        *,
        config: ExpediaBacktestConfig,
        performance_predictor: SegmentPerformancePredictor | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._performance_predictor = performance_predictor

    @log_context_scope
    def run(
        self,
        cutoffs: Sequence[datetime],
        *,
        fixed_scenarios: Mapping[datetime, Sequence[ExpediaBacktestScenario]]
        | None = None,
    ) -> ExpediaBacktestRun:
        started_at = now_ms()
        log.info(
            "started",
            {
                "cutoffCount": len(cutoffs),
                "lookbackDays": self._config.lookback_days,
                "outcomeDays": self._config.outcome_days,
                "profilePoolLimit": self._config.profile_pool_limit,
            },
        )
        results: list[ExpediaBacktestResult] = []
        skipped: list[ExpediaBacktestSkippedScenario] = []
        for cutoff in cutoffs:
            normalized_cutoff = _as_utc_datetime(cutoff)
            if normalized_cutoff is None:
                raise ValueError("cutoff must not be null")
            observation_start = normalized_cutoff - timedelta(
                days=self._config.lookback_days
            )
            outcome_end = normalized_cutoff + timedelta(
                days=self._config.outcome_days
            )
            if fixed_scenarios is None:
                scenarios = self._repository.list_scenarios(
                    observation_start=observation_start,
                    cutoff=normalized_cutoff,
                    limit=self._config.max_scenarios_per_cutoff,
                    min_users=self._config.min_scenario_users,
                    user_sample_modulo=self._config.user_sample_modulo,
                    user_sample_remainder=self._config.user_sample_remainder,
                    season=self._config.season,
                    excluded_destination_ids=(
                        self._config.excluded_destination_ids
                    ),
                )
            else:
                scenarios = list(fixed_scenarios.get(normalized_cutoff, ()))
            log.info(
                "backtest_cutoff_started",
                {
                    "cutoff": normalized_cutoff,
                    "observationStart": observation_start,
                    "outcomeEnd": outcome_end,
                    "scenarioCount": len(scenarios),
                },
            )
            cutoff_result_count = 0
            for scenario in scenarios:
                profiles = self._repository.list_user_profiles(
                    scenario=scenario,
                    observation_start=observation_start,
                    cutoff=normalized_cutoff,
                    limit=self._config.profile_pool_limit,
                    user_sample_modulo=self._config.user_sample_modulo,
                    user_sample_remainder=self._config.user_sample_remainder,
                )
                if len(profiles) < self._config.min_sample_size:
                    skipped.append(
                        ExpediaBacktestSkippedScenario(
                            cutoff=normalized_cutoff,
                            scenario_id=scenario.scenario_id,
                            reason="insufficient_profiles",
                        )
                    )
                    continue
                promotion = _promotion_for_scenario(scenario, self._config)
                intent = _intent_for_scenario(scenario)
                segments = generate_raw_event_segment_definitions(
                    promotion=promotion,
                    intent=intent,
                    compilation=compile_raw_event_intent(intent),
                    profiles=profiles,
                    max_suggested_segments=self._config.max_suggested_segments,
                    min_sample_size=self._config.min_sample_size,
                    performance_predictor=self._performance_predictor,
                )
                if not segments:
                    skipped.append(
                        ExpediaBacktestSkippedScenario(
                            cutoff=normalized_cutoff,
                            scenario_id=scenario.scenario_id,
                            reason="no_segment_candidates",
                        )
                    )
                    continue
                eligible_user_ids = tuple(profile.user_id for profile in profiles)
                future_users = self._repository.future_booking_users(
                    scenario=scenario,
                    outcome_end=outcome_end,
                    eligible_user_ids=eligible_user_ids,
                )
                scenario_results = _evaluate_segments(
                    scenario=scenario,
                    observation_start=observation_start,
                    outcome_end=outcome_end,
                    eligible_user_ids=eligible_user_ids,
                    future_users=future_users,
                    segments=segments,
                )
                results.extend(scenario_results)
                cutoff_result_count += len(scenario_results)
            log.info(
                "backtest_cutoff_completed",
                {
                    "cutoff": normalized_cutoff,
                    "scenarioCount": len(scenarios),
                    "candidateResultCount": cutoff_result_count,
                },
            )
        response = ExpediaBacktestRun(tuple(results), tuple(skipped))
        log.info(
            "completed",
            {
                "candidateResultCount": len(response.results),
                "skippedScenarioCount": len(response.skipped_scenarios),
                "durationMs": duration_ms(started_at),
            },
        )
        return response

    def run_scenarios(
        self,
        scenarios: Sequence[ExpediaBacktestScenario],
    ) -> ExpediaBacktestRun:
        scenarios_by_cutoff: dict[datetime, list[ExpediaBacktestScenario]] = (
            defaultdict(list)
        )
        for scenario in scenarios:
            cutoff = _as_utc_datetime(scenario.cutoff)
            if cutoff is None:
                raise ValueError("scenario cutoff must not be null")
            scenarios_by_cutoff[cutoff].append(scenario)
        if not scenarios_by_cutoff:
            raise ValueError("sealed scenario list must not be empty")
        return self.run(
            sorted(scenarios_by_cutoff),
            fixed_scenarios=scenarios_by_cutoff,
        )


def run_temporal_holdout_backtest(
    repository: ExpediaBacktestRepository,
    *,
    config: ExpediaBacktestConfig,
    training_cutoffs: Sequence[datetime],
    holdout_cutoffs: Sequence[datetime],
) -> ExpediaTemporalHoldoutRun:
    if not training_cutoffs or not holdout_cutoffs:
        raise ValueError("training and validation cutoffs must not be empty")
    normalized_training = tuple(
        value
        for cutoff in training_cutoffs
        if (value := _as_utc_datetime(cutoff)) is not None
    )
    normalized_holdout = tuple(
        value
        for cutoff in holdout_cutoffs
        if (value := _as_utc_datetime(cutoff)) is not None
    )
    training_outcome_end = max(normalized_training) + timedelta(
        days=config.outcome_days
    )
    if training_outcome_end > min(normalized_holdout):
        raise ValueError(
            "training outcomes must end before the first validation cutoff"
        )

    training_config = replace(
        config,
        max_suggested_segments=max(
            config.max_suggested_segments,
            len(MODEL_CANDIDATE_TYPES),
        ),
    )
    training_feature_run = ExpediaSegmentBacktestService(
        repository,
        config=training_config,
    ).run(normalized_training)
    examples = calibration_examples_from_run(training_feature_run)
    model = fit_logistic_segment_performance_model(
        examples,
        training_metadata={
            "target": "future_contextual_booking_rate",
            "training_start_cutoff": min(normalized_training).isoformat(),
            "training_end_cutoff": max(normalized_training).isoformat(),
            "outcome_days": config.outcome_days,
            "user_sample_modulo": config.user_sample_modulo,
            "profile_pool_limit": config.profile_pool_limit,
        },
    )
    training_run = ExpediaSegmentBacktestService(
        repository,
        config=training_config,
        performance_predictor=model,
    ).run(normalized_training)
    holdout_run = ExpediaSegmentBacktestService(
        repository,
        config=config,
        performance_predictor=model,
    ).run(normalized_holdout)
    return ExpediaTemporalHoldoutRun(
        training_run=training_run,
        holdout_run=holdout_run,
        calibration_model=model,
    )


def calibration_examples_from_run(
    run: ExpediaBacktestRun,
) -> list[CalibrationTrainingExample]:
    examples: list[CalibrationTrainingExample] = []
    for result in run.results:
        if result.sample_size <= 0:
            continue
        examples.append(
            CalibrationTrainingExample(
                features=SegmentPerformanceFeatures.from_json(
                    result.performance_features
                ),
                success_count=result.contextual_booking_user_count,
                sample_size=result.sample_size,
            )
        )
    if len(examples) < 2:
        raise ExpediaBacktestError(
            "training backtest produced too few calibration examples"
        )
    return examples


def validate_train_csv_header(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ExpediaBacktestError(f"Expedia train CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as source:
        header = next(csv.reader(source), None)
    if header is None:
        raise ExpediaBacktestError(f"Expedia train CSV is empty: {path}")
    missing = [column for column in EXPEDIA_TRAIN_COLUMNS if column not in header]
    if missing:
        raise ExpediaBacktestError(
            "Expedia backtest requires train.csv with future labels; "
            f"missing columns: {', '.join(missing)}"
        )


def validate_source_window(
    stats: ExpediaSourceStats,
    *,
    cutoffs: Sequence[datetime],
    lookback_days: int,
    outcome_days: int,
) -> None:
    if stats.row_count <= 0:
        raise ExpediaBacktestError(
            "Expedia source table is empty; run the prepare command with train.csv"
        )
    if stats.first_event_at is None or stats.last_event_at is None:
        raise ExpediaBacktestError("Expedia source table has no valid date_time values")
    required_start = min(cutoffs) - timedelta(days=lookback_days)
    required_end = max(cutoffs) + timedelta(days=outcome_days)
    if stats.first_event_at > required_start:
        raise ExpediaBacktestError(
            "source data starts after the required observation window: "
            f"required={required_start.isoformat()}, actual={stats.first_event_at.isoformat()}"
        )
    if stats.last_event_at < required_end - timedelta(microseconds=1):
        raise ExpediaBacktestError(
            "source data ends before the required outcome window: "
            f"required={required_end.isoformat()}, actual={stats.last_event_at.isoformat()}"
        )


def monthly_cutoffs(start: date, end: date) -> list[datetime]:
    if end < start:
        raise ValueError("end cutoff must be on or after start cutoff")
    current = start
    values: list[datetime] = []
    while current <= end:
        values.append(datetime(current.year, current.month, current.day, tzinfo=UTC))
        month = current.month + 1
        year = current.year
        if month == 13:
            month = 1
            year += 1
        try:
            current = current.replace(year=year, month=month)
        except ValueError as exc:
            raise ValueError("monthly cutoffs require a day valid in every month") from exc
    return values


def summarize_backtest(run: ExpediaBacktestRun) -> dict[str, Any]:
    results = list(run.results)
    grouped: dict[tuple[str, str], list[ExpediaBacktestResult]] = defaultdict(list)
    for result in results:
        grouped[(result.cutoff.isoformat(), result.scenario_id)].append(result)
    evaluable_grouped = {
        key: scenario_results
        for key, scenario_results in grouped.items()
        if any(
            result.baseline_contextual_conversion_rate > 0
            for result in scenario_results
        )
    }
    evaluable_results = [
        result
        for scenario_results in evaluable_grouped.values()
        for result in scenario_results
    ]
    calibration_results = results
    rank_one = [result for result in evaluable_results if result.rank == 1]
    calibration_rank_one = [
        result for result in calibration_results if result.rank == 1
    ]
    rank_one_is_best = 0
    for scenario_results in evaluable_grouped.values():
        rank_one_result = next(
            (result for result in scenario_results if result.rank == 1),
            None,
        )
        if rank_one_result is None:
            continue
        best_actual = max(
            result.actual_contextual_conversion_rate for result in scenario_results
        )
        if rank_one_result.actual_contextual_conversion_rate >= best_actual:
            rank_one_is_best += 1
    by_candidate_type: dict[str, dict[str, Any]] = {}
    for candidate_type in sorted(
        {result.candidate_type for result in evaluable_results}
    ):
        candidates = [
            result
            for result in evaluable_results
            if result.candidate_type == candidate_type
        ]
        by_candidate_type[candidate_type] = {
            "count": len(candidates),
            "mean_predicted_conversion_rate": _mean(
                result.predicted_conversion_rate for result in candidates
            ),
            "mean_actual_contextual_conversion_rate": _mean(
                result.actual_contextual_conversion_rate for result in candidates
            ),
            "mean_actual_any_conversion_rate": _mean(
                result.actual_any_conversion_rate for result in candidates
            ),
            "mean_absolute_lift_percentage_points": _mean(
                result.absolute_lift_percentage_points for result in candidates
            ),
            "mean_calibration_error_percentage_points": _mean(
                result.calibration_error_percentage_points for result in candidates
            ),
            "brier_score": _brier_score(candidates),
        }
    return {
        "scenario_count": len(grouped),
        "evaluable_scenario_count": len(evaluable_grouped),
        "unevaluable_scenario_count": len(grouped) - len(evaluable_grouped),
        "candidate_result_count": len(results),
        "skipped_scenario_count": len(run.skipped_scenarios),
        "rank_one_count": len(rank_one),
        "rank_one_mean_predicted_conversion_rate": _mean(
            result.predicted_conversion_rate for result in rank_one
        ),
        "rank_one_mean_actual_contextual_conversion_rate": _mean(
            result.actual_contextual_conversion_rate for result in rank_one
        ),
        "rank_one_mean_baseline_contextual_conversion_rate": _mean(
            result.baseline_contextual_conversion_rate for result in rank_one
        ),
        "rank_one_mean_actual_any_conversion_rate": _mean(
            result.actual_any_conversion_rate for result in rank_one
        ),
        "rank_one_mean_baseline_any_conversion_rate": _mean(
            result.baseline_any_conversion_rate for result in rank_one
        ),
        "rank_one_mean_any_booking_lift_percentage_points": _mean(
            (
                result.actual_any_conversion_rate
                - result.baseline_any_conversion_rate
            )
            * 100.0
            for result in rank_one
        ),
        "rank_one_mean_absolute_lift_percentage_points": _mean(
            result.absolute_lift_percentage_points for result in rank_one
        ),
        "rank_one_mean_calibration_error_percentage_points": _mean(
            result.calibration_error_percentage_points
            for result in calibration_rank_one
        ),
        "rank_one_brier_score": _brier_score(calibration_rank_one),
        "all_candidate_mean_absolute_error_percentage_points": _mean(
            result.calibration_error_percentage_points
            for result in calibration_results
        ),
        "all_candidate_brier_score": _brier_score(calibration_results),
        "all_candidate_prediction_bias_percentage_points": _mean(
            (
                result.predicted_conversion_rate
                - result.actual_contextual_conversion_rate
            )
            * 100.0
            for result in calibration_results
        ),
        "rank_one_beats_baseline_rate": _safe_rate(
            sum(
                result.actual_contextual_conversion_rate
                > result.baseline_contextual_conversion_rate
                for result in rank_one
            ),
            len(rank_one),
        ),
        "rank_one_is_best_rate": _safe_rate(
            rank_one_is_best, len(evaluable_grouped)
        ),
        "by_candidate_type": by_candidate_type,
    }


def write_backtest_artifacts(
    run: ExpediaBacktestRun,
    *,
    output_dir: Path,
    source_stats: ExpediaSourceStats,
    config: ExpediaBacktestConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    skipped_path = output_dir / "skipped_scenarios.csv"
    result_rows = [result.to_dict() for result in run.results]
    _write_csv(result_path, result_rows)
    _write_csv(
        skipped_path,
        [scenario.to_dict() for scenario in run.skipped_scenarios],
    )
    summary = {
        "source": {
            **asdict(source_stats),
            "first_event_at": source_stats.first_event_at.isoformat()
            if source_stats.first_event_at
            else None,
            "last_event_at": source_stats.last_event_at.isoformat()
            if source_stats.last_event_at
            else None,
            "booking_row_rate": source_stats.booking_row_rate,
        },
        "config": asdict(config),
        "metrics": summarize_backtest(run),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(run, summary), encoding="utf-8")
    return {
        "results": result_path,
        "summary": summary_path,
        "report": report_path,
        "skipped": skipped_path,
    }


def write_temporal_holdout_artifacts(
    run: ExpediaTemporalHoldoutRun,
    *,
    output_dir: Path,
    source_stats: ExpediaSourceStats,
    config: ExpediaBacktestConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_paths = write_backtest_artifacts(
        run.training_run,
        output_dir=output_dir / "training-2013",
        source_stats=source_stats,
        config=replace(
            config,
            max_suggested_segments=max(
                config.max_suggested_segments,
                len(MODEL_CANDIDATE_TYPES),
            ),
        ),
    )
    validation_paths = write_backtest_artifacts(
        run.holdout_run,
        output_dir=output_dir / "development-validation-2014",
        source_stats=source_stats,
        config=config,
    )
    model_path = output_dir / "contextual_booking_calibration_v1.json"
    write_segment_performance_model(run.calibration_model, model_path)
    summary_path = output_dir / "temporal_validation_summary.json"
    summary = {
        "split": {
            "training": "2013",
            "development_validation": "2014",
            "final_test": "not_run",
            "target": "future_contextual_booking_rate",
        },
        "config": asdict(config),
        "model": run.calibration_model.to_json(),
        "training_metrics": summarize_backtest(run.training_run),
        "validation_metrics": summarize_backtest(run.holdout_run),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "temporal_validation_report.md"
    report_path.write_text(
        _temporal_holdout_markdown_report(summary),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "summary": summary_path,
        "report": report_path,
        "training_results": training_paths["results"],
        "validation_results": validation_paths["results"],
    }


def validate_table_identifier(value: str) -> str:
    if not _TABLE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid ClickHouse table identifier: {value!r}")
    return value


def _evaluate_segments(
    *,
    scenario: ExpediaBacktestScenario,
    observation_start: datetime,
    outcome_end: datetime,
    eligible_user_ids: Sequence[str],
    future_users: ExpediaFutureBookingUsers,
    segments: Sequence[Any],
) -> list[ExpediaBacktestResult]:
    eligible = set(eligible_user_ids)
    contextual_baseline_count = len(
        eligible & set(future_users.contextual_booking_user_ids)
    )
    any_baseline_count = len(eligible & set(future_users.any_booking_user_ids))
    baseline_contextual_rate = _safe_rate(contextual_baseline_count, len(eligible))
    baseline_any_rate = _safe_rate(any_baseline_count, len(eligible))
    results: list[ExpediaBacktestResult] = []
    for rank, segment in enumerate(segments, start=1):
        candidate_user_ids = set(segment.rule_json.get("candidate_user_ids", []))
        contextual_count = len(
            candidate_user_ids & set(future_users.contextual_booking_user_ids)
        )
        any_count = len(candidate_user_ids & set(future_users.any_booking_user_ids))
        sample_size = len(candidate_user_ids)
        actual_contextual_rate = _safe_rate(contextual_count, sample_size)
        actual_any_rate = _safe_rate(any_count, sample_size)
        performance_estimate = segment.profile_json.get("performance_estimate", {})
        predicted_rate = float(performance_estimate.get("value", 0.0) or 0.0)
        performance_features = segment.profile_json.get("performance_features", {})
        signal_metrics = segment.profile_json.get("signal_metrics", {})
        absolute_lift = (actual_contextual_rate - baseline_contextual_rate) * 100.0
        relative_lift = (
            actual_contextual_rate / baseline_contextual_rate - 1.0
            if baseline_contextual_rate > 0
            else None
        )
        results.append(
            ExpediaBacktestResult(
                cutoff=scenario.cutoff,
                observation_start=observation_start,
                outcome_end=outcome_end,
                scenario_id=scenario.scenario_id,
                target_destination_id=scenario.target_destination_id,
                season=scenario.season,
                rank=rank,
                segment_id=segment.segment_id,
                candidate_type=str(
                    segment.profile_json.get("candidate_type", "unknown")
                ),
                rank_role=str(segment.profile_json.get("rank_role", "")),
                sample_size=sample_size,
                total_eligible_user_count=len(eligible),
                matching_profile_count=int(
                    signal_metrics.get("matching_profile_count", sample_size) or 0
                ),
                performance_features=dict(performance_features),
                prediction_method=str(
                    performance_estimate.get("method", "unknown")
                ),
                prediction_model_version=str(
                    performance_estimate.get("model_version", "unknown")
                ),
                predicted_conversion_rate=predicted_rate,
                actual_contextual_conversion_rate=actual_contextual_rate,
                actual_any_conversion_rate=actual_any_rate,
                baseline_contextual_conversion_rate=baseline_contextual_rate,
                baseline_any_conversion_rate=baseline_any_rate,
                contextual_booking_user_count=contextual_count,
                any_booking_user_count=any_count,
                absolute_lift_percentage_points=absolute_lift,
                relative_lift=relative_lift,
                calibration_error_percentage_points=abs(
                    predicted_rate - actual_contextual_rate
                )
                * 100.0,
                recommendation_score=float(
                    segment.profile_json.get("recommendation_score", 0.0) or 0.0
                ),
            )
        )
    return results


def _promotion_for_scenario(
    scenario: ExpediaBacktestScenario,
    config: ExpediaBacktestConfig,
) -> PromotionRecord:
    season_text = f" {scenario.season}" if scenario.season else ""
    return PromotionRecord(
        project_id="expedia_backtest",
        campaign_id=f"backtest_{scenario.cutoff:%Y%m%d}",
        promotion_id=f"promo_{scenario.scenario_id}",
        channel="email",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.10"),
        goal_basis="all_segments",
        min_sample_size=config.min_sample_size,
        landing_url=(
            "https://backtest.local/hotels/search?destination_id="
            f"{scenario.target_destination_id}"
        ),
        message_brief=(
            f"Expedia destination {scenario.target_destination_id}{season_text} "
            "hotel booking promotion"
        ),
    )


def _intent_for_scenario(scenario: ExpediaBacktestScenario) -> PromotionIntent:
    season = (scenario.season,) if scenario.season else ()
    destination = str(scenario.target_destination_id)
    return PromotionIntent(
        summary=f"목적지 {destination} 숙소 예약 프로모션",
        product="hotel",
        season=season,
        destinations=(destination,),
        benefits=(),
        audience_hints=(),
        channel="email",
        goal_metric="booking_conversion_rate",
        funnel_goal="booking_start_or_complete",
        desired_behaviors=(
            "hotel_detail_view",
            "booking_start_without_complete",
            "recent_destination_search",
        ),
        explicit_conditions=(*season, destination, "hotel", "email"),
        source="expedia_backtest_ground_truth",
    )


def _profile_query(source_table: str) -> str:
    return f"""
    SELECT
        {{project_id:String}} AS project_id,
        concat('{_EXPEDIA_USER_PREFIX}', toString(user_id)) AS user_id,
        sum(
            3
            + if(cnt >= 2 OR is_booking = 1, 1, 0)
            + if(is_booking = 1 OR (is_booking = 0 AND cnt >= 4), 1, 0)
            + if(is_booking = 1, 1, 0)
        ) AS event_count,
        count() AS hotel_search_count,
        countIf(cnt >= 2 OR is_booking = 1) AS hotel_click_count,
        count() AS hotel_detail_view_count,
        toUInt64(0) AS promotion_impression_count,
        toUInt64(0) AS promotion_click_count,
        toUInt64(0) AS campaign_redirect_click_count,
        toUInt64(0) AS campaign_landing_count,
        countIf(is_booking = 1 OR (is_booking = 0 AND cnt >= 4))
            AS booking_start_count,
        countIf(is_booking = 1) AS booking_complete_count,
        toUInt64(0) AS booking_cancel_count,
        countIf(
            is_package = 1
        ) AS deal_event_count,
        toUInt64(0) AS free_cancellation_count,
        toUInt64(0) AS breakfast_included_count,
        toUInt64(0) AS price_event_count,
        toFloat64(0) AS avg_price,
        groupUniqArray(20)(toString(srch_destination_id)) AS destination_values,
        groupUniqArray(20)(if(isNull(srch_ci), '', toString(assumeNotNull(srch_ci))))
            AS checkin_dates,
        groupUniqArray(20)(toString(hotel_market)) AS hotel_market_values,
        groupUniqArray(20)(toString(hotel_cluster)) AS hotel_cluster_values,
        CAST([], 'Array(String)') AS age_group_values,
        CAST([], 'Array(String)') AS gender_values,
        groupUniqArray(10)(
            multiIf(
                srch_children_cnt > 0, 'family_travel',
                is_package = 1, 'package_travel',
                srch_adults_cnt = 1 AND srch_children_cnt = 0,
                    'business_or_solo_travel',
                'leisure_travel'
            )
        ) AS preferred_category_values,
        countIf(srch_destination_id = {{target_destination_id:UInt64}})
            AS destination_match_count,
        countIf(
            NOT isNull(srch_ci)
            AND NOT empty({{season_months:Array(UInt8)}})
            AND toMonth(assumeNotNull(srch_ci)) IN {{season_months:Array(UInt8)}}
        ) AS season_match_count
    FROM {source_table}
    WHERE date_time >= toDateTime64(
            parseDateTimeBestEffort({{observation_start:String}}), 3, 'UTC'
          )
      AND date_time < toDateTime64(
            parseDateTimeBestEffort({{cutoff:String}}), 3, 'UTC'
          )
      AND modulo(cityHash64(toString(user_id)), {{sample_modulo:UInt64}})
            = {{sample_remainder:UInt64}}
    GROUP BY user_id
    ORDER BY max(date_time) DESC, user_id ASC
    LIMIT {{limit:UInt32}}
    """


def _profile_from_row(row: Any) -> RawEventUserSignalRecord:
    return RawEventUserSignalRecord(
        project_id=str(_row_value(row, "project_id", 0)),
        user_id=str(_row_value(row, "user_id", 1)),
        event_count=int(_row_value(row, "event_count", 2) or 0),
        hotel_search_count=int(_row_value(row, "hotel_search_count", 3) or 0),
        hotel_click_count=int(_row_value(row, "hotel_click_count", 4) or 0),
        hotel_detail_view_count=int(
            _row_value(row, "hotel_detail_view_count", 5) or 0
        ),
        promotion_impression_count=int(
            _row_value(row, "promotion_impression_count", 6) or 0
        ),
        promotion_click_count=int(
            _row_value(row, "promotion_click_count", 7) or 0
        ),
        campaign_redirect_click_count=int(
            _row_value(row, "campaign_redirect_click_count", 8) or 0
        ),
        campaign_landing_count=int(
            _row_value(row, "campaign_landing_count", 9) or 0
        ),
        booking_start_count=int(_row_value(row, "booking_start_count", 10) or 0),
        booking_complete_count=int(
            _row_value(row, "booking_complete_count", 11) or 0
        ),
        booking_cancel_count=int(
            _row_value(row, "booking_cancel_count", 12) or 0
        ),
        deal_event_count=int(_row_value(row, "deal_event_count", 13) or 0),
        free_cancellation_count=int(
            _row_value(row, "free_cancellation_count", 14) or 0
        ),
        breakfast_included_count=int(
            _row_value(row, "breakfast_included_count", 15) or 0
        ),
        price_event_count=int(_row_value(row, "price_event_count", 16) or 0),
        avg_price=float(_row_value(row, "avg_price", 17) or 0.0),
        destination_values=_string_tuple(_row_value(row, "destination_values", 18)),
        checkin_dates=_string_tuple(_row_value(row, "checkin_dates", 19)),
        hotel_market_values=_string_tuple(
            _row_value(row, "hotel_market_values", 20)
        ),
        hotel_cluster_values=_string_tuple(
            _row_value(row, "hotel_cluster_values", 21)
        ),
        age_group_values=_string_tuple(_row_value(row, "age_group_values", 22)),
        gender_values=_string_tuple(_row_value(row, "gender_values", 23)),
        preferred_category_values=_string_tuple(
            _row_value(row, "preferred_category_values", 24)
        ),
        destination_match_count=int(
            _row_value(row, "destination_match_count", 25) or 0
        ),
        season_match_count=int(_row_value(row, "season_match_count", 26) or 0),
    )


def _create_source_table_sql(table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        date_time DateTime64(3, 'UTC'),
        site_name UInt16,
        posa_continent UInt16,
        user_location_country UInt32,
        user_location_region UInt32,
        user_location_city UInt32,
        orig_destination_distance Nullable(Float64),
        user_id UInt64,
        is_mobile UInt8,
        is_package UInt8,
        channel UInt16,
        srch_ci Nullable(Date),
        srch_co Nullable(Date),
        srch_adults_cnt UInt16,
        srch_children_cnt UInt16,
        srch_rm_cnt UInt16,
        srch_destination_id UInt64,
        srch_destination_type_id UInt16,
        is_booking UInt8,
        cnt UInt32,
        hotel_continent UInt16,
        hotel_country UInt32,
        hotel_market UInt32,
        hotel_cluster UInt16
    )
    ENGINE = MergeTree
    ORDER BY (user_id, date_time, srch_destination_id, hotel_market, hotel_cluster)
    """


def _markdown_report(
    run: ExpediaBacktestRun,
    summary: Mapping[str, Any],
) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Expedia AI 세그먼트 추천 백테스트",
        "",
        "## 핵심 결과",
        "",
        f"- 평가 시나리오: {metrics['scenario_count']}개",
        "- 미래 목적지 예약이 있어 평가 가능한 시나리오: "
        f"{metrics['evaluable_scenario_count']}개",
        f"- 추천 후보 결과: {metrics['candidate_result_count']}개",
        "- Rank 1 실제 전환율: "
        f"{_format_percent(metrics['rank_one_mean_actual_contextual_conversion_rate'])}",
        "- 전체 기준 실제 전환율: "
        f"{_format_percent(metrics['rank_one_mean_baseline_contextual_conversion_rate'])}",
        "- Rank 1 평균 절대 향상: "
        f"{metrics['rank_one_mean_absolute_lift_percentage_points']:.2f}%p",
        "- Rank 1 목적지 무관 예약률: "
        f"{_format_percent(metrics['rank_one_mean_actual_any_conversion_rate'])}",
        "- 전체 기준 목적지 무관 예약률: "
        f"{_format_percent(metrics['rank_one_mean_baseline_any_conversion_rate'])}",
        "- Rank 1이 기준선을 이긴 비율: "
        f"{_format_percent(metrics['rank_one_beats_baseline_rate'])}",
        "- Rank 1이 실제 최고 후보였던 비율: "
        f"{_format_percent(metrics['rank_one_is_best_rate'])}",
        "- 예상값 평균 절대 오차: "
        f"{metrics['rank_one_mean_calibration_error_percentage_points']:.2f}%p",
        "",
        "## 후보별 결과",
        "",
        "| 기준일 | 목적지 | Rank | 후보 유형 | 예상 전환율 | "
        "미래 실제 전환율 | 기준 전환율 | 향상(%p) | 표본 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in run.results:
        lines.append(
            "| "
            f"{result.cutoff.date().isoformat()} | {result.target_destination_id} | "
            f"{result.rank} | {result.candidate_type} | "
            f"{_format_percent(result.predicted_conversion_rate)} | "
            f"{_format_percent(result.actual_contextual_conversion_rate)} | "
            f"{_format_percent(result.baseline_contextual_conversion_rate)} | "
            f"{result.absolute_lift_percentage_points:.2f} | {result.sample_size} |"
        )
    lines.extend(
        [
            "",
            "## 해석 주의사항",
            "",
            "- 추천 입력에는 기준일 이전 행동만 사용했습니다.",
            "- 실제 전환은 기준일 이후 평가 구간의 사용자별 `is_booking=1`로 "
            "계산했습니다.",
            "- 목적지 무관 예약률만 높고 해당 목적지 전환 향상이 낮으면 프로모션 "
            "맞춤 추천보다 일반 고전환 고객 추천에 가깝습니다.",
            "- Expedia에는 광고 노출 대조군이 없으므로 이 결과는 미래 예약 가능성을 "
            "검증하며 광고의 인과적 증분 효과를 의미하지 않습니다.",
            "- 프로모션 클릭/랜딩 신호가 원본에 없어 `promotion_responsive` 후보는 "
            "평가되지 않을 수 있습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _temporal_holdout_markdown_report(summary: Mapping[str, Any]) -> str:
    training = summary["training_metrics"]
    validation = summary["validation_metrics"]
    model = summary["model"]
    return "\n".join(
        [
            "# Expedia 2013 학습 / 2014 개발 검증",
            "",
            "## 시간 분리",
            "",
            "- 2013년 기준일 이전 행동과 이후 30일 목적지 예약으로 보정식을 학습했습니다.",
            "- 2014년 결과는 모델 학습에는 사용하지 않았지만 반복적인 로직 개선에 사용한 개발 검증 데이터입니다.",
            f"- 학습 예시: {model['training_metadata']['training_example_count']}개",
            "",
            "## 2014년 개발 검증 결과",
            "",
            f"- 평가 시나리오: {validation['scenario_count']}개",
            "- Rank 1이 실제 최고 후보였던 비율: "
            f"{_format_percent(validation['rank_one_is_best_rate'])}",
            "- Rank 1이 기준선을 이긴 비율: "
            f"{_format_percent(validation['rank_one_beats_baseline_rate'])}",
            "- Rank 1 평균 절대 오차: "
            f"{validation['rank_one_mean_calibration_error_percentage_points']:.2f}%p",
            f"- Rank 1 Brier score: {validation['rank_one_brier_score']:.6f}",
            "- 전체 후보 평균 절대 오차: "
            f"{validation['all_candidate_mean_absolute_error_percentage_points']:.2f}%p",
            f"- 전체 후보 Brier score: {validation['all_candidate_brier_score']:.6f}",
            "- 전체 후보 예측 편향: "
            f"{validation['all_candidate_prediction_bias_percentage_points']:.2f}%p",
            "",
            "## 학습 구간 참고 지표",
            "",
            f"- 학습 시나리오: {training['scenario_count']}개",
            "- 학습 Rank 1 실제 최고 후보 비율: "
            f"{_format_percent(training['rank_one_is_best_rate'])}",
            "",
            "## 해석 주의사항",
            "",
            "- Expedia에는 실제 광고 노출 대조군이 없으므로 미래 목적지 예약 가능성을 검증한 결과입니다.",
            "- 이 결과를 본 뒤 추천 규칙을 수정했으므로 최종 일반화 성능으로 주장할 수 없습니다.",
            "- 최종 성능은 별도로 봉인한 미평가 목적지 테스트를 코드 동결 후 한 번만 실행해 확인해야 합니다.",
            "- `user_sample_modulo=1`은 해시 표본 없이 전체 원천 데이터에서 운영과 같은 후보 풀을 조회합니다.",
            "",
        ]
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scenario_id(*, cutoff: datetime, destination_id: int, season: str | None) -> str:
    suffix = f"_{season}" if season else ""
    return f"{cutoff:%Y%m%d}_destination_{destination_id}{suffix}"


def _source_user_id(user_id: str) -> int:
    if not user_id.startswith(_EXPEDIA_USER_PREFIX):
        raise ExpediaBacktestError(f"unexpected Expedia backtest user id: {user_id}")
    try:
        return int(user_id.removeprefix(_EXPEDIA_USER_PREFIX))
    except ValueError as exc:
        raise ExpediaBacktestError(
            f"unexpected Expedia backtest user id: {user_id}"
        ) from exc


def _clickhouse_rows(result: Any) -> list[Any]:
    named_results = getattr(result, "named_results", None)
    if callable(named_results):
        return list(named_results())
    return list(getattr(result, "result_rows", []))


def _row_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return row[index]


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(item) for item in value if str(item))


def _as_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _clickhouse_datetime(value: datetime) -> str:
    normalized = _as_utc_datetime(value)
    if normalized is None:
        raise ValueError("datetime must not be null")
    return normalized.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _brier_score(results: Sequence[ExpediaBacktestResult]) -> float:
    total_users = sum(result.sample_size for result in results)
    if total_users <= 0:
        return 0.0
    total_error = 0.0
    for result in results:
        prediction = max(0.0, min(1.0, result.predicted_conversion_rate))
        successes = result.contextual_booking_user_count
        failures = max(result.sample_size - successes, 0)
        total_error += successes * (1.0 - prediction) ** 2
        total_error += failures * prediction**2
    return total_error / total_users


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"
