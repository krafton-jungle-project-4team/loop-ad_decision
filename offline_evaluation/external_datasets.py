from __future__ import annotations

import csv
import heapq
import io
import itertools
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

from app.analysis.raw_event_segments import PromotionIntent
from app.analysis.repositories import PromotionRecord, RawEventUserSignalRecord
from offline_evaluation.external_backtest import (
    ExternalBacktestError,
    ExternalDatasetManifest,
    ExternalEvaluationCase,
    source_file_descriptor,
    stable_bucket,
    stable_score,
)


SYNERISE_SELECTION_RECENCY_DAYS = 14
EXTERNAL_DEVELOPMENT_ROLE = "development_diagnostic"
EXTERNAL_SEALED_FINAL_ROLE = "sealed_final"
EXTERNAL_EVALUATION_ROLES = frozenset(
    {EXTERNAL_DEVELOPMENT_ROLE, EXTERNAL_SEALED_FINAL_ROLE}
)


@dataclass(frozen=True, slots=True)
class ExternalAdapterConfig:
    profile_pool_limit: int = 1000
    max_scenarios: int = 3
    min_scenario_users: int = 20
    sample_modulo: int = 1
    sample_remainder: int = 0
    sample_remainders: tuple[int, ...] = ()
    evaluation_role: str = EXTERNAL_DEVELOPMENT_ROLE
    include_checksum: bool = True
    cutoff: datetime = datetime(2022, 11, 10, tzinfo=UTC)
    lookback_days: int = 90
    outcome_days: int = 28

    def __post_init__(self) -> None:
        if self.profile_pool_limit <= 0 or self.max_scenarios <= 0:
            raise ValueError("profile and scenario limits must be positive")
        if self.min_scenario_users <= 0:
            raise ValueError("min_scenario_users must be positive")
        if self.sample_modulo <= 0:
            raise ValueError("sample_modulo must be positive")
        if not 0 <= self.sample_remainder < self.sample_modulo:
            raise ValueError("sample_remainder must be smaller than sample_modulo")
        if self.sample_remainders:
            if len(set(self.sample_remainders)) != len(self.sample_remainders):
                raise ValueError("sample_remainders must not contain duplicates")
            if any(
                not 0 <= remainder < self.sample_modulo
                for remainder in self.sample_remainders
            ):
                raise ValueError(
                    "every sample remainder must be smaller than sample_modulo"
                )
        if self.evaluation_role not in EXTERNAL_EVALUATION_ROLES:
            raise ValueError("unsupported external evaluation role")
        if self.lookback_days <= 0 or self.outcome_days <= 0:
            raise ValueError("lookback_days and outcome_days must be positive")
        if self.cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")

    @property
    def effective_sample_remainders(self) -> tuple[int, ...]:
        return self.sample_remainders or (self.sample_remainder,)

    def includes_subject(self, subject_id: str) -> bool:
        return (
            stable_bucket(subject_id, self.sample_modulo)
            in self.effective_sample_remainders
        )


@dataclass(frozen=True, slots=True)
class ExternalDatasetBundle:
    manifest: ExternalDatasetManifest
    cases: tuple[ExternalEvaluationCase, ...]


def load_external_dataset(
    dataset_id: str,
    source_dir: Path,
    *,
    config: ExternalAdapterConfig,
) -> ExternalDatasetBundle:
    loaders = {
        "airbnb": load_airbnb_dataset,
        "booking-com": (
            load_booking_com_final_dataset
            if config.evaluation_role == EXTERNAL_SEALED_FINAL_ROLE
            else load_booking_com_dataset
        ),
        "synerise": load_synerise_dataset,
    }
    try:
        loader = loaders[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unsupported external dataset: {dataset_id}") from exc
    return loader(source_dir, config=config)


def external_source_paths(
    dataset_id: str,
    source_dir: Path,
    *,
    evaluation_role: str,
) -> tuple[Path, ...]:
    if evaluation_role not in EXTERNAL_EVALUATION_ROLES:
        raise ValueError("unsupported external evaluation role")
    if dataset_id == "booking-com":
        if evaluation_role == EXTERNAL_SEALED_FINAL_ROLE:
            return (
                source_dir / "test_set.csv",
                source_dir / "ground_truth.csv",
            )
        return (source_dir / "train_set.csv",)
    if dataset_id == "airbnb":
        return (
            source_dir / "train_users_2.csv.zip",
            source_dir / "sessions.csv.zip",
        )
    if dataset_id == "synerise":
        return tuple(
            source_dir / name
            for name in (
                "search_query.parquet",
                "add_to_cart.parquet",
                "remove_from_cart.parquet",
                "product_buy.parquet",
                "product_properties.parquet",
            )
        )
    raise ValueError(f"unsupported external dataset: {dataset_id}")


@dataclass(frozen=True, slots=True)
class _BookingTrip:
    trip_id: str
    observation: tuple[Mapping[str, str], ...]
    outcome: Mapping[str, str]


def load_booking_com_dataset(
    source_dir: Path,
    *,
    config: ExternalAdapterConfig,
) -> ExternalDatasetBundle:
    train_path = source_dir / "train_set.csv"
    sampler: _BoundedStableSample[_BookingTrip] = _BoundedStableSample(
        config.profile_pool_limit
    )
    for trip in _iter_booking_trips(train_path):
        if not config.includes_subject(trip.trip_id):
            continue
        sampler.add(trip.trip_id, trip)
    trips = sampler.values()
    if not trips:
        raise ExternalBacktestError("Booking.com sampling produced no trips")

    observed_city_users: Counter[str] = Counter()
    for trip in trips:
        observed_city_users.update(
            set(_nonempty(row.get("city_id")) for row in trip.observation) - {""}
        )
    target_cities = _top_targets(
        observed_city_users,
        limit=config.max_scenarios,
        min_users=config.min_scenario_users,
    )
    cases = tuple(
        _booking_case(target_city=target_city, trips=trips, config=config)
        for target_city in target_cities
    )
    manifest = ExternalDatasetManifest(
        dataset_id="booking-com",
        source_version="booking.multi-destination-trips.v1",
        evaluation_design="within_trip_sequential_holdout",
        outcome_name="next_itinerary_city_match_rate",
        supports_temporal_holdout=True,
        supported_claims=(
            "이전 숙박 도시 이력으로 다음 여행 도시와 맞는 세그먼트를 찾는 능력",
            "목적지 반복 관심 후보의 외부 숙박 데이터 일반화",
            "후보 Rank별 다음 도시 적중률과 baseline 대비 lift",
        ),
        unsupported_claims=(
            "검색 또는 호텔 상세 조회 이후의 예약 전환율",
            "미예약 사용자에 대한 예약 가능성",
            "프로모션 노출·클릭에 따른 증분 효과",
        ),
        signal_mappings={
            "destination_values": {
                "source": "city_id in prior itinerary reservations",
                "support": "direct",
            },
            "hotel_search_count": {
                "source": "count of prior itinerary reservations",
                "support": "proxy",
            },
            "booking_complete_count": {
                "source": "count of prior itinerary reservations",
                "support": "direct",
            },
            "outcome": {
                "source": "last city_id in each train_set trip",
                "support": "direct",
            },
        },
        source_files=(
            source_file_descriptor(
                train_path,
                include_checksum=config.include_checksum,
            ),
        ),
        evaluation_role=config.evaluation_role,
        prediction_error_comparability_reason=(
            "Booking.com next itinerary city is not the Expedia model's "
            "future same-destination booking target"
        ),
        notes=(
            "기본 validation은 train_set 여행의 마지막 도시만 숨겨 평가합니다.",
            "test_set과 ground_truth.csv는 기본 validation에서 읽지 않아 별도 최종 검증용으로 보존됩니다.",
            "예약 생성 시각이 없어 시간 기반 마케팅 전환이 아니라 여행 순서 기반 holdout입니다.",
        ),
    )
    return ExternalDatasetBundle(manifest=manifest, cases=cases)


def load_booking_com_final_dataset(
    source_dir: Path,
    *,
    config: ExternalAdapterConfig,
) -> ExternalDatasetBundle:
    if config.evaluation_role != EXTERNAL_SEALED_FINAL_ROLE:
        raise ValueError("Booking.com official test requires the sealed final role")
    test_path = source_dir / "test_set.csv"
    ground_truth_path = source_dir / "ground_truth.csv"
    ground_truth = _read_booking_ground_truth(ground_truth_path)
    sampler: _BoundedStableSample[_BookingTrip] = _BoundedStableSample(
        config.profile_pool_limit
    )
    for trip in _iter_booking_final_trips(test_path, ground_truth=ground_truth):
        if not config.includes_subject(trip.trip_id):
            continue
        sampler.add(trip.trip_id, trip)
    trips = sampler.values()
    if not trips:
        raise ExternalBacktestError(
            "Booking.com official test sampling produced no trips"
        )

    observed_city_users: Counter[str] = Counter()
    for trip in trips:
        observed_city_users.update(
            set(_nonempty(row.get("city_id")) for row in trip.observation) - {""}
        )
    target_cities = _top_targets(
        observed_city_users,
        limit=config.max_scenarios,
        min_users=config.min_scenario_users,
    )
    cases = tuple(
        _booking_case(
            target_city=target_city,
            trips=trips,
            config=config,
            evaluation_design="official_test_ground_truth_holdout",
            user_id_prefix="booking-final-trip",
            scenario_prefix="booking-com-official-next-city",
        )
        for target_city in target_cities
    )
    manifest = ExternalDatasetManifest(
        dataset_id="booking-com",
        source_version="booking.multi-destination-trips.official-test.v1",
        evaluation_design="official_test_ground_truth_holdout",
        outcome_name="next_itinerary_city_match_rate",
        supports_temporal_holdout=True,
        supported_claims=(
            "공식 test 여행 이력으로 다음 여행 도시와 맞는 세그먼트를 찾는 능력",
            "개발용 train_set과 분리된 사용자에서의 baseline 대비 lift",
            "후보 Rank별 다음 도시 적중률",
        ),
        unsupported_claims=(
            "검색 또는 호텔 상세 조회 이후의 예약 전환율",
            "프로모션 노출에 따른 증분 효과",
            "Expedia 예상 예약률의 보정 정확도",
        ),
        signal_mappings={
            "destination_values": {
                "source": "non-placeholder city_id in official test itinerary",
                "support": "direct",
            },
            "hotel_search_count": {
                "source": "count of observed official test reservations",
                "support": "proxy",
            },
            "booking_complete_count": {
                "source": "count of observed official test reservations",
                "support": "direct",
            },
            "outcome": {
                "source": "ground_truth.csv city_id",
                "support": "direct",
            },
        },
        source_files=tuple(
            source_file_descriptor(path, include_checksum=config.include_checksum)
            for path in (test_path, ground_truth_path)
        ),
        evaluation_role=config.evaluation_role,
        prediction_error_comparability_reason=(
            "Booking.com next itinerary city is not the Expedia model's "
            "future same-destination booking target"
        ),
        notes=(
            "test_set.csv의 city_id=0 placeholder는 관찰 feature에서 제외합니다.",
            "ground_truth.csv는 봉인 평가 실행을 시작한 뒤에만 outcome으로 읽습니다.",
            "예약 생성 시각이 없어 여행 순서 기반 holdout으로 해석합니다.",
        ),
    )
    return ExternalDatasetBundle(manifest=manifest, cases=cases)


def _iter_booking_trips(path: Path) -> Iterator[_BookingTrip]:
    if not path.is_file():
        raise ExternalBacktestError(f"Booking.com train_set.csv not found: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"utrip_id", "city_id", "checkin", "checkout"}
        if not required.issubset(reader.fieldnames or ()):
            raise ExternalBacktestError("Booking.com train_set.csv schema is invalid")
        for trip_id, rows in itertools.groupby(reader, key=lambda row: row["utrip_id"]):
            ordered = sorted(rows, key=lambda row: (row["checkin"], row["checkout"]))
            if len(ordered) < 2:
                continue
            yield _BookingTrip(
                trip_id=trip_id,
                observation=tuple(ordered[:-1]),
                outcome=ordered[-1],
            )


def _read_booking_ground_truth(path: Path) -> dict[str, Mapping[str, str]]:
    if not path.is_file():
        raise ExternalBacktestError(
            f"Booking.com ground_truth.csv not found: {path}"
        )
    outcomes: dict[str, Mapping[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"utrip_id", "city_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ExternalBacktestError(
                "Booking.com ground_truth.csv schema is invalid"
            )
        for row in reader:
            trip_id = _nonempty(row.get("utrip_id"))
            if trip_id:
                outcomes[trip_id] = row
    return outcomes


def _iter_booking_final_trips(
    path: Path,
    *,
    ground_truth: Mapping[str, Mapping[str, str]],
) -> Iterator[_BookingTrip]:
    if not path.is_file():
        raise ExternalBacktestError(f"Booking.com test_set.csv not found: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"utrip_id", "city_id", "checkin", "checkout"}
        if not required.issubset(reader.fieldnames or ()):
            raise ExternalBacktestError("Booking.com test_set.csv schema is invalid")
        for trip_id, rows in itertools.groupby(reader, key=lambda row: row["utrip_id"]):
            outcome = ground_truth.get(trip_id)
            if outcome is None:
                continue
            observation = tuple(
                sorted(
                    (
                        row
                        for row in rows
                        if _nonempty(row.get("city_id")) not in {"", "0"}
                    ),
                    key=lambda row: (row["checkin"], row["checkout"]),
                )
            )
            if not observation:
                continue
            yield _BookingTrip(
                trip_id=trip_id,
                observation=observation,
                outcome=outcome,
            )


def _booking_case(
    *,
    target_city: str,
    trips: Sequence[_BookingTrip],
    config: ExternalAdapterConfig,
    evaluation_design: str = "within_trip_sequential_holdout",
    user_id_prefix: str = "booking-trip",
    scenario_prefix: str = "booking-com-next-city",
) -> ExternalEvaluationCase:
    profiles: list[RawEventUserSignalRecord] = []
    positive_ids: set[str] = set()
    for trip in trips:
        user_id = f"{user_id_prefix}-{trip.trip_id}"
        observed_cities = tuple(
            _dedupe(_nonempty(row.get("city_id")) for row in trip.observation)
        )
        observed_countries = tuple(
            _dedupe(_nonempty(row.get("hotel_country")) for row in trip.observation)
        )
        profiles.append(
            _profile(
                project_id="external-booking-com",
                user_id=user_id,
                event_count=len(trip.observation),
                hotel_search_count=len(trip.observation),
                booking_complete_count=len(trip.observation),
                destination_values=observed_cities,
                hotel_market_values=observed_countries,
                preferred_category_values=tuple(
                    _dedupe(
                        value
                        for row in trip.observation
                        for value in (
                            _nonempty(row.get("device_class")),
                            f"affiliate:{_nonempty(row.get('affiliate_id'))}",
                        )
                        if value
                    )
                ),
                destination_match_count=sum(
                    city == target_city
                    for row in trip.observation
                    if (city := _nonempty(row.get("city_id")))
                ),
            )
        )
        if _nonempty(trip.outcome.get("city_id")) == target_city:
            positive_ids.add(user_id)
    scenario_id = f"{scenario_prefix}-{target_city}"
    promotion, intent = _destination_promotion(
        dataset_id="booking-com",
        scenario_id=scenario_id,
        target_value=target_city,
        min_sample_size=config.min_scenario_users,
        product="hotel",
    )
    return ExternalEvaluationCase(
        dataset_id="booking-com",
        scenario_id=scenario_id,
        target_value=target_city,
        target_label=f"도시 {target_city}",
        outcome_name="next_itinerary_city_match_rate",
        evaluation_design=evaluation_design,
        profiles=tuple(profiles),
        positive_user_ids=frozenset(positive_ids),
        promotion=promotion,
        intent=intent,
    )


@dataclass(frozen=True, slots=True)
class _AirbnbUser:
    user_id: str
    country_destination: str
    age_group: str
    gender: str
    affiliate_channel: str


@dataclass(slots=True)
class _AirbnbSignals:
    event_count: int = 0
    search_count: int = 0
    click_count: int = 0
    detail_count: int = 0


_AIRBNB_OUTCOME_ACTION_TYPES = {
    "booking_request",
    "booking_response",
    "partner_callback",
}


def load_airbnb_dataset(
    source_dir: Path,
    *,
    config: ExternalAdapterConfig,
) -> ExternalDatasetBundle:
    users_path = source_dir / "train_users_2.csv.zip"
    sessions_path = source_dir / "sessions.csv.zip"
    candidate_limit = max(config.profile_pool_limit * 5, config.profile_pool_limit)
    users = _read_airbnb_users(users_path, config=config, limit=candidate_limit)
    signals: dict[str, _AirbnbSignals] = defaultdict(_AirbnbSignals)
    with _zipped_dict_reader(sessions_path, "sessions.csv") as reader:
        for row in reader:
            user_id = _nonempty(row.get("user_id"))
            if user_id not in users:
                continue
            action_type = _nonempty(row.get("action_type")).lower()
            if action_type in _AIRBNB_OUTCOME_ACTION_TYPES:
                continue
            action = _nonempty(row.get("action")).lower()
            action_detail = _nonempty(row.get("action_detail")).lower()
            signal = signals[user_id]
            signal.event_count += 1
            if "search" in action or action_detail == "view_search_results":
                signal.search_count += 1
            if action_type == "click":
                signal.click_count += 1
            if action in {"show", "lookup"} or action_detail in {
                "p3",
                "view_listing",
                "view_search_results",
            }:
                signal.detail_count += 1

    selected_user_ids = sorted(
        (user_id for user_id, signal in signals.items() if signal.event_count > 0),
        key=stable_score,
    )[: config.profile_pool_limit]
    profiles: list[RawEventUserSignalRecord] = []
    positive_ids: set[str] = set()
    for user_id in selected_user_ids:
        user = users[user_id]
        signal = signals[user_id]
        external_user_id = f"airbnb-user-{user_id}"
        profiles.append(
            _profile(
                project_id="external-airbnb",
                user_id=external_user_id,
                event_count=signal.event_count,
                hotel_search_count=signal.search_count,
                hotel_click_count=signal.click_count,
                hotel_detail_view_count=signal.detail_count,
                age_group_values=(user.age_group,) if user.age_group else (),
                gender_values=(user.gender,) if user.gender else (),
                preferred_category_values=(
                    f"affiliate:{user.affiliate_channel}",
                )
                if user.affiliate_channel
                else (),
            )
        )
        if user.country_destination and user.country_destination != "NDF":
            positive_ids.add(external_user_id)
    if len(profiles) < config.min_scenario_users:
        raise ExternalBacktestError(
            "Airbnb session sampling produced too few observable users"
        )
    scenario_id = "airbnb-search-active-booking"
    promotion = _promotion(
        dataset_id="airbnb",
        scenario_id=scenario_id,
        min_sample_size=config.min_scenario_users,
        message_brief="숙소를 탐색한 신규 사용자의 첫 예약을 유도하는 프로모션",
    )
    intent = PromotionIntent(
        summary="숙소 탐색 사용자의 첫 예약 유도",
        product="hotel",
        season=(),
        destinations=(),
        benefits=(),
        audience_hints=(),
        channel="email",
        goal_metric="booking_conversion_rate",
        funnel_goal="booking_complete",
        desired_behaviors=("hotel_detail_view", "recent_destination_search"),
        explicit_conditions=("hotel", "search", "booking"),
        source="external_backtest_ground_truth",
    )
    case = ExternalEvaluationCase(
        dataset_id="airbnb",
        scenario_id=scenario_id,
        target_value="first_booking",
        target_label="첫 숙소 예약",
        outcome_name="observed_first_booking_rate",
        evaluation_design="static_outcome_label_holdout",
        profiles=tuple(profiles),
        positive_user_ids=frozenset(positive_ids),
        promotion=promotion,
        intent=intent,
    )
    manifest = ExternalDatasetManifest(
        dataset_id="airbnb",
        source_version="airbnb.new-user-bookings.2015",
        evaluation_design="static_outcome_label_holdout",
        outcome_name="observed_first_booking_rate",
        supports_temporal_holdout=False,
        supported_claims=(
            "검색·클릭·상세 탐색 신호가 있는 후보의 관측된 첫 예약률",
            "행동 활성 세그먼트와 전체 사용자 baseline의 정적 결과 차이",
        ),
        unsupported_claims=(
            "행동 이후 특정 기간 안에 발생한 미래 예약 전환율",
            "특정 목적지 프로모션과 사용자 목적지 관심의 정합성",
            "프로모션 노출로 발생한 증분 전환",
        ),
        signal_mappings={
            "hotel_search_count": {
                "source": "action/action_detail containing search",
                "support": "derived",
            },
            "hotel_click_count": {
                "source": "action_type=click",
                "support": "direct",
            },
            "hotel_detail_view_count": {
                "source": "show/lookup/listing detail actions",
                "support": "derived",
            },
            "outcome": {
                "source": "country_destination != NDF",
                "support": "direct",
            },
        },
        source_files=tuple(
            source_file_descriptor(path, include_checksum=config.include_checksum)
            for path in (users_path, sessions_path)
        ),
        evaluation_role=config.evaluation_role,
        prediction_error_comparability_reason=(
            "Airbnb static first-booking label has no matching observation window "
            "for the Expedia model target"
        ),
        notes=(
            "세션 행에 절대 이벤트 시각이 없어 temporal backtest로 해석하지 않습니다.",
            "booking_request/booking_response/partner_callback은 결과 누수를 줄이기 위해 관찰 feature에서 제외합니다.",
            "country_destination은 outcome으로만 사용하고 행동 profile에는 넣지 않습니다.",
        ),
    )
    return ExternalDatasetBundle(manifest=manifest, cases=(case,))


def _read_airbnb_users(
    path: Path,
    *,
    config: ExternalAdapterConfig,
    limit: int,
) -> dict[str, _AirbnbUser]:
    sampler: _BoundedStableSample[_AirbnbUser] = _BoundedStableSample(limit)
    with _zipped_dict_reader(path, "train_users_2.csv") as reader:
        for row in reader:
            user_id = _nonempty(row.get("id"))
            if not user_id:
                continue
            if not config.includes_subject(user_id):
                continue
            user = _AirbnbUser(
                user_id=user_id,
                country_destination=_nonempty(row.get("country_destination")),
                age_group=_age_group(_nonempty(row.get("age"))),
                gender=_normalized_gender(_nonempty(row.get("gender"))),
                affiliate_channel=_nonempty(row.get("affiliate_channel")),
            )
            sampler.add(user_id, user)
    return {user.user_id: user for user in sampler.values()}


@dataclass(slots=True)
class _SyneriseSignals:
    event_count: int = 0
    search_count: int = 0
    add_count: int = 0
    remove_count: int = 0
    buy_count: int = 0
    sku_counts: Counter[int] = field(default_factory=Counter)
    future_buy_skus: set[int] = field(default_factory=set)


def load_synerise_dataset(
    source_dir: Path,
    *,
    config: ExternalAdapterConfig,
) -> ExternalDatasetBundle:
    try:
        import pyarrow.dataset as arrow_dataset
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise ExternalBacktestError(
            "Synerise evaluation requires pyarrow; install the project dev extras"
        ) from exc

    paths = {
        "search": source_dir / "search_query.parquet",
        "add": source_dir / "add_to_cart.parquet",
        "remove": source_dir / "remove_from_cart.parquet",
        "buy": source_dir / "product_buy.parquet",
        "properties": source_dir / "product_properties.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise ExternalBacktestError(f"Synerise source file not found: {path}")
    observation_start = config.cutoff - timedelta(days=config.lookback_days)
    outcome_end = config.cutoff + timedelta(days=config.outcome_days)
    selected_client_ids = _select_synerise_clients(
        tuple((event_kind, paths[event_kind]) for event_kind in ("search", "add", "remove", "buy")),
        config=config,
        selection_start=max(
            observation_start,
            config.cutoff - timedelta(days=SYNERISE_SELECTION_RECENCY_DAYS),
        ),
        arrow_dataset=arrow_dataset,
    )
    selected_client_id_set = set(selected_client_ids)
    signals: dict[int, _SyneriseSignals] = defaultdict(_SyneriseSignals)

    _scan_synerise_events(
        paths["search"],
        event_kind="search",
        signals=signals,
        selected_client_ids=selected_client_id_set,
        config=config,
        observation_start=observation_start,
        outcome_end=outcome_end,
        arrow_dataset=arrow_dataset,
    )
    for event_kind in ("add", "remove", "buy"):
        _scan_synerise_events(
            paths[event_kind],
            event_kind=event_kind,
            signals=signals,
            selected_client_ids=selected_client_id_set,
            config=config,
            observation_start=observation_start,
            outcome_end=outcome_end,
            arrow_dataset=arrow_dataset,
        )
    selected_client_ids = [
        client_id
        for client_id in selected_client_ids
        if signals[client_id].event_count > 0
    ]
    relevant_skus = {
        sku
        for client_id in selected_client_ids
        for sku in (
            set(signals[client_id].sku_counts)
            | signals[client_id].future_buy_skus
        )
    }
    sku_properties = _load_synerise_properties(
        paths["properties"],
        relevant_skus=relevant_skus,
        parquet=parquet,
    )
    category_users: Counter[str] = Counter()
    for client_id in selected_client_ids:
        categories = {
            str(sku_properties[sku][0])
            for sku in signals[client_id].sku_counts
            if sku in sku_properties
        }
        category_users.update(categories)
    target_categories = _top_targets(
        category_users,
        limit=config.max_scenarios,
        min_users=config.min_scenario_users,
    )
    cases = tuple(
        _synerise_case(
            target_category=category,
            client_ids=selected_client_ids,
            signals=signals,
            sku_properties=sku_properties,
            config=config,
        )
        for category in target_categories
    )
    manifest = ExternalDatasetManifest(
        dataset_id="synerise",
        source_version="synerise.recsys-2025.raw",
        evaluation_design="time_window_purchase_holdout",
        outcome_name="future_target_category_purchase_rate",
        supports_temporal_holdout=True,
        supported_claims=(
            "검색·장바구니·구매 이력으로 미래 구매 가능성이 높은 후보를 찾는 능력",
            "퍼널 이탈형·가격 민감형·카테고리 반복 관심형 후보의 Rank 품질",
            "시간 분리된 미래 구매율과 예상값의 차이 및 baseline 대비 lift",
        ),
        unsupported_claims=(
            "숙박 목적지 또는 호텔에 대한 도메인 정합성",
            "실제 광고 노출로 발생한 증분 구매 효과",
            "page_visit URL을 특정 상품 상세 조회로 해석한 결과",
        ),
        signal_mappings={
            "hotel_search_count": {
                "source": "search_query count",
                "support": "cross_domain_proxy",
            },
            "booking_start_count": {
                "source": "add_to_cart count",
                "support": "cross_domain_proxy",
            },
            "booking_complete_count": {
                "source": "product_buy count in observation window",
                "support": "cross_domain_proxy",
            },
            "booking_cancel_count": {
                "source": "remove_from_cart count",
                "support": "cross_domain_proxy",
            },
            "destination_values": {
                "source": "product category IDs",
                "support": "cross_domain_proxy",
            },
            "price_event_count": {
                "source": "interactions joined with product price bucket",
                "support": "derived",
            },
            "outcome": {
                "source": "product_buy in target category after cutoff",
                "support": "direct",
            },
        },
        source_files=tuple(
            source_file_descriptor(path, include_checksum=config.include_checksum)
            for path in paths.values()
        ),
        evaluation_role=config.evaluation_role,
        prediction_error_comparability_reason=(
            "Synerise retail category purchase is a cross-domain proxy, not an "
            "Expedia future hotel booking target"
        ),
        notes=(
            f"관찰 구간은 {observation_start.isoformat()}부터 {config.cutoff.isoformat()} 직전까지입니다.",
            f"결과 구간은 {config.cutoff.isoformat()}부터 {outcome_end.isoformat()} 직전까지입니다.",
            f"profile pool은 cutoff 이전 최근 {SYNERISE_SELECTION_RECENCY_DAYS}일 내 활동 사용자에서 선택합니다.",
            "page_visit는 URL과 SKU 관계가 없어 feature에서 제외합니다.",
            "리테일 신호는 후보 생성·랭킹 구조의 외부 검증이며 숙박 성능 근거로 합산하지 않습니다.",
        ),
    )
    return ExternalDatasetBundle(manifest=manifest, cases=cases)


def _select_synerise_clients(
    event_paths: Sequence[tuple[str, Path]],
    *,
    config: ExternalAdapterConfig,
    selection_start: datetime,
    arrow_dataset: Any,
) -> list[int]:
    sampler: _BoundedStableSample[int] = _BoundedStableSample(
        config.profile_pool_limit
    )
    start_text = selection_start.strftime("%Y-%m-%d %H:%M:%S")
    cutoff_text = config.cutoff.strftime("%Y-%m-%d %H:%M:%S")
    for _, path in event_paths:
        scanner = arrow_dataset.dataset(path, format="parquet").scanner(
            columns=["client_id"],
            filter=(arrow_dataset.field("timestamp") >= start_text)
            & (arrow_dataset.field("timestamp") < cutoff_text),
            batch_size=262_144,
            use_threads=True,
        )
        for batch in scanner.to_batches():
            for client_id in batch.column(0).to_pylist():
                if not config.includes_subject(str(client_id)):
                    continue
                sampler.add(str(client_id), int(client_id))
    clients = sampler.values()
    if not clients:
        raise ExternalBacktestError("Synerise sampling produced no clients")
    return clients


def _scan_synerise_events(
    path: Path,
    *,
    event_kind: str,
    signals: dict[int, _SyneriseSignals],
    selected_client_ids: set[int],
    config: ExternalAdapterConfig,
    observation_start: datetime,
    outcome_end: datetime,
    arrow_dataset: Any,
) -> None:
    columns = ["client_id", "timestamp"]
    if event_kind != "search":
        columns.append("sku")
    start_text = observation_start.strftime("%Y-%m-%d %H:%M:%S")
    end_text = outcome_end.strftime("%Y-%m-%d %H:%M:%S")
    scanner = arrow_dataset.dataset(path, format="parquet").scanner(
        columns=columns,
        filter=(arrow_dataset.field("timestamp") >= start_text)
        & (arrow_dataset.field("timestamp") < end_text),
        batch_size=131_072,
        use_threads=True,
    )
    cutoff_text = config.cutoff.strftime("%Y-%m-%d %H:%M:%S")
    for batch in scanner.to_batches():
        values = batch.to_pydict()
        skus = values.get("sku", [None] * len(values["client_id"]))
        for client_id, timestamp, sku in zip(
            values["client_id"],
            values["timestamp"],
            skus,
            strict=True,
        ):
            if client_id not in selected_client_ids:
                continue
            signal = signals[client_id]
            if timestamp < cutoff_text:
                signal.event_count += 1
                if event_kind == "search":
                    signal.search_count += 1
                elif event_kind == "add":
                    signal.add_count += 1
                elif event_kind == "remove":
                    signal.remove_count += 1
                elif event_kind == "buy":
                    signal.buy_count += 1
                if sku is not None:
                    signal.sku_counts[int(sku)] += 1
            elif event_kind == "buy" and sku is not None:
                signal.future_buy_skus.add(int(sku))


def _load_synerise_properties(
    path: Path,
    *,
    relevant_skus: set[int],
    parquet: Any,
) -> dict[int, tuple[int, int]]:
    properties: dict[int, tuple[int, int]] = {}
    if not relevant_skus:
        return properties
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(
        columns=["sku", "category", "price"],
        batch_size=131_072,
    ):
        values = batch.to_pydict()
        for sku, category, price in zip(
            values["sku"],
            values["category"],
            values["price"],
            strict=True,
        ):
            if sku in relevant_skus:
                properties[int(sku)] = (int(category), int(price))
    return properties


def _synerise_case(
    *,
    target_category: str,
    client_ids: Sequence[int],
    signals: Mapping[int, _SyneriseSignals],
    sku_properties: Mapping[int, tuple[int, int]],
    config: ExternalAdapterConfig,
) -> ExternalEvaluationCase:
    profiles: list[RawEventUserSignalRecord] = []
    positive_ids: set[str] = set()
    target_category_id = int(target_category)
    for client_id in client_ids:
        signal = signals[client_id]
        category_counts: Counter[int] = Counter()
        prices: list[int] = []
        for sku, count in signal.sku_counts.items():
            properties = sku_properties.get(sku)
            if properties is None:
                continue
            category, price = properties
            category_counts[category] += count
            prices.extend([price] * count)
        user_id = f"synerise-client-{client_id}"
        profiles.append(
            _profile(
                project_id="external-synerise",
                user_id=user_id,
                event_count=signal.event_count,
                hotel_search_count=signal.search_count,
                booking_start_count=signal.add_count,
                booking_complete_count=signal.buy_count,
                booking_cancel_count=signal.remove_count,
                price_event_count=len(prices),
                avg_price=sum(prices) / len(prices) if prices else 0.0,
                destination_values=tuple(str(value) for value in category_counts),
                hotel_cluster_values=tuple(
                    str(value) for value in signal.sku_counts
                )[:20],
                preferred_category_values=tuple(
                    f"category:{value}" for value in category_counts
                )[:20],
                destination_match_count=category_counts[target_category_id],
            )
        )
        if any(
            sku_properties.get(sku, (-1, -1))[0] == target_category_id
            for sku in signal.future_buy_skus
        ):
            positive_ids.add(user_id)
    scenario_id = f"synerise-category-{target_category}"
    promotion, intent = _destination_promotion(
        dataset_id="synerise",
        scenario_id=scenario_id,
        target_value=target_category,
        min_sample_size=config.min_scenario_users,
        product="retail_product",
        benefits=("price",),
    )
    return ExternalEvaluationCase(
        dataset_id="synerise",
        scenario_id=scenario_id,
        target_value=target_category,
        target_label=f"상품 카테고리 {target_category}",
        outcome_name="future_target_category_purchase_rate",
        evaluation_design="time_window_purchase_holdout",
        profiles=tuple(profiles),
        positive_user_ids=frozenset(positive_ids),
        promotion=promotion,
        intent=intent,
    )


T = TypeVar("T")


class _BoundedStableSample(Iterable[T]):
    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("stable sample limit must be positive")
        self._limit = limit
        self._heap: list[tuple[int, str, T]] = []
        self._keys: set[str] = set()

    def add(self, key: str, value: T) -> None:
        if key in self._keys:
            return
        score = stable_score(key)
        entry = (-score, key, value)
        if len(self._heap) < self._limit:
            heapq.heappush(self._heap, entry)
            self._keys.add(key)
            return
        worst_score = -self._heap[0][0]
        if score < worst_score:
            _, removed_key, _ = self._heap[0]
            heapq.heapreplace(self._heap, entry)
            self._keys.remove(removed_key)
            self._keys.add(key)

    def values(self) -> list[T]:
        return [
            value
            for _, _, value in sorted(
                self._heap,
                key=lambda entry: (-entry[0], entry[1]),
            )
        ]

    def __iter__(self) -> Iterator[T]:
        return iter(self.values())


class _ZipCsvContext:
    def __init__(self, path: Path, expected_member: str) -> None:
        self._path = path
        self._expected_member = expected_member
        self._archive: zipfile.ZipFile | None = None
        self._raw: Any = None
        self._text: io.TextIOWrapper | None = None

    def __enter__(self) -> csv.DictReader:
        if not self._path.is_file():
            raise ExternalBacktestError(f"Airbnb source file not found: {self._path}")
        self._archive = zipfile.ZipFile(self._path)
        members = self._archive.namelist()
        if self._expected_member not in members:
            raise ExternalBacktestError(
                f"Airbnb archive missing {self._expected_member}: {self._path}"
            )
        self._raw = self._archive.open(self._expected_member)
        self._text = io.TextIOWrapper(self._raw, encoding="utf-8-sig", newline="")
        return csv.DictReader(self._text)

    def __exit__(self, *_: object) -> None:
        if self._text is not None:
            self._text.close()
        if self._archive is not None:
            self._archive.close()


def _zipped_dict_reader(path: Path, member: str) -> _ZipCsvContext:
    return _ZipCsvContext(path, member)


def _destination_promotion(
    *,
    dataset_id: str,
    scenario_id: str,
    target_value: str,
    min_sample_size: int,
    product: str,
    benefits: tuple[str, ...] = (),
) -> tuple[PromotionRecord, PromotionIntent]:
    promotion = _promotion(
        dataset_id=dataset_id,
        scenario_id=scenario_id,
        min_sample_size=min_sample_size,
        message_brief=f"{target_value} 조건과 관련된 전환을 유도하는 검증 프로모션",
    )
    intent = PromotionIntent(
        summary=f"{target_value} 조건 전환 프로모션",
        product=product,
        season=(),
        destinations=(target_value,),
        benefits=benefits,
        audience_hints=(),
        channel="email",
        goal_metric="booking_conversion_rate",
        funnel_goal="booking_complete",
        desired_behaviors=(
            "recent_destination_search",
            "booking_start_without_complete",
            "price_sensitive",
        ),
        explicit_conditions=(target_value, product, *benefits),
        source="external_backtest_ground_truth",
    )
    return promotion, intent


def _promotion(
    *,
    dataset_id: str,
    scenario_id: str,
    min_sample_size: int,
    message_brief: str,
) -> PromotionRecord:
    return PromotionRecord(
        project_id=f"external-{dataset_id}",
        campaign_id=f"backtest-{dataset_id}",
        promotion_id=f"promo-{scenario_id}",
        channel="email",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.10"),
        goal_basis="all_segments",
        min_sample_size=min_sample_size,
        landing_url=f"https://backtest.local/{dataset_id}/{scenario_id}",
        message_brief=message_brief,
    )


def _profile(
    *,
    project_id: str,
    user_id: str,
    event_count: int,
    hotel_search_count: int = 0,
    hotel_click_count: int = 0,
    hotel_detail_view_count: int = 0,
    booking_start_count: int = 0,
    booking_complete_count: int = 0,
    booking_cancel_count: int = 0,
    price_event_count: int = 0,
    avg_price: float = 0.0,
    destination_values: tuple[str, ...] = (),
    hotel_market_values: tuple[str, ...] = (),
    hotel_cluster_values: tuple[str, ...] = (),
    age_group_values: tuple[str, ...] = (),
    gender_values: tuple[str, ...] = (),
    preferred_category_values: tuple[str, ...] = (),
    destination_match_count: int = 0,
) -> RawEventUserSignalRecord:
    return RawEventUserSignalRecord(
        project_id=project_id,
        user_id=user_id,
        event_count=event_count,
        hotel_search_count=hotel_search_count,
        hotel_click_count=hotel_click_count,
        hotel_detail_view_count=hotel_detail_view_count,
        promotion_impression_count=0,
        promotion_click_count=0,
        campaign_redirect_click_count=0,
        campaign_landing_count=0,
        booking_start_count=booking_start_count,
        booking_complete_count=booking_complete_count,
        booking_cancel_count=booking_cancel_count,
        deal_event_count=0,
        free_cancellation_count=0,
        breakfast_included_count=0,
        price_event_count=price_event_count,
        avg_price=avg_price,
        destination_values=destination_values,
        checkin_dates=(),
        hotel_market_values=hotel_market_values,
        hotel_cluster_values=hotel_cluster_values,
        age_group_values=age_group_values,
        gender_values=gender_values,
        preferred_category_values=preferred_category_values,
        destination_match_count=destination_match_count,
        season_match_count=0,
    )


def _top_targets(
    counts: Counter[str],
    *,
    limit: int,
    min_users: int,
) -> tuple[str, ...]:
    targets = [
        value
        for value, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= min_users
    ][:limit]
    if not targets:
        raise ExternalBacktestError(
            "no evaluation targets meet min_scenario_users; lower the "
            "threshold or increase the profile pool"
        )
    return tuple(targets)


def _age_group(value: str) -> str:
    try:
        age = int(float(value))
    except (TypeError, ValueError):
        return ""
    if not 14 <= age <= 100:
        return ""
    lower = age // 10 * 10
    return f"{lower}s"


def _normalized_gender(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"male", "female", "other"}:
        return normalized
    return ""


def _nonempty(value: object) -> str:
    return str(value or "").strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
