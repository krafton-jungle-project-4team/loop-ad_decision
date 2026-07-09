from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, Sequence
from urllib.parse import parse_qs, urlparse

from app.analysis.repositories import (
    PromotionRecord,
    RawEventUserSignalRecord,
    SegmentDefinitionRecord,
    UserBehaviorVectorRecord,
)
from app.analysis.raw_event_segments import (
    PromotionIntentExtractor,
    compile_raw_event_intent,
    destination_terms_from_intent,
    generate_raw_event_segment_definitions,
    season_months_from_intent,
)
from app.analysis.vector_service import DEFAULT_VECTOR_VERSION, VECTOR_DIM
from app.logging import log, log_context_scope, now_ms, duration_ms


DEFAULT_VECTOR_POOL_LIMIT = 1000
DEFAULT_VECTOR_SAMPLE_LIMIT = 200
DEFAULT_MAX_SUGGESTED_SEGMENTS = 3
DEFAULT_MIN_CLUSTER_SIZE = 2
KMEANS_ITERATIONS = 8

FEATURE_LABELS = {
    0: "Hotel page viewers",
    1: "Hotel search users",
    2: "Hotel click users",
    3: "Hotel detail viewers",
    4: "Promotion impression users",
    5: "Promotion click responders",
    6: "Campaign redirect users",
    7: "Campaign landing users",
    8: "Booking starters",
    9: "Booking converters",
    10: "Booking cancellation risk users",
    11: "Mixed event hotel users",
    56: "Promotion-engaged hotel users",
    57: "Experiment-exposed hotel users",
    58: "Segment-tagged hotel users",
    59: "Free cancellation seekers",
    60: "Breakfast-included seekers",
    61: "Higher-price hotel shoppers",
    62: "Booking conversion ready users",
    63: "Promotion click responsive users",
}

FEATURE_HOTEL_PAGE_VIEW = 0
FEATURE_HOTEL_SEARCH = 1
FEATURE_HOTEL_CLICK = 2
FEATURE_HOTEL_DETAIL = 3
FEATURE_PROMOTION_IMPRESSION = 4
FEATURE_PROMOTION_CLICK = 5
FEATURE_CAMPAIGN_REDIRECT = 6
FEATURE_CAMPAIGN_LANDING = 7
FEATURE_BOOKING_START = 8
FEATURE_BOOKING_COMPLETE = 9
FEATURE_CANCEL_RISK = 10
FEATURE_MIXED_HOTEL_EVENT = 11
FEATURE_PROMOTION_ENGAGED = 56
FEATURE_EXPERIMENT_EXPOSED = 57
FEATURE_SEGMENT_TAGGED = 58
FEATURE_FREE_CANCELLATION = 59
FEATURE_BREAKFAST_INCLUDED = 60
FEATURE_HIGHER_PRICE = 61
FEATURE_BOOKING_READY = 62
FEATURE_PROMOTION_CLICK_RESPONSIVE = 63

GOAL_FEATURE_WEIGHTS: Mapping[str, Mapping[int, float]] = {
    "booking_conversion_rate": {
        FEATURE_BOOKING_READY: 1.0,
        FEATURE_BOOKING_COMPLETE: 0.85,
        FEATURE_BOOKING_START: 0.75,
        FEATURE_HOTEL_DETAIL: 0.55,
        FEATURE_HOTEL_SEARCH: 0.35,
    },
    "inflow_rate": {
        FEATURE_CAMPAIGN_REDIRECT: 0.9,
        FEATURE_CAMPAIGN_LANDING: 0.85,
        FEATURE_HOTEL_PAGE_VIEW: 0.65,
        FEATURE_HOTEL_SEARCH: 0.5,
        FEATURE_PROMOTION_ENGAGED: 0.45,
    },
    "funnel_step_rate": {
        FEATURE_HOTEL_SEARCH: 0.75,
        FEATURE_HOTEL_CLICK: 0.65,
        FEATURE_HOTEL_DETAIL: 0.65,
        FEATURE_BOOKING_START: 0.65,
        FEATURE_MIXED_HOTEL_EVENT: 0.4,
    },
}

CHANNEL_FEATURE_WEIGHTS: Mapping[str, Mapping[int, float]] = {
    "email": {
        FEATURE_CAMPAIGN_REDIRECT: 0.65,
        FEATURE_CAMPAIGN_LANDING: 0.65,
        FEATURE_PROMOTION_ENGAGED: 0.45,
    },
    "sms": {
        FEATURE_CAMPAIGN_REDIRECT: 0.55,
        FEATURE_CAMPAIGN_LANDING: 0.45,
        FEATURE_PROMOTION_ENGAGED: 0.35,
    },
    "onsite_banner": {
        FEATURE_PROMOTION_IMPRESSION: 0.75,
        FEATURE_PROMOTION_CLICK: 0.75,
        FEATURE_PROMOTION_CLICK_RESPONSIVE: 0.55,
        FEATURE_EXPERIMENT_EXPOSED: 0.35,
    },
}

MESSAGE_KEYWORD_FEATURE_WEIGHTS: tuple[
    tuple[tuple[str, ...], Mapping[int, float], str],
    ...,
] = (
    (
        ("booking", "reservation", "book", "예약", "전환", "구매"),
        {
            FEATURE_BOOKING_READY: 0.85,
            FEATURE_BOOKING_START: 0.55,
            FEATURE_BOOKING_COMPLETE: 0.45,
        },
        "booking_intent",
    ),
    (
        ("search", "explore", "browse", "검색", "탐색", "둘러"),
        {
            FEATURE_HOTEL_SEARCH: 0.7,
            FEATURE_HOTEL_DETAIL: 0.45,
            FEATURE_HOTEL_PAGE_VIEW: 0.35,
        },
        "hotel_discovery",
    ),
    (
        ("click", "redirect", "landing", "클릭", "링크", "랜딩", "유입"),
        {
            FEATURE_CAMPAIGN_REDIRECT: 0.7,
            FEATURE_CAMPAIGN_LANDING: 0.65,
            FEATURE_PROMOTION_CLICK_RESPONSIVE: 0.45,
        },
        "campaign_response",
    ),
    (
        ("free cancellation", "flexible cancellation", "무료 취소", "취소"),
        {
            FEATURE_FREE_CANCELLATION: 0.8,
            FEATURE_CANCEL_RISK: 0.35,
        },
        "free_cancellation",
    ),
    (
        ("breakfast", "조식"),
        {
            FEATURE_BREAKFAST_INCLUDED: 0.75,
        },
        "breakfast_included",
    ),
    (
        ("premium", "luxury", "고급", "럭셔리", "프리미엄"),
        {
            FEATURE_HIGHER_PRICE: 0.7,
        },
        "premium_hotel",
    ),
    (
        ("summer", "deal", "discount", "special", "여름", "특가", "할인", "혜택"),
        {
            FEATURE_HOTEL_SEARCH: 0.45,
            FEATURE_CAMPAIGN_LANDING: 0.45,
            FEATURE_PROMOTION_ENGAGED: 0.35,
        },
        "seasonal_deal",
    ),
)

LOCATION_KEYWORDS = ("jeju", "busan", "seoul", "제주", "부산", "서울", "강릉", "여수")
HOTEL_STYLE_KEYWORDS = (
    "family",
    "couple",
    "beach",
    "ocean",
    "resort",
    "가족",
    "커플",
    "바다",
    "오션",
    "리조트",
)


class UserBehaviorVectorSampler(Protocol):
    def list_recent(
        self,
        *,
        project_id: str,
        limit: int = DEFAULT_VECTOR_POOL_LIMIT,
        vector_version: str = DEFAULT_VECTOR_VERSION,
    ) -> list[UserBehaviorVectorRecord]:
        ...


class RawEventUserSignalSampler(Protocol):
    def list_raw_event_user_signals(
        self,
        *,
        project_id: str,
        destination_terms: Sequence[str] = (),
        season_months: Sequence[int] = (),
        limit: int = DEFAULT_VECTOR_POOL_LIMIT,
    ) -> list[RawEventUserSignalRecord]:
        ...


@dataclass(frozen=True)
class _UserVector:
    user_id: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class _Cluster:
    index: int
    users: tuple[_UserVector, ...]
    centroid: tuple[float, ...]
    score: float


@dataclass(frozen=True)
class _PromotionIntent:
    vector: tuple[float, ...]
    basis: Mapping[str, object]


@dataclass(frozen=True)
class _ScoredCluster:
    cluster: _Cluster
    promotion_similarity: float
    cluster_quality_score: float
    sample_size_score: float
    recommendation_score: float
    matched_features: tuple[str, ...]


class VectorClusterSegmentSuggester:
    def __init__(
        self,
        *,
        user_behavior_vector_repository: UserBehaviorVectorSampler,
        raw_event_signal_repository: RawEventUserSignalSampler | None = None,
        promotion_intent_extractor: PromotionIntentExtractor | None = None,
        vector_pool_limit: int = DEFAULT_VECTOR_POOL_LIMIT,
        vector_sample_limit: int = DEFAULT_VECTOR_SAMPLE_LIMIT,
        max_suggested_segments: int = DEFAULT_MAX_SUGGESTED_SEGMENTS,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        vector_version: str = DEFAULT_VECTOR_VERSION,
    ) -> None:
        if vector_pool_limit <= 0:
            raise ValueError("vector_pool_limit must be positive")
        if vector_sample_limit <= 0:
            raise ValueError("vector_sample_limit must be positive")
        if vector_pool_limit < vector_sample_limit:
            raise ValueError(
                "vector_pool_limit must be greater than or equal to vector_sample_limit"
            )
        if max_suggested_segments <= 0:
            raise ValueError("max_suggested_segments must be positive")
        if min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be positive")

        self._user_behavior_vector_repository = user_behavior_vector_repository
        self._raw_event_signal_repository = raw_event_signal_repository
        self._promotion_intent_extractor = promotion_intent_extractor
        self._vector_pool_limit = vector_pool_limit
        self._vector_sample_limit = vector_sample_limit
        self._max_suggested_segments = max_suggested_segments
        self._min_cluster_size = min_cluster_size
        self._vector_version = vector_version

    @log_context_scope
    def suggest_segments(self, *, promotion: PromotionRecord) -> list[SegmentDefinitionRecord]:
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": promotion.project_id,
                "campaignId": promotion.campaign_id,
                "promotionId": promotion.promotion_id,
            }
        )
        log.info("started", {"promotion": promotion})
        sample_seed = _promotion_sample_seed(promotion)
        raw_event_segments = self._suggest_raw_event_segments(promotion=promotion)
        if len(raw_event_segments) >= self._max_suggested_segments:
            log.info(
                "raw_event_intent_segments_created",
                {
                    "suggestedSegmentCount": len(raw_event_segments),
                    "durationMs": duration_ms(started_at),
                },
            )
            log.info(
                "completed",
                {"response": raw_event_segments, "durationMs": duration_ms(started_at)},
            )
            return raw_event_segments[: self._max_suggested_segments]

        user_vectors = self._load_user_vectors(promotion, sample_seed)
        if len(user_vectors) < self._min_cluster_size:
            log.warn("user_vector_sample_insufficient", {"userVectorCount": len(user_vectors), "minClusterSize": self._min_cluster_size})
            log.info(
                "completed",
                {"response": raw_event_segments, "durationMs": duration_ms(started_at)},
            )
            return raw_event_segments

        cluster_count = min(
            self._max_suggested_segments,
            max(1, len(user_vectors) // self._min_cluster_size),
        )
        clusters = _cluster_user_vectors(user_vectors, cluster_count)
        if not clusters:
            log.warn("vector_clusters_empty", {"userVectorCount": len(user_vectors)})
            log.info(
                "completed",
                {"response": raw_event_segments, "durationMs": duration_ms(started_at)},
            )
            return raw_event_segments

        total_eligible_user_count = len(user_vectors)
        promotion_intent = _promotion_intent(promotion)
        scored_clusters = _score_clusters(
            clusters=clusters,
            promotion_intent=promotion_intent,
            total_eligible_user_count=total_eligible_user_count,
        )
        response = [
            _segment_definition_from_cluster(
                promotion=promotion,
                scored_cluster=scored_cluster,
                rank=rank,
                total_eligible_user_count=total_eligible_user_count,
                sample_seed=sample_seed,
                vector_version=self._vector_version,
                promotion_vector_basis=promotion_intent.basis,
            )
            for rank, scored_cluster in enumerate(
                sorted(
                    scored_clusters,
                    key=lambda scored_cluster: (
                        -scored_cluster.recommendation_score,
                        -scored_cluster.promotion_similarity,
                        -scored_cluster.cluster_quality_score,
                        -len(scored_cluster.cluster.users),
                        scored_cluster.cluster.index,
                    ),
                )
            )
            if len(scored_cluster.cluster.users) >= self._min_cluster_size
        ][: self._max_suggested_segments]
        log.info(
            "vector_clusters_created",
            {
                "clusterCount": len(clusters),
                "suggestedSegmentCount": len(response),
                "totalEligibleUserCount": total_eligible_user_count,
                "promotionVectorBasis": promotion_intent.basis,
            },
        )
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        if raw_event_segments:
            raw_segment_ids = {segment.segment_id for segment in raw_event_segments}
            combined_segments = [
                *raw_event_segments,
                *[
                    segment
                    for segment in response
                    if segment.segment_id not in raw_segment_ids
                ],
            ][: self._max_suggested_segments]
            log.info(
                "raw_event_segments_combined_with_vector_fallback",
                {
                    "rawEventSegmentCount": len(raw_event_segments),
                    "vectorFallbackSegmentCount": len(combined_segments)
                    - len(raw_event_segments),
                },
            )
            return combined_segments
        return response

    def _suggest_raw_event_segments(
        self,
        *,
        promotion: PromotionRecord,
    ) -> list[SegmentDefinitionRecord]:
        if (
            self._raw_event_signal_repository is None
            or self._promotion_intent_extractor is None
        ):
            return []
        try:
            intent = self._promotion_intent_extractor.extract(promotion)
            compilation = compile_raw_event_intent(intent)
            profiles = self._raw_event_signal_repository.list_raw_event_user_signals(
                project_id=promotion.project_id,
                destination_terms=destination_terms_from_intent(intent),
                season_months=season_months_from_intent(intent),
                limit=self._vector_pool_limit,
            )
            log.info(
                "raw_event_user_signals_loaded",
                {
                    "userSignalCount": len(profiles),
                    "intent": intent.to_json(),
                    "compiledConditionCount": len(compilation.compiled_conditions),
                },
            )
            return generate_raw_event_segment_definitions(
                promotion=promotion,
                intent=intent,
                compilation=compilation,
                profiles=profiles[: self._vector_sample_limit],
                max_suggested_segments=self._max_suggested_segments,
                min_sample_size=self._min_cluster_size,
            )
        except Exception as exc:
            log.warn(
                "raw_event_intent_segments_failed",
                {
                    "promotionId": promotion.promotion_id,
                    "err": exc,
                },
            )
            return []

    def _load_user_vectors(
        self,
        promotion: PromotionRecord,
        sample_seed: str,
    ) -> list[_UserVector]:
        records = self._user_behavior_vector_repository.list_recent(
            project_id=promotion.project_id,
            limit=self._vector_pool_limit,
            vector_version=self._vector_version,
        )
        log.info("user_vectors_loaded", {"userVectorCount": len(records)})
        sampled_records = sorted(
            records,
            key=lambda record: (
                _sample_sort_digest(sample_seed=sample_seed, user_id=record.user_id),
                record.user_id,
            ),
        )[: self._vector_sample_limit]
        return [
            _UserVector(
                user_id=record.user_id,
                values=tuple(_l2_normalize(record.vector_values, record.vector_dim)),
            )
            for record in sampled_records
        ]


def _cluster_user_vectors(
    user_vectors: Sequence[_UserVector],
    cluster_count: int,
) -> list[_Cluster]:
    centroids = _initial_centroids(user_vectors, cluster_count)
    assignments: dict[int, list[_UserVector]] = {}
    for _ in range(KMEANS_ITERATIONS):
        assignments = {index: [] for index in range(len(centroids))}
        for user_vector in user_vectors:
            cluster_index = _nearest_centroid_index(user_vector.values, centroids)
            assignments[cluster_index].append(user_vector)

        next_centroids: list[tuple[float, ...]] = []
        for index, centroid in enumerate(centroids):
            assigned_vectors = assignments.get(index, [])
            next_centroids.append(
                _mean_vector([vector.values for vector in assigned_vectors])
                if assigned_vectors
                else centroid
            )
        if next_centroids == centroids:
            break
        centroids = next_centroids

    return [
        _Cluster(
            index=index,
            users=tuple(assigned_vectors),
            centroid=centroids[index],
            score=_mean_similarity(assigned_vectors, centroids[index]),
        )
        for index, assigned_vectors in assignments.items()
        if assigned_vectors
    ]


def _initial_centroids(
    user_vectors: Sequence[_UserVector],
    cluster_count: int,
) -> list[tuple[float, ...]]:
    centroids = [user_vectors[0].values]
    while len(centroids) < cluster_count:
        next_vector = max(
            user_vectors,
            key=lambda user_vector: (
                -min(
                    _cosine_similarity(user_vector.values, centroid)
                    for centroid in centroids
                ),
                user_vector.user_id,
            ),
        )
        if next_vector.values in centroids:
            break
        centroids.append(next_vector.values)
    return centroids


def _nearest_centroid_index(
    values: Sequence[float],
    centroids: Sequence[Sequence[float]],
) -> int:
    return max(
        range(len(centroids)),
        key=lambda index: (_cosine_similarity(values, centroids[index]), -index),
    )


def _mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    totals = [0.0] * VECTOR_DIM
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += float(value)
    try:
        return tuple(
            _l2_normalize([value / len(vectors) for value in totals], VECTOR_DIM)
        )
    except ValueError:
        return tuple(float(value) for value in vectors[0])


def _mean_similarity(
    user_vectors: Sequence[_UserVector],
    centroid: Sequence[float],
) -> float:
    if not user_vectors:
        return 0.0
    return sum(
        _cosine_similarity(user_vector.values, centroid)
        for user_vector in user_vectors
    ) / len(user_vectors)


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return sum(
        float(left_value) * float(right_value)
        for left_value, right_value in zip(left, right)
    )


def _promotion_intent(promotion: PromotionRecord) -> _PromotionIntent:
    weights = [0.0] * VECTOR_DIM
    reasons_by_feature: dict[int, list[str]] = {}
    matched_keywords: list[str] = []

    _apply_feature_weights(
        weights=weights,
        reasons_by_feature=reasons_by_feature,
        feature_weights=GOAL_FEATURE_WEIGHTS.get(promotion.goal_metric, {}),
        reason=f"goal_metric:{promotion.goal_metric}",
    )
    _apply_feature_weights(
        weights=weights,
        reasons_by_feature=reasons_by_feature,
        feature_weights=CHANNEL_FEATURE_WEIGHTS.get(promotion.channel, {}),
        reason=f"channel:{promotion.channel}",
    )
    _apply_landing_url_weights(
        promotion.landing_url,
        weights=weights,
        reasons_by_feature=reasons_by_feature,
        matched_keywords=matched_keywords,
    )
    _apply_message_brief_weights(
        promotion.message_brief,
        weights=weights,
        reasons_by_feature=reasons_by_feature,
        matched_keywords=matched_keywords,
    )

    if not any(weights):
        _add_feature_weight(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            index=FEATURE_HOTEL_SEARCH,
            weight=0.5,
            reason="default:hotel_search",
        )
        _add_feature_weight(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            index=FEATURE_HOTEL_DETAIL,
            weight=0.5,
            reason="default:hotel_detail",
        )

    normalized_vector = tuple(_l2_normalize(weights, VECTOR_DIM))
    weighted_features = [
        {
            "feature": _feature_label(index),
            "weight": round(weight, 6),
            "reasons": reasons_by_feature.get(index, []),
        }
        for index, weight in sorted(
            enumerate(weights),
            key=lambda item: (-abs(item[1]), item[0]),
        )
        if weight and _feature_label(index) is not None
    ][:8]
    return _PromotionIntent(
        vector=normalized_vector,
        basis={
            "channel": promotion.channel,
            "goal_metric": promotion.goal_metric,
            "goal_basis": promotion.goal_basis,
            "landing_url": promotion.landing_url,
            "message_keywords": matched_keywords[:12],
            "weighted_features": weighted_features,
        },
    )


def _apply_feature_weights(
    *,
    weights: list[float],
    reasons_by_feature: dict[int, list[str]],
    feature_weights: Mapping[int, float],
    reason: str,
) -> None:
    for index, weight in feature_weights.items():
        _add_feature_weight(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            index=index,
            weight=weight,
            reason=reason,
        )


def _apply_landing_url_weights(
    landing_url: str | None,
    *,
    weights: list[float],
    reasons_by_feature: dict[int, list[str]],
    matched_keywords: list[str],
) -> None:
    if not landing_url:
        return
    parsed_url = urlparse(landing_url)
    path = parsed_url.path.lower()
    query_values = " ".join(
        value.lower()
        for values in parse_qs(parsed_url.query).values()
        for value in values
    )
    searchable = f"{landing_url.lower()} {query_values}"

    if "/hotel" in path:
        _apply_feature_weights(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            feature_weights={
                FEATURE_HOTEL_DETAIL: 0.75,
                FEATURE_BOOKING_START: 0.45,
                FEATURE_BOOKING_READY: 0.35,
            },
            reason="landing_url:hotel_detail",
        )
    if "/search" in path:
        _apply_feature_weights(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            feature_weights={
                FEATURE_HOTEL_SEARCH: 0.75,
                FEATURE_HOTEL_CLICK: 0.35,
                FEATURE_HOTEL_PAGE_VIEW: 0.3,
            },
            reason="landing_url:search",
        )
    if "deal" in searchable or "summer" in searchable or "특가" in searchable:
        _apply_feature_weights(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            feature_weights={
                FEATURE_CAMPAIGN_LANDING: 0.4,
                FEATURE_PROMOTION_ENGAGED: 0.3,
            },
            reason="landing_url:deal",
        )

    _apply_text_bucket_weights(
        searchable,
        weights=weights,
        reasons_by_feature=reasons_by_feature,
        matched_keywords=matched_keywords,
        reason_prefix="landing_url",
    )


def _apply_message_brief_weights(
    message_brief: str | None,
    *,
    weights: list[float],
    reasons_by_feature: dict[int, list[str]],
    matched_keywords: list[str],
) -> None:
    if not message_brief:
        return
    searchable = message_brief.lower()
    for keywords, feature_weights, reason in MESSAGE_KEYWORD_FEATURE_WEIGHTS:
        matched = [keyword for keyword in keywords if keyword.lower() in searchable]
        if not matched:
            continue
        matched_keywords.extend(
            keyword for keyword in matched if keyword not in matched_keywords
        )
        _apply_feature_weights(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            feature_weights=feature_weights,
            reason=f"message_brief:{reason}",
        )

    _apply_text_bucket_weights(
        searchable,
        weights=weights,
        reasons_by_feature=reasons_by_feature,
        matched_keywords=matched_keywords,
        reason_prefix="message_brief",
    )


def _apply_text_bucket_weights(
    text: str,
    *,
    weights: list[float],
    reasons_by_feature: dict[int, list[str]],
    matched_keywords: list[str],
    reason_prefix: str,
) -> None:
    for keyword in LOCATION_KEYWORDS:
        if keyword.lower() not in text:
            continue
        if keyword not in matched_keywords:
            matched_keywords.append(keyword)
        _add_feature_weight(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            index=32 + _stable_bucket(keyword, 16),
            weight=0.45,
            reason=f"{reason_prefix}:location",
        )
    for keyword in HOTEL_STYLE_KEYWORDS:
        if keyword.lower() not in text:
            continue
        if keyword not in matched_keywords:
            matched_keywords.append(keyword)
        _add_feature_weight(
            weights=weights,
            reasons_by_feature=reasons_by_feature,
            index=16 + _stable_bucket(keyword, 16),
            weight=0.45,
            reason=f"{reason_prefix}:hotel_style",
        )


def _stable_bucket(value: str, bucket_count: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]  # noqa: S324
    return int(digest, 16) % bucket_count


def _add_feature_weight(
    *,
    weights: list[float],
    reasons_by_feature: dict[int, list[str]],
    index: int,
    weight: float,
    reason: str,
) -> None:
    weights[index] += weight
    reasons = reasons_by_feature.setdefault(index, [])
    if reason not in reasons:
        reasons.append(reason)


def _score_clusters(
    *,
    clusters: Sequence[_Cluster],
    promotion_intent: _PromotionIntent,
    total_eligible_user_count: int,
) -> list[_ScoredCluster]:
    return [
        _score_cluster(
            cluster=cluster,
            promotion_intent=promotion_intent,
            total_eligible_user_count=total_eligible_user_count,
        )
        for cluster in clusters
    ]


def _score_cluster(
    *,
    cluster: _Cluster,
    promotion_intent: _PromotionIntent,
    total_eligible_user_count: int,
) -> _ScoredCluster:
    promotion_similarity = max(
        0.0,
        _cosine_similarity(promotion_intent.vector, cluster.centroid),
    )
    cluster_quality_score = max(0.0, min(1.0, cluster.score))
    sample_size_score = (
        len(cluster.users) / total_eligible_user_count
        if total_eligible_user_count > 0
        else 0.0
    )
    recommendation_score = (
        0.65 * promotion_similarity
        + 0.20 * cluster_quality_score
        + 0.15 * sample_size_score
    )
    return _ScoredCluster(
        cluster=cluster,
        promotion_similarity=promotion_similarity,
        cluster_quality_score=cluster_quality_score,
        sample_size_score=sample_size_score,
        recommendation_score=recommendation_score,
        matched_features=tuple(
            _matched_feature_labels(
                promotion_vector=promotion_intent.vector,
                centroid=cluster.centroid,
            )
        ),
    )


def _matched_feature_labels(
    *,
    promotion_vector: Sequence[float],
    centroid: Sequence[float],
    limit: int = 4,
) -> list[str]:
    labels: list[str] = []
    for index, score in sorted(
        (
            (index, abs(float(intent_value)) * abs(float(cluster_value)))
            for index, (intent_value, cluster_value) in enumerate(
                zip(promotion_vector, centroid)
            )
        ),
        key=lambda item: (-item[1], item[0]),
    ):
        if score == 0:
            continue
        label = _feature_label(index)
        if label is None or label in labels:
            continue
        labels.append(label)
        if len(labels) == limit:
            break
    if labels:
        return labels
    return _top_feature_labels(centroid, limit=limit)


def _l2_normalize(
    vector_values: Sequence[float],
    vector_dim: int,
) -> list[float]:
    if vector_dim != VECTOR_DIM:
        raise ValueError("user behavior vector_dim must be 64")
    if len(vector_values) != VECTOR_DIM:
        raise ValueError("user behavior vector_values must contain 64 values")
    norm = math.sqrt(sum(float(value) * float(value) for value in vector_values))
    if norm == 0:
        raise ValueError("user behavior vector must not be a zero vector")
    return [float(value) / norm for value in vector_values]


def _segment_definition_from_cluster(
    *,
    promotion: PromotionRecord,
    scored_cluster: _ScoredCluster,
    rank: int,
    total_eligible_user_count: int,
    sample_seed: str,
    vector_version: str,
    promotion_vector_basis: Mapping[str, object],
) -> SegmentDefinitionRecord:
    cluster = scored_cluster.cluster
    segment_id = _suggested_segment_id(
        promotion_id=promotion.promotion_id,
        rank=rank,
        centroid=cluster.centroid,
    )
    sample_size = len(cluster.users)
    sample_ratio = _sample_ratio(
        sample_size=sample_size,
        total_eligible_user_count=total_eligible_user_count,
    )
    candidate_user_ids = [user_vector.user_id for user_vector in cluster.users]
    top_common_features = _top_feature_labels(cluster.centroid)
    promotion_matched_features = list(scored_cluster.matched_features)
    natural_language_query = (
        "Users grouped by similar hotel behavior vectors and ranked by "
        "promotion intent similarity."
    )
    if promotion_matched_features:
        natural_language_query = (
            f"{natural_language_query} Matched promotion signals: "
            f"{', '.join(promotion_matched_features)}."
        )
    elif top_common_features:
        natural_language_query = (
            f"{natural_language_query} Strongest signals: "
            f"{', '.join(top_common_features)}."
        )
    return SegmentDefinitionRecord(
        segment_id=segment_id,
        project_id=promotion.project_id,
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        segment_name=_segment_name_from_cluster(
            cluster=cluster,
            rank=rank,
            preferred_features=promotion_matched_features,
        ),
        source="ai_suggested",
        query_preview_id=None,
        natural_language_query=natural_language_query,
        generated_sql=None,
        rule_json={
            "source": "user_vector_clustering",
            "vector_version": vector_version,
            "cluster_index": cluster.index,
            "sample_seed": sample_seed,
            "candidate_user_ids": candidate_user_ids,
        },
        profile_json={
            "primary_segment": segment_id,
            "source": "user_vector_clustering",
            "vector_version": vector_version,
            "cluster_index": cluster.index,
            "cluster_score": round(cluster.score, 6),
            "promotion_cluster_similarity": round(
                scored_cluster.promotion_similarity,
                6,
            ),
            "cluster_quality_score": round(scored_cluster.cluster_quality_score, 6),
            "sample_size_score": round(scored_cluster.sample_size_score, 6),
            "recommendation_score": round(scored_cluster.recommendation_score, 6),
            "score_components": {
                "promotion_cluster_similarity": round(
                    scored_cluster.promotion_similarity,
                    6,
                ),
                "cluster_quality": round(scored_cluster.cluster_quality_score, 6),
                "sample_size": round(scored_cluster.sample_size_score, 6),
                "final_score": round(scored_cluster.recommendation_score, 6),
                "weights": {
                    "promotion_cluster_similarity": 0.65,
                    "cluster_quality": 0.20,
                    "sample_size": 0.15,
                },
            },
            "top_common_features": top_common_features,
            "promotion_matched_features": promotion_matched_features,
            "promotion_vector_basis": promotion_vector_basis,
            "promotion": {
                "channel": promotion.channel,
                "goal_metric": promotion.goal_metric,
            },
        },
        sample_size=sample_size,
        total_eligible_user_count=total_eligible_user_count,
        sample_ratio=sample_ratio,
        status="active",
    )


def _sample_ratio(
    *,
    sample_size: int,
    total_eligible_user_count: int,
) -> Decimal:
    if total_eligible_user_count <= 0:
        return Decimal("0")
    return Decimal(sample_size / total_eligible_user_count).quantize(Decimal("0.000001"))


def _promotion_sample_seed(promotion: PromotionRecord) -> str:
    seed_input = ":".join(
        [
            promotion.project_id,
            promotion.campaign_id,
            promotion.promotion_id,
            promotion.channel,
            promotion.goal_metric,
            promotion.message_brief or "",
            promotion.landing_url or "",
        ]
    )
    return hashlib.sha1(seed_input.encode("utf-8")).hexdigest()[:12]  # noqa: S324


def _sample_sort_digest(*, sample_seed: str, user_id: str) -> str:
    return hashlib.sha1(f"{sample_seed}:{user_id}".encode("utf-8")).hexdigest()  # noqa: S324


def _segment_name_from_cluster(
    *,
    cluster: _Cluster,
    rank: int,
    preferred_features: Sequence[str] | None = None,
) -> str:
    labels = list(preferred_features or [])[:2] or _top_feature_labels(
        cluster.centroid,
        limit=2,
    )
    if not labels:
        return f"Hotel behavior cluster {rank + 1}"
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]} with {labels[1].lower()}"


def _top_feature_labels(
    centroid: Sequence[float],
    *,
    limit: int = 3,
) -> list[str]:
    labels: list[str] = []
    for index, value in sorted(
        enumerate(centroid),
        key=lambda item: (-abs(float(item[1])), item[0]),
    ):
        if value == 0:
            continue
        label = _feature_label(index)
        if label is None or label in labels:
            continue
        labels.append(label)
        if len(labels) == limit:
            break
    return labels


def _feature_label(index: int) -> str | None:
    if index in FEATURE_LABELS:
        return FEATURE_LABELS[index]
    if 16 <= index <= 31:
        return f"Hotel cluster bucket {index - 16} affinity users"
    if 32 <= index <= 47:
        return f"Hotel market bucket {index - 32} affinity users"
    if 48 <= index <= 55:
        return f"Hotel page path bucket {index - 48} users"
    return None


def _suggested_segment_id(
    *,
    promotion_id: str,
    rank: int,
    centroid: Sequence[float],
) -> str:
    digest = hashlib.sha1(  # noqa: S324 - stable non-security identifier.
        ":".join(
            [
                promotion_id,
                str(rank),
                ",".join(f"{value:.6f}" for value in centroid[:8]),
            ]
        ).encode("utf-8")
    ).hexdigest()[:10]
    promotion_part = _safe_identifier_part(promotion_id)[:36]
    return f"seg_ai_cluster_{promotion_part}_{rank + 1}_{digest}"


def _safe_identifier_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in value
    )
