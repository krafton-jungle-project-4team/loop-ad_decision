from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, Sequence

from app.analysis.repositories import (
    PromotionRecord,
    SegmentDefinitionRecord,
    UserBehaviorVectorRecord,
)
from app.analysis.vector_service import DEFAULT_VECTOR_VERSION, VECTOR_DIM


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


class UserBehaviorVectorSampler(Protocol):
    def list_recent(
        self,
        *,
        project_id: str,
        limit: int = DEFAULT_VECTOR_POOL_LIMIT,
        vector_version: str = DEFAULT_VECTOR_VERSION,
    ) -> list[UserBehaviorVectorRecord]:
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


class VectorClusterSegmentSuggester:
    def __init__(
        self,
        *,
        user_behavior_vector_repository: UserBehaviorVectorSampler,
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
        self._vector_pool_limit = vector_pool_limit
        self._vector_sample_limit = vector_sample_limit
        self._max_suggested_segments = max_suggested_segments
        self._min_cluster_size = min_cluster_size
        self._vector_version = vector_version

    def suggest_segments(self, *, promotion: PromotionRecord) -> list[SegmentDefinitionRecord]:
        sample_seed = _promotion_sample_seed(promotion)
        user_vectors = self._load_user_vectors(promotion, sample_seed)
        if len(user_vectors) < self._min_cluster_size:
            return []

        cluster_count = min(
            self._max_suggested_segments,
            max(1, len(user_vectors) // self._min_cluster_size),
        )
        clusters = _cluster_user_vectors(user_vectors, cluster_count)
        if not clusters:
            return []

        total_eligible_user_count = len(user_vectors)
        return [
            _segment_definition_from_cluster(
                promotion=promotion,
                cluster=cluster,
                rank=rank,
                total_eligible_user_count=total_eligible_user_count,
                sample_seed=sample_seed,
                vector_version=self._vector_version,
            )
            for rank, cluster in enumerate(
                sorted(
                    clusters,
                    key=lambda cluster: (
                        -cluster.score,
                        -len(cluster.users),
                        cluster.index,
                    ),
                )
            )
            if len(cluster.users) >= self._min_cluster_size
        ][: self._max_suggested_segments]

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
    cluster: _Cluster,
    rank: int,
    total_eligible_user_count: int,
    sample_seed: str,
    vector_version: str,
) -> SegmentDefinitionRecord:
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
    natural_language_query = (
        "Users grouped by similar hotel behavior vectors for this promotion."
    )
    if top_common_features:
        natural_language_query = (
            f"{natural_language_query} Strongest signals: "
            f"{', '.join(top_common_features)}."
        )
    return SegmentDefinitionRecord(
        segment_id=segment_id,
        project_id=promotion.project_id,
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        segment_name=_segment_name_from_cluster(cluster=cluster, rank=rank),
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
            "top_common_features": top_common_features,
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


def _segment_name_from_cluster(*, cluster: _Cluster, rank: int) -> str:
    labels = _top_feature_labels(cluster.centroid, limit=2)
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
