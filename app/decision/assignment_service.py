from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from app.decision.matcher import (
    ANN_CANDIDATE_LIMIT,
    FALLBACK_REASON_BELOW_THRESHOLD,
    FALLBACK_REASON_INVALID_USER_VECTOR,
    FALLBACK_REASON_NO_CANDIDATE,
    FALLBACK_SEGMENT_ID,
    MatchResult,
    SegmentCandidateReranker,
    SegmentMatchValidationError,
    SegmentVector,
    UserVector,
    invalid_user_vector_result,
    parse_vector_values,
)
from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWriter,
    PromotionRunWriter,
    PromotionTargetSegmentReader,
    SegmentVectorReader,
    SegmentVectorRecord,
    UserBehaviorVectorReader,
    UserSegmentAssignmentWrite,
    UserSegmentAssignmentWriter,
)
from app.decision.schemas import (
    AdExperimentStatus,
    AssignmentSource,
    SegmentAssignmentBuildRequest,
    SegmentAssignmentBuildResponse,
)


INSUFFICIENT_REASON_STATUS = AdExperimentStatus.INSUFFICIENT_DATA.value
MATCHING_MODE = "pgvector_hnsw_rerank"


class SegmentAssignmentRunNotFoundError(Exception):
    pass


class SegmentAssignmentValidationError(Exception):
    pass


@dataclass(frozen=True)
class _ExperimentSet:
    non_fallback: list[AdExperimentRecord]
    fallback: AdExperimentRecord | None


@dataclass(frozen=True)
class _BuildMatchResult:
    matches: Mapping[str, MatchResult]
    ann_candidate_count: int
    exact_reranked_pair_count: int
    ann_underfilled_user_count: int

    @property
    def fallback_count(self) -> int:
        return sum(1 for result in self.matches.values() if result.fallback)

    @property
    def below_threshold_fallback_count(self) -> int:
        return _fallback_reason_count(self.matches, FALLBACK_REASON_BELOW_THRESHOLD)

    @property
    def no_candidate_fallback_count(self) -> int:
        return _fallback_reason_count(self.matches, FALLBACK_REASON_NO_CANDIDATE)

    @property
    def invalid_user_vector_fallback_count(self) -> int:
        return _fallback_reason_count(
            self.matches,
            FALLBACK_REASON_INVALID_USER_VECTOR,
        )


class SegmentAssignmentService:
    def __init__(
        self,
        *,
        promotion_run_repository: PromotionRunWriter,
        ad_experiment_repository: AdExperimentWriter,
        promotion_target_segment_repository: PromotionTargetSegmentReader,
        segment_vector_repository: SegmentVectorReader,
        user_behavior_vector_repository: UserBehaviorVectorReader,
        user_segment_assignment_repository: UserSegmentAssignmentWriter,
        reranker: SegmentCandidateReranker,
    ) -> None:
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._promotion_target_segment_repository = promotion_target_segment_repository
        self._segment_vector_repository = segment_vector_repository
        self._user_behavior_vector_repository = user_behavior_vector_repository
        self._user_segment_assignment_repository = user_segment_assignment_repository
        self._reranker = reranker

    def build_assignments(
        self,
        *,
        promotion_run_id: str,
        request: SegmentAssignmentBuildRequest,
    ) -> SegmentAssignmentBuildResponse:
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            raise SegmentAssignmentRunNotFoundError(
                f"promotion run not found: {promotion_run_id}"
            )

        min_sample_size = _extract_min_sample_size(run.goal_snapshot_json)
        experiments = _split_experiments(
            self._ad_experiment_repository.list_by_run(promotion_run_id)
        )
        if not experiments.non_fallback:
            raise SegmentAssignmentValidationError(
                "at least one non-fallback ad experiment is required"
            )

        segment_vectors = self._load_segment_vectors(
            project_id=run.project_id,
            promotion_id=run.promotion_id,
            analysis_id=run.analysis_id,
            segment_ids=[
                experiment.segment_id for experiment in experiments.non_fallback
            ],
            vector_version=request.vector_version,
        )
        eligible_users = self._load_eligible_users(
            project_id=run.project_id,
            request=request,
        )
        existing_user_ids = (
            self._user_segment_assignment_repository.list_existing_user_ids(
                promotion_run_id=run.promotion_run_id,
                user_ids=[user.user_id for user in eligible_users],
            )
        )
        users_to_match = [
            user for user in eligible_users if user.user_id not in existing_user_ids
        ]

        try:
            build_result = self._build_match_results(
                project_id=run.project_id,
                promotion_id=run.promotion_id,
                analysis_id=run.analysis_id,
                vector_version=request.vector_version,
                users=users_to_match,
                segment_vectors=segment_vectors,
            )
        except SegmentMatchValidationError as exc:
            raise SegmentAssignmentValidationError(str(exc)) from exc

        fallback_needed = any(
            result.fallback for result in build_result.matches.values()
        )
        if fallback_needed and experiments.fallback is None:
            raise SegmentAssignmentValidationError(
                "fallback ad experiment is required when fallback assignments exist"
            )

        assignments = _build_assignment_writes(
            project_id=run.project_id,
            promotion_run_id=run.promotion_run_id,
            matches=build_result.matches,
            experiments=experiments,
            assigned_at=datetime.now(UTC),
            expires_in_days=request.expires_in_days,
        )
        inserted_count = self._user_segment_assignment_repository.insert_many(
            assignments
        )

        segment_counts = (
            self._user_segment_assignment_repository.count_by_run_segments(
                promotion_run_id=run.promotion_run_id,
                segment_ids=[
                    experiment.segment_id for experiment in experiments.non_fallback
                ],
            )
        )
        insufficient_count = self._mark_insufficient_segments(
            analysis_id=run.analysis_id,
            experiments=experiments.non_fallback,
            segment_counts=segment_counts,
            min_sample_size=min_sample_size,
        )

        return SegmentAssignmentBuildResponse(
            promotion_run_id=run.promotion_run_id,
            matching_mode=MATCHING_MODE,
            vector_version=request.vector_version,
            ann_candidate_limit=ANN_CANDIDATE_LIMIT,
            ann_candidate_count=build_result.ann_candidate_count,
            exact_reranked_pair_count=build_result.exact_reranked_pair_count,
            assignment_count=inserted_count,
            batch_has_fallback=fallback_needed,
            fallback_count=build_result.fallback_count,
            below_threshold_fallback_count=(
                build_result.below_threshold_fallback_count
            ),
            no_candidate_fallback_count=build_result.no_candidate_fallback_count,
            invalid_user_vector_fallback_count=(
                build_result.invalid_user_vector_fallback_count
            ),
            ann_underfilled_user_count=build_result.ann_underfilled_user_count,
            skipped_existing_count=len(existing_user_ids),
            insufficient_segment_count=insufficient_count,
            status="completed",
        )

    def _build_match_results(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        vector_version: str,
        users: Sequence[UserVector],
        segment_vectors: Sequence[SegmentVector],
    ) -> _BuildMatchResult:
        if not segment_vectors:
            raise SegmentMatchValidationError(
                "at least one non-fallback segment vector is required"
            )

        segment_vector_ids = [segment.segment_vector_id for segment in segment_vectors]
        expected_candidate_count = min(len(segment_vector_ids), ANN_CANDIDATE_LIMIT)
        matches: dict[str, MatchResult] = {}
        ann_candidate_count = 0
        exact_reranked_pair_count = 0
        ann_underfilled_user_count = 0

        self._segment_vector_repository.configure_ann_search()
        for user in users:
            normalized_user_vector = self._reranker.normalize_user_vector(user)
            if normalized_user_vector is None:
                matches[user.user_id] = invalid_user_vector_result()
                continue

            candidate_records = self._segment_vector_repository.list_ann_candidates(
                project_id=project_id,
                promotion_id=promotion_id,
                analysis_id=analysis_id,
                segment_vector_ids=segment_vector_ids,
                vector_version=vector_version,
                query_vector=normalized_user_vector,
                limit=ANN_CANDIDATE_LIMIT,
            )
            candidate_count = len(candidate_records)
            ann_candidate_count += candidate_count
            exact_reranked_pair_count += candidate_count
            if candidate_count < expected_candidate_count:
                ann_underfilled_user_count += 1

            matches[user.user_id] = self._reranker.rerank(
                normalized_user_vector=normalized_user_vector,
                candidates=[
                    _segment_vector_from_record(record, require_embedding=True)
                    for record in candidate_records
                ],
            )

        return _BuildMatchResult(
            matches=matches,
            ann_candidate_count=ann_candidate_count,
            exact_reranked_pair_count=exact_reranked_pair_count,
            ann_underfilled_user_count=ann_underfilled_user_count,
        )

    def _load_segment_vectors(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        segment_ids: Sequence[str],
        vector_version: str,
    ) -> list[SegmentVector]:
        records = self._segment_vector_repository.list_for_run_segments(
            project_id=project_id,
            promotion_id=promotion_id,
            analysis_id=analysis_id,
            segment_ids=segment_ids,
            vector_version=vector_version,
        )
        records_by_segment: dict[str, list[SegmentVectorRecord]] = defaultdict(list)
        for record in records:
            records_by_segment[record.segment_id].append(record)

        segment_vectors: list[SegmentVector] = []
        for segment_id in segment_ids:
            segment_records = records_by_segment.get(segment_id, [])
            if len(segment_records) != 1:
                raise SegmentAssignmentValidationError(
                    "each non-fallback segment must have exactly one segment vector: "
                    f"{segment_id}"
                )
            segment_vectors.append(
                _segment_vector_from_record(
                    segment_records[0],
                    require_embedding=True,
                )
            )
        return segment_vectors

    def _load_eligible_users(
        self,
        *,
        project_id: str,
        request: SegmentAssignmentBuildRequest,
    ) -> list[UserVector]:
        if request.user_ids:
            records = self._user_behavior_vector_repository.list_by_user_ids(
                project_id=project_id,
                user_ids=request.user_ids,
                vector_version=request.vector_version,
            )
        else:
            if request.eligible_user_limit is None:
                raise SegmentAssignmentValidationError(
                    "eligible_user_limit is required when user_ids is omitted"
                )
            records = self._user_behavior_vector_repository.list_for_project(
                project_id=project_id,
                vector_version=request.vector_version,
                limit=request.eligible_user_limit,
            )
        return [
            UserVector(
                user_id=record.user_id,
                vector_dim=int(record.vector_dim),
                vector_values=record.vector_values,
            )
            for record in records
        ]

    def _mark_insufficient_segments(
        self,
        *,
        analysis_id: str,
        experiments: Sequence[AdExperimentRecord],
        segment_counts: Mapping[str, int],
        min_sample_size: int,
    ) -> int:
        insufficient_count = 0
        for experiment in experiments:
            assigned_user_count = segment_counts.get(experiment.segment_id, 0)
            if assigned_user_count >= min_sample_size:
                continue

            insufficient_count += 1
            self._ad_experiment_repository.update_status(
                ad_experiment_id=experiment.ad_experiment_id,
                status=INSUFFICIENT_REASON_STATUS,
            )
            self._promotion_target_segment_repository.update_status(
                analysis_id=analysis_id,
                segment_id=experiment.segment_id,
                status=INSUFFICIENT_REASON_STATUS,
            )
        return insufficient_count


def _split_experiments(experiments: Sequence[AdExperimentRecord]) -> _ExperimentSet:
    if not experiments:
        raise SegmentAssignmentValidationError("ad experiments are required")

    fallback: AdExperimentRecord | None = None
    non_fallback: list[AdExperimentRecord] = []
    for experiment in experiments:
        if experiment.segment_id == FALLBACK_SEGMENT_ID:
            fallback = experiment
        else:
            non_fallback.append(experiment)
    return _ExperimentSet(non_fallback=non_fallback, fallback=fallback)


def _build_assignment_writes(
    *,
    project_id: str,
    promotion_run_id: str,
    matches: Mapping[str, MatchResult],
    experiments: _ExperimentSet,
    assigned_at: datetime,
    expires_in_days: int | None,
) -> list[UserSegmentAssignmentWrite]:
    experiments_by_segment = {
        experiment.segment_id: experiment for experiment in experiments.non_fallback
    }
    if experiments.fallback is not None:
        experiments_by_segment[experiments.fallback.segment_id] = experiments.fallback

    expires_at = (
        assigned_at + timedelta(days=expires_in_days)
        if expires_in_days is not None
        else None
    )
    assignments: list[UserSegmentAssignmentWrite] = []
    for user_id, result in matches.items():
        experiment = experiments_by_segment[result.segment_id]
        assignments.append(
            UserSegmentAssignmentWrite(
                project_id=project_id,
                promotion_run_id=promotion_run_id,
                user_id=user_id,
                segment_id=experiment.segment_id,
                ad_experiment_id=experiment.ad_experiment_id,
                content_id=experiment.content_id,
                content_option_id=experiment.content_option_id,
                similarity_score=_score_to_decimal(result.similarity_score),
                fallback=result.fallback,
                fallback_reason=result.fallback_reason,
                assignment_source=(
                    AssignmentSource.FALLBACK.value
                    if result.fallback
                    else AssignmentSource.DECISION_BATCH.value
                ),
                assigned_at=assigned_at,
                expires_at=expires_at,
            )
        )
    return assignments


def _extract_min_sample_size(snapshot: Mapping[str, Any]) -> int:
    if "min_sample_size" not in snapshot:
        raise SegmentAssignmentValidationError(
            "goal_snapshot_json.min_sample_size is required"
        )
    value = snapshot["min_sample_size"]
    if isinstance(value, bool):
        raise SegmentAssignmentValidationError(
            "goal_snapshot_json.min_sample_size must be an integer"
        )
    try:
        min_sample_size = int(value)
    except (TypeError, ValueError) as exc:
        raise SegmentAssignmentValidationError(
            "goal_snapshot_json.min_sample_size must be an integer"
        ) from exc
    if min_sample_size < 0:
        raise SegmentAssignmentValidationError(
            "goal_snapshot_json.min_sample_size must not be negative"
        )
    return min_sample_size


def _segment_vector_from_record(
    record: SegmentVectorRecord,
    *,
    require_embedding: bool,
) -> SegmentVector:
    if record.embedding is None:
        if require_embedding:
            raise SegmentAssignmentValidationError(
                f"segment embedding is required: {record.segment_id}"
            )
        embedding_values = parse_vector_values(record.vector_values)
    else:
        embedding_values = parse_vector_values(record.embedding)
    return SegmentVector(
        segment_vector_id=record.segment_vector_id,
        segment_id=record.segment_id,
        vector_dim=int(record.vector_dim),
        embedding_values=embedding_values,
    )


def _fallback_reason_count(
    matches: Mapping[str, MatchResult],
    fallback_reason: str,
) -> int:
    return sum(
        1
        for result in matches.values()
        if result.fallback and result.fallback_reason == fallback_reason
    )


def _score_to_decimal(score: float | None) -> Decimal | None:
    if score is None:
        return None
    clamped_score = min(1.0, max(0.0, float(score)))
    return Decimal(str(clamped_score)).quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_UP,
    )
