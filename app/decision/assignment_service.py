from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterator, Mapping, Sequence

from app.decision.matcher import (
    ANN_CANDIDATE_LIMIT,
    ANN_QUERY_USER_BATCH_SIZE,
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
    UserBehaviorVectorRecord,
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
from app.logging import log, log_context_scope, now_ms, duration_ms


INSUFFICIENT_REASON_STATUS = AdExperimentStatus.INSUFFICIENT_DATA.value
MATCHING_MODE = "pgvector_hnsw_rerank"
ASSIGNMENT_PAGE_SIZE = 10_000
DEFAULT_VECTOR_VERSION = "v1"
AUDIENCE_SCOPE_BASE = "user_behavior_vectors"


class SegmentAssignmentRunNotFoundError(Exception):
    pass


class SegmentAssignmentValidationError(Exception):
    pass


@dataclass(frozen=True)
class _ExperimentSet:
    non_fallback: list[AdExperimentRecord]
    fallback: AdExperimentRecord | None


@dataclass(frozen=True)
class _EffectiveAudienceScope:
    effective_vector_version: str
    effective_limit: int | None
    source: str | None
    user_ids: list[str] | None


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


@dataclass
class _BuildDiagnostics:
    ann_candidate_count: int = 0
    exact_reranked_pair_count: int = 0
    assignment_count: int = 0
    batch_has_fallback: bool = False
    fallback_count: int = 0
    below_threshold_fallback_count: int = 0
    no_candidate_fallback_count: int = 0
    invalid_user_vector_fallback_count: int = 0
    ann_underfilled_user_count: int = 0
    skipped_existing_count: int = 0

    def accumulate_matches(self, result: _BuildMatchResult) -> None:
        self.ann_candidate_count += result.ann_candidate_count
        self.exact_reranked_pair_count += result.exact_reranked_pair_count
        self.batch_has_fallback = (
            self.batch_has_fallback or result.fallback_count > 0
        )
        self.fallback_count += result.fallback_count
        self.below_threshold_fallback_count += result.below_threshold_fallback_count
        self.no_candidate_fallback_count += result.no_candidate_fallback_count
        self.invalid_user_vector_fallback_count += (
            result.invalid_user_vector_fallback_count
        )
        self.ann_underfilled_user_count += result.ann_underfilled_user_count


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

    @log_context_scope
    def build_assignments(
        self,
        *,
        promotion_run_id: str,
        request: SegmentAssignmentBuildRequest,
    ) -> SegmentAssignmentBuildResponse:
        started_at = now_ms()
        log.assign_context({"promotionRunId": promotion_run_id})
        log.info("started", {"promotionRunId": promotion_run_id, "request": request})
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            log.warn("promotion_run_not_found", {"promotionRunId": promotion_run_id})
            raise SegmentAssignmentRunNotFoundError(
                f"promotion run not found: {promotion_run_id}"
            )
        log.assign_context(
            {
                "projectId": run.project_id,
                "campaignId": run.campaign_id,
                "promotionId": run.promotion_id,
                "analysisId": run.analysis_id,
                "generationId": run.generation_id,
            }
        )
        log.info("promotion_run_loaded", {"promotionRun": run})

        min_sample_size = _extract_min_sample_size(run.goal_snapshot_json)
        audience_scope = _build_effective_audience_scope(
            goal_snapshot_json=run.goal_snapshot_json,
            request=request,
            project_id=run.project_id,
        )
        experiments = _split_experiments(
            self._ad_experiment_repository.list_by_run(promotion_run_id)
        )
        if not experiments.non_fallback:
            log.warn("ad_experiments_empty", {"promotionRunId": promotion_run_id})
            raise SegmentAssignmentValidationError(
                "at least one non-fallback ad experiment is required"
            )
        log.info("ad_experiments_loaded", {"nonFallbackCount": len(experiments.non_fallback), "hasFallback": experiments.fallback is not None})

        segment_vectors = self._load_segment_vectors(
            project_id=run.project_id,
            promotion_id=run.promotion_id,
            analysis_id=run.analysis_id,
            segment_ids=[
                experiment.segment_id for experiment in experiments.non_fallback
            ],
            vector_version=audience_scope.effective_vector_version,
        )
        log.info("segment_vectors_loaded", {"segmentVectorCount": len(segment_vectors)})
        diagnostics = _BuildDiagnostics()
        assigned_at = datetime.now(UTC)
        for page_number, eligible_users in enumerate(
            self._iter_eligible_user_pages(
                project_id=run.project_id,
                audience_scope=audience_scope,
            ),
            start=1,
        ):
            log.info(
                "eligible_users_page_loaded",
                {
                    "pageNumber": page_number,
                    "eligibleUserCount": len(eligible_users),
                },
            )
            existing_user_ids = (
                self._user_segment_assignment_repository.list_existing_user_ids(
                    promotion_run_id=run.promotion_run_id,
                    user_ids=[user.user_id for user in eligible_users],
                )
            )
            diagnostics.skipped_existing_count += len(existing_user_ids)
            users_to_match = [
                user
                for user in eligible_users
                if user.user_id not in existing_user_ids
            ]
            if existing_user_ids:
                log.info(
                    "existing_assignments_skipped",
                    {
                        "pageNumber": page_number,
                        "userCount": len(existing_user_ids),
                    },
                )

            try:
                build_result = self._build_match_results(
                    project_id=run.project_id,
                    promotion_id=run.promotion_id,
                    analysis_id=run.analysis_id,
                    vector_version=audience_scope.effective_vector_version,
                    users=users_to_match,
                    segment_vectors=segment_vectors,
                )
            except (SegmentMatchValidationError, ValueError) as exc:
                log.warn("segment_matching_invalid", {"err": exc})
                raise SegmentAssignmentValidationError(str(exc)) from exc

            fallback_needed = build_result.fallback_count > 0
            if fallback_needed and experiments.fallback is None:
                log.warn(
                    "fallback_ad_experiment_missing",
                    {
                        "pageNumber": page_number,
                        "fallbackCount": build_result.fallback_count,
                    },
                )
                raise SegmentAssignmentValidationError(
                    "fallback ad experiment is required when fallback assignments exist"
                )
            log.info(
                "segment_matches_created",
                {
                    "pageNumber": page_number,
                    "matchCount": len(build_result.matches),
                    "fallbackCount": build_result.fallback_count,
                    "annCandidateCount": build_result.ann_candidate_count,
                    "exactRerankedPairCount": build_result.exact_reranked_pair_count,
                },
            )

            assignments = _build_assignment_writes(
                project_id=run.project_id,
                promotion_run_id=run.promotion_run_id,
                matches=build_result.matches,
                experiments=experiments,
                assigned_at=assigned_at,
                expires_in_days=request.expires_in_days,
            )
            inserted_count = self._user_segment_assignment_repository.insert_many(
                assignments
            )
            diagnostics.assignment_count += inserted_count
            diagnostics.accumulate_matches(build_result)
            log.info(
                "segment_assignments_page_created",
                {
                    "pageNumber": page_number,
                    "assignmentCount": inserted_count,
                },
            )

        # #193 keeps the assignment-count readiness behavior for response
        # compatibility. #194 will move readiness to evaluation event samples.
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
        if insufficient_count:
            log.info("segments_marked_insufficient", {"insufficientSegmentCount": insufficient_count})

        response = SegmentAssignmentBuildResponse(
            promotion_run_id=run.promotion_run_id,
            matching_mode=MATCHING_MODE,
            vector_version=audience_scope.effective_vector_version,
            ann_candidate_limit=ANN_CANDIDATE_LIMIT,
            ann_candidate_count=diagnostics.ann_candidate_count,
            exact_reranked_pair_count=diagnostics.exact_reranked_pair_count,
            assignment_count=diagnostics.assignment_count,
            batch_has_fallback=diagnostics.batch_has_fallback,
            fallback_count=diagnostics.fallback_count,
            below_threshold_fallback_count=(
                diagnostics.below_threshold_fallback_count
            ),
            no_candidate_fallback_count=diagnostics.no_candidate_fallback_count,
            invalid_user_vector_fallback_count=(
                diagnostics.invalid_user_vector_fallback_count
            ),
            ann_underfilled_user_count=diagnostics.ann_underfilled_user_count,
            skipped_existing_count=diagnostics.skipped_existing_count,
            insufficient_segment_count=insufficient_count,
            status="completed",
        )
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response

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
            log.warn("segment_vectors_empty", {"analysisId": analysis_id})
            raise SegmentMatchValidationError(
                "at least one non-fallback segment vector is required"
            )

        segment_vector_ids = [segment.segment_vector_id for segment in segment_vectors]
        expected_candidate_count = min(len(segment_vector_ids), ANN_CANDIDATE_LIMIT)
        matches: dict[str, MatchResult] = {}
        ann_candidate_count = 0
        exact_reranked_pair_count = 0
        ann_underfilled_user_count = 0

        normalized_users: list[tuple[str, list[float]]] = []
        for user in _deduplicate_users_by_id(users):
            normalized_user_vector = self._reranker.normalize_user_vector(user)
            if normalized_user_vector is None:
                matches[user.user_id] = invalid_user_vector_result()
                continue
            normalized_users.append((user.user_id, normalized_user_vector))

        if normalized_users:
            self._segment_vector_repository.configure_ann_search()

        for user_chunk in _chunks(normalized_users, ANN_QUERY_USER_BATCH_SIZE):
            user_ids = [user_id for user_id, _vector in user_chunk]
            query_vectors = [vector for _user_id, vector in user_chunk]
            candidates_by_user = (
                self._segment_vector_repository.list_ann_candidates_for_users(
                    project_id=project_id,
                    promotion_id=promotion_id,
                    analysis_id=analysis_id,
                    segment_vector_ids=segment_vector_ids,
                    vector_version=vector_version,
                    user_ids=user_ids,
                    query_vectors=query_vectors,
                    limit=ANN_CANDIDATE_LIMIT,
                )
            )
            for user_id, normalized_user_vector in user_chunk:
                candidate_records = candidates_by_user.get(user_id, [])
                candidate_count = len(candidate_records)
                ann_candidate_count += candidate_count
                exact_reranked_pair_count += candidate_count
                if candidate_count < expected_candidate_count:
                    ann_underfilled_user_count += 1

                matches[user_id] = self._reranker.rerank(
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
                log.warn("segment_vector_invalid", {"segmentId": segment_id, "segmentVectorCount": len(segment_records)})
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

    def _iter_eligible_user_pages(
        self,
        *,
        project_id: str,
        audience_scope: _EffectiveAudienceScope,
    ) -> Iterator[list[UserVector]]:
        if audience_scope.user_ids is not None:
            user_ids = sorted(set(audience_scope.user_ids))
            if audience_scope.effective_limit is not None:
                user_ids = user_ids[: audience_scope.effective_limit]

            previous_user_id: str | None = None
            for index in range(0, len(user_ids), ASSIGNMENT_PAGE_SIZE):
                user_id_page = user_ids[index : index + ASSIGNMENT_PAGE_SIZE]
                records = self._user_behavior_vector_repository.list_by_user_ids(
                    project_id=project_id,
                    user_ids=user_id_page,
                    vector_version=audience_scope.effective_vector_version,
                    source=audience_scope.source,
                )
                _validate_user_vector_page(
                    records,
                    after_user_id=previous_user_id,
                )
                if records:
                    previous_user_id = records[-1].user_id
                yield _user_vectors_from_records(records)
            return

        after_user_id: str | None = None
        remaining_limit = audience_scope.effective_limit
        while remaining_limit is None or remaining_limit > 0:
            page_size = (
                ASSIGNMENT_PAGE_SIZE
                if remaining_limit is None
                else min(ASSIGNMENT_PAGE_SIZE, remaining_limit)
            )
            records = self._user_behavior_vector_repository.list_for_project(
                project_id=project_id,
                vector_version=audience_scope.effective_vector_version,
                limit=page_size,
                source=audience_scope.source,
                after_user_id=after_user_id,
            )
            if not records:
                return
            if len(records) > page_size:
                raise SegmentAssignmentValidationError(
                    "eligible user page exceeded the requested page size"
                )
            _validate_user_vector_page(records, after_user_id=after_user_id)

            yield _user_vectors_from_records(records)

            page_count = len(records)
            after_user_id = records[-1].user_id
            if remaining_limit is not None:
                remaining_limit -= page_count
                if remaining_limit <= 0:
                    return
            if page_count < page_size:
                return

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
        log.warn("ad_experiments_empty")
        raise SegmentAssignmentValidationError("ad experiments are required")

    fallback: AdExperimentRecord | None = None
    non_fallback: list[AdExperimentRecord] = []
    for experiment in experiments:
        if experiment.segment_id == FALLBACK_SEGMENT_ID:
            fallback = experiment
        else:
            non_fallback.append(experiment)
    return _ExperimentSet(non_fallback=non_fallback, fallback=fallback)


def _deduplicate_users_by_id(users: Sequence[UserVector]) -> list[UserVector]:
    seen_user_ids: set[str] = set()
    deduplicated: list[UserVector] = []
    for user in users:
        if user.user_id in seen_user_ids:
            continue
        seen_user_ids.add(user.user_id)
        deduplicated.append(user)
    return deduplicated


def _validate_user_vector_page(
    records: Sequence[UserBehaviorVectorRecord],
    *,
    after_user_id: str | None,
) -> None:
    user_ids = [record.user_id for record in records]
    if len(user_ids) != len(set(user_ids)):
        raise SegmentAssignmentValidationError(
            "eligible user page contains duplicate user_id values"
        )
    if user_ids != sorted(user_ids):
        raise SegmentAssignmentValidationError(
            "eligible user page must be ordered by user_id ascending"
        )
    if after_user_id is not None and any(
        user_id <= after_user_id for user_id in user_ids
    ):
        raise SegmentAssignmentValidationError(
            "eligible user cursor must increase monotonically"
        )


def _user_vectors_from_records(
    records: Sequence[UserBehaviorVectorRecord],
) -> list[UserVector]:
    return [
        UserVector(
            user_id=record.user_id,
            vector_dim=int(record.vector_dim),
            vector_values=record.vector_values,
        )
        for record in records
    ]


def _build_effective_audience_scope(
    *,
    goal_snapshot_json: Mapping[str, Any],
    request: SegmentAssignmentBuildRequest,
    project_id: str,
) -> _EffectiveAudienceScope:
    del project_id  # promotion_run.project_id is the only allowed project context.
    raw_scope = goal_snapshot_json.get("audience_scope")
    if raw_scope is not None and not isinstance(raw_scope, Mapping):
        raise SegmentAssignmentValidationError("audience_scope must be an object")

    request_vector_version_is_explicit = "vector_version" in request.model_fields_set
    request_user_ids = list(request.user_ids) if request.user_ids else None
    request_limit = request.eligible_user_limit

    if raw_scope is None:
        return _EffectiveAudienceScope(
            effective_vector_version=request.vector_version or DEFAULT_VECTOR_VERSION,
            effective_limit=request_limit,
            source=None,
            user_ids=request_user_ids,
        )

    if request_user_ids:
        raise SegmentAssignmentValidationError(
            "audience_scope and user_ids cannot be combined in MVP"
        )

    base = raw_scope.get("base", AUDIENCE_SCOPE_BASE)
    if base != AUDIENCE_SCOPE_BASE:
        raise SegmentAssignmentValidationError(
            "audience_scope.base must be user_behavior_vectors"
        )

    scope_vector_version = raw_scope.get("vector_version")
    if scope_vector_version is not None:
        if not isinstance(scope_vector_version, str) or not scope_vector_version:
            raise SegmentAssignmentValidationError(
                "audience_scope.vector_version must be a non-empty string"
            )
        if (
            request_vector_version_is_explicit
            and request.vector_version != scope_vector_version
        ):
            raise SegmentAssignmentValidationError(
                "request.vector_version must match audience_scope.vector_version"
            )
        effective_vector_version = scope_vector_version
    else:
        effective_vector_version = request.vector_version or DEFAULT_VECTOR_VERSION

    filters = raw_scope.get("filters", {})
    source = _parse_audience_filters(filters)
    selection_policy = raw_scope.get("selection_policy", {})
    scope_limit = _parse_selection_policy_limit(selection_policy)
    _validate_selection_policy_ordering(selection_policy)

    if scope_limit is not None and request_limit is not None:
        effective_limit = min(scope_limit, request_limit)
    elif scope_limit is not None:
        effective_limit = scope_limit
    else:
        effective_limit = request_limit

    return _EffectiveAudienceScope(
        effective_vector_version=effective_vector_version,
        effective_limit=effective_limit,
        source=source,
        user_ids=None,
    )


def _parse_audience_filters(raw_filters: Any) -> str | None:
    if raw_filters is None:
        return None
    if not isinstance(raw_filters, Mapping):
        raise SegmentAssignmentValidationError(
            "audience_scope.filters must be an object"
        )
    if "project_id" in raw_filters:
        raise SegmentAssignmentValidationError(
            "audience_scope.filters.project_id is not allowed"
        )

    allowed_keys = {"has_valid_vector", "source"}
    unknown_keys = sorted(set(raw_filters) - allowed_keys)
    if unknown_keys:
        raise SegmentAssignmentValidationError(
            "unsupported audience_scope.filters keys: " + ", ".join(unknown_keys)
        )

    if "has_valid_vector" in raw_filters and raw_filters["has_valid_vector"] is not True:
        raise SegmentAssignmentValidationError(
            "audience_scope.filters.has_valid_vector must be true"
        )

    source = raw_filters.get("source")
    if source is None:
        return None
    if not isinstance(source, str) or not source.strip():
        raise SegmentAssignmentValidationError(
            "audience_scope.filters.source must be a non-empty string"
        )
    return source.strip()


def _parse_selection_policy_limit(raw_selection_policy: Any) -> int | None:
    selection_policy = _selection_policy_mapping(raw_selection_policy)
    raw_limit = selection_policy.get("limit")
    if raw_limit is None:
        return None
    if isinstance(raw_limit, bool):
        raise SegmentAssignmentValidationError(
            "audience_scope.selection_policy.limit must be a positive integer"
        )
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise SegmentAssignmentValidationError(
            "audience_scope.selection_policy.limit must be a positive integer"
        ) from exc
    if limit < 1:
        raise SegmentAssignmentValidationError(
            "audience_scope.selection_policy.limit must be a positive integer"
        )
    return limit


def _validate_selection_policy_ordering(raw_selection_policy: Any) -> None:
    selection_policy = _selection_policy_mapping(raw_selection_policy)
    unknown_keys = sorted(set(selection_policy) - {"limit", "ordering", "mode"})
    if unknown_keys:
        raise SegmentAssignmentValidationError(
            "unsupported audience_scope.selection_policy keys: "
            + ", ".join(unknown_keys)
        )
    mode = selection_policy.get("mode")
    if mode is not None and mode != "batch":
        raise SegmentAssignmentValidationError(
            "audience_scope.selection_policy.mode must be batch"
        )
    ordering = selection_policy.get("ordering")
    if ordering is not None and ordering != "user_id_asc":
        raise SegmentAssignmentValidationError(
            "audience_scope.selection_policy.ordering must be user_id_asc"
        )


def _selection_policy_mapping(raw_selection_policy: Any) -> Mapping[str, Any]:
    if raw_selection_policy is None:
        return {}
    if not isinstance(raw_selection_policy, Mapping):
        raise SegmentAssignmentValidationError(
            "audience_scope.selection_policy must be an object"
        )
    return raw_selection_policy


def _chunks(
    items: Sequence[tuple[str, list[float]]],
    size: int,
) -> list[Sequence[tuple[str, list[float]]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


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
    if record.source == "fixture":
        raise SegmentAssignmentValidationError(
            f"fixture segment vector is not allowed: {record.segment_id}"
        )
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
