from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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
    normalize_values,
    parse_vector_values,
)
from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentReader,
    PromotionRunWriter,
    PromotionRunRecord,
    SegmentVectorReader,
    SegmentVectorRecord,
    UserBehaviorVectorRecord,
    UserBehaviorVectorReader,
    UserSegmentAssignmentInsertRecord,
    UserSegmentAssignmentRunAggregateRecord,
    UserSegmentAssignmentWrite,
    UserSegmentAssignmentWriter,
)
from app.decision.schemas import (
    AssignmentSource,
    SegmentAssignmentBuildRequest,
    SegmentAssignmentBuildResponse,
)
from app.logging import log, log_context_scope, now_ms, duration_ms


MATCHING_MODE = "pgvector_hnsw_rerank"
ASSIGNMENT_PAGE_SIZE = 10_000
DEFAULT_VECTOR_VERSION = "v1"
AUDIENCE_SCOPE_BASE = "user_behavior_vectors"
ASSIGNMENT_MODE_LIVE_KEYSET = "live_keyset"
ASSIGNMENT_MODE_EXPLICIT_USER_IDS = "explicit_user_ids"
ANN_NOT_APPLIED_NO_USERS = "no_users_to_match"
ANN_NOT_APPLIED_NO_VALID_VECTORS = "no_valid_user_vectors"
FALLBACK_REASON_KEYS = (
    FALLBACK_REASON_BELOW_THRESHOLD,
    FALLBACK_REASON_NO_CANDIDATE,
    FALLBACK_REASON_INVALID_USER_VECTOR,
)
SIMILARITY_BUCKET_KEYS = (
    "not_available",
    "0_00_to_0_50",
    "0_50_to_0_65",
    "0_65_to_0_80",
    "0_80_to_0_90",
    "gte_0_90",
)


class SegmentAssignmentRunNotFoundError(Exception):
    pass


class SegmentAssignmentValidationError(Exception):
    pass


@dataclass(frozen=True)
class AssignmentExperimentSet:
    non_fallback: list[AdExperimentRecord]
    fallback: AdExperimentRecord | None


@dataclass(frozen=True)
class AssignmentAudienceScope:
    effective_vector_version: str
    effective_limit: int | None
    source: str | None
    user_ids: list[str] | None


@dataclass(frozen=True)
class AssignmentBuildInput:
    run: PromotionRunRecord
    audience_scope: AssignmentAudienceScope
    experiments: AssignmentExperimentSet
    segment_vectors: tuple[SegmentVector, ...]


@dataclass(frozen=True)
class AssignmentPageSelection:
    eligible_users: tuple[UserVector, ...]
    users_to_match: tuple[UserVector, ...]
    skipped_existing_count: int


@dataclass(frozen=True)
class AssignmentPageMatchResult:
    matches: Mapping[str, MatchResult]
    ann_candidate_count: int
    exact_reranked_pair_count: int
    ann_underfilled_user_count: int
    exact_rescue_user_count: int
    ann_query_user_count: int

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


@dataclass(frozen=True)
class AssignmentPageWriteOutcome:
    attempted_count: int
    inserted_records: tuple[UserSegmentAssignmentInsertRecord, ...]


@dataclass
class AssignmentDiagnostics:
    page_count: int = 0
    processed_user_count: int = 0
    users_to_match_count: int = 0
    ann_candidate_count: int = 0
    exact_reranked_pair_count: int = 0
    ann_underfilled_user_count: int = 0
    exact_rescue_user_count: int = 0
    ann_applied: bool = False
    assignment_count: int = 0
    insert_conflict_count: int = 0
    segment_assignment_counts: dict[str, int] = field(default_factory=dict)
    fallback_count: int = 0
    fallback_reason_counts: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in FALLBACK_REASON_KEYS}
    )
    similarity_score_buckets: dict[str, int] = field(
        default_factory=lambda: {bucket: 0 for bucket in SIMILARITY_BUCKET_KEYS}
    )
    skipped_existing_count: int = 0

    def accumulate_page(
        self,
        *,
        processed_user_count: int,
        users_to_match_count: int,
        skipped_existing_count: int,
    ) -> None:
        self.page_count += 1
        self.processed_user_count += processed_user_count
        self.users_to_match_count += users_to_match_count
        self.skipped_existing_count += skipped_existing_count

    def accumulate_matching(self, result: AssignmentPageMatchResult) -> None:
        self.ann_candidate_count += result.ann_candidate_count
        self.exact_reranked_pair_count += result.exact_reranked_pair_count
        self.ann_underfilled_user_count += result.ann_underfilled_user_count
        self.exact_rescue_user_count += result.exact_rescue_user_count
        self.ann_applied = self.ann_applied or result.ann_query_user_count > 0

    def accumulate_inserted(
        self,
        *,
        attempted_count: int,
        inserted_records: Sequence[UserSegmentAssignmentInsertRecord],
    ) -> None:
        inserted_count = len(inserted_records)
        if inserted_count > attempted_count:
            raise SegmentAssignmentValidationError(
                "inserted assignment count exceeded attempted count"
            )

        self.assignment_count += inserted_count
        self.insert_conflict_count += attempted_count - inserted_count
        for record in inserted_records:
            self.segment_assignment_counts[record.segment_id] = (
                self.segment_assignment_counts.get(record.segment_id, 0) + 1
            )
            if record.fallback:
                self.fallback_count += 1
                if record.fallback_reason in self.fallback_reason_counts:
                    self.fallback_reason_counts[record.fallback_reason] += 1
            bucket = _similarity_score_bucket(record.similarity_score)
            self.similarity_score_buckets[bucket] += 1

    @property
    def fallback_rate(self) -> float | None:
        if self.assignment_count == 0:
            return None
        return round(self.fallback_count / self.assignment_count, 6)

    @property
    def ann_not_applied_reason(self) -> str | None:
        if self.ann_applied:
            return None
        if self.users_to_match_count == 0:
            return ANN_NOT_APPLIED_NO_USERS
        return ANN_NOT_APPLIED_NO_VALID_VECTORS

    def validate_totals(self) -> None:
        expected_processed_count = (
            self.skipped_existing_count
            + self.assignment_count
            + self.insert_conflict_count
        )
        if self.processed_user_count != expected_processed_count:
            raise SegmentAssignmentValidationError(
                "assignment diagnostics totals are inconsistent"
            )


class AssignmentInputLoader:
    def __init__(
        self,
        *,
        promotion_run_repository: PromotionRunWriter,
        ad_experiment_repository: AdExperimentReader,
        segment_vector_repository: SegmentVectorReader,
        user_behavior_vector_repository: UserBehaviorVectorReader,
        page_size: int | None = None,
    ) -> None:
        page_size = ASSIGNMENT_PAGE_SIZE if page_size is None else page_size
        if page_size <= 0:
            raise ValueError("assignment page_size must be positive")
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._segment_vector_repository = segment_vector_repository
        self._user_behavior_vector_repository = user_behavior_vector_repository
        self._page_size = page_size

    def load(
        self,
        *,
        promotion_run_id: str,
        request: SegmentAssignmentBuildRequest,
    ) -> AssignmentBuildInput:
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            log.warn("promotion_run_not_found", {"promotionRunId": promotion_run_id})
            raise SegmentAssignmentRunNotFoundError(
                f"promotion run not found: {promotion_run_id}"
            )
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
        segment_vectors = self._load_segment_vectors(
            project_id=run.project_id,
            promotion_id=run.promotion_id,
            analysis_id=run.analysis_id,
            segment_ids=[
                experiment.segment_id for experiment in experiments.non_fallback
            ],
            vector_version=audience_scope.effective_vector_version,
        )
        return AssignmentBuildInput(
            run=run,
            audience_scope=audience_scope,
            experiments=experiments,
            segment_vectors=tuple(segment_vectors),
        )

    def iter_user_pages(
        self,
        build_input: AssignmentBuildInput,
    ) -> Iterator[list[UserVector]]:
        project_id = build_input.run.project_id
        audience_scope = build_input.audience_scope
        if audience_scope.user_ids is not None:
            user_ids = sorted(set(audience_scope.user_ids))
            if audience_scope.effective_limit is not None:
                user_ids = user_ids[: audience_scope.effective_limit]

            previous_user_id: str | None = None
            for index in range(0, len(user_ids), self._page_size):
                user_id_page = user_ids[index : index + self._page_size]
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
                self._page_size
                if remaining_limit is None
                else min(self._page_size, remaining_limit)
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
                log.warn(
                    "segment_vector_invalid",
                    {
                        "segmentId": segment_id,
                        "segmentVectorCount": len(segment_records),
                    },
                )
                raise SegmentAssignmentValidationError(
                    "each non-fallback segment must have exactly one segment vector: "
                    f"{segment_id}"
                )
            segment_vectors.append(
                _validated_segment_vector_from_record(
                    segment_records[0],
                    require_embedding=True,
                )
            )
        return segment_vectors


class AssignmentPageMatcher:
    def __init__(
        self,
        *,
        segment_vector_repository: SegmentVectorReader,
        reranker: SegmentCandidateReranker,
    ) -> None:
        self._segment_vector_repository = segment_vector_repository
        self._reranker = reranker

    def match_page(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        vector_version: str,
        users: Sequence[UserVector],
        segment_vectors: Sequence[SegmentVector],
    ) -> AssignmentPageMatchResult:
        if not segment_vectors:
            log.warn("segment_vectors_empty", {"analysisId": analysis_id})
            raise SegmentMatchValidationError(
                "at least one non-fallback segment vector is required"
            )

        expected_segments_by_vector_id = {
            segment.segment_vector_id: segment.segment_id
            for segment in segment_vectors
        }
        segment_vector_ids = list(expected_segments_by_vector_id)
        expected_candidate_count = min(len(segment_vector_ids), ANN_CANDIDATE_LIMIT)
        matches: dict[str, MatchResult] = {}
        ann_candidate_count = 0
        exact_reranked_pair_count = 0
        ann_underfilled_user_count = 0
        exact_rescue_user_count = 0

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
                ann_candidate_count += len(candidate_records)
                valid_candidates: list[SegmentVector] = []
                seen_vector_ids: set[str] = set()
                invalid_candidate_result = False
                for record in candidate_records:
                    expected_segment_id = expected_segments_by_vector_id.get(
                        record.segment_vector_id
                    )
                    if (
                        expected_segment_id is None
                        or record.segment_id != expected_segment_id
                        or record.segment_vector_id in seen_vector_ids
                    ):
                        invalid_candidate_result = True
                        continue
                    seen_vector_ids.add(record.segment_vector_id)
                    try:
                        valid_candidates.append(
                            _validated_segment_vector_from_record(
                                record,
                                require_embedding=True,
                            )
                        )
                    except (SegmentAssignmentValidationError, ValueError):
                        invalid_candidate_result = True

                candidate_count = len(valid_candidates)
                is_underfilled = candidate_count < expected_candidate_count
                requires_exact_rescue = (
                    invalid_candidate_result
                    or candidate_count != expected_candidate_count
                )
                if is_underfilled:
                    ann_underfilled_user_count += 1
                if requires_exact_rescue:
                    exact_rescue_user_count += 1
                    rerank_candidates = segment_vectors
                else:
                    rerank_candidates = valid_candidates
                exact_reranked_pair_count += len(rerank_candidates)

                matches[user_id] = self._reranker.rerank(
                    normalized_user_vector=normalized_user_vector,
                    candidates=rerank_candidates,
                )

        return AssignmentPageMatchResult(
            matches=matches,
            ann_candidate_count=ann_candidate_count,
            exact_reranked_pair_count=exact_reranked_pair_count,
            ann_underfilled_user_count=ann_underfilled_user_count,
            exact_rescue_user_count=exact_rescue_user_count,
            ann_query_user_count=len(normalized_users),
        )


class ExactAssignmentPageMatcher:
    """Production-safe exact matcher for future build workers."""

    def __init__(self, *, reranker: SegmentCandidateReranker) -> None:
        self._reranker = reranker

    def match_page(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        vector_version: str,
        users: Sequence[UserVector],
        segment_vectors: Sequence[SegmentVector],
    ) -> AssignmentPageMatchResult:
        del project_id, promotion_id, vector_version
        if not segment_vectors:
            log.warn("segment_vectors_empty", {"analysisId": analysis_id})
            raise SegmentMatchValidationError(
                "at least one non-fallback segment vector is required"
            )
        matches: dict[str, MatchResult] = {}
        exact_reranked_pair_count = 0
        for user in _deduplicate_users_by_id(users):
            normalized_user_vector = self._reranker.normalize_user_vector(user)
            if normalized_user_vector is None:
                matches[user.user_id] = invalid_user_vector_result()
                continue
            exact_reranked_pair_count += len(segment_vectors)
            matches[user.user_id] = self._reranker.rerank(
                normalized_user_vector=normalized_user_vector,
                candidates=segment_vectors,
            )
        return AssignmentPageMatchResult(
            matches=matches,
            ann_candidate_count=0,
            exact_reranked_pair_count=exact_reranked_pair_count,
            ann_underfilled_user_count=0,
            exact_rescue_user_count=0,
            ann_query_user_count=0,
        )


class AssignmentResultWriter:
    def __init__(
        self,
        *,
        user_segment_assignment_repository: UserSegmentAssignmentWriter,
    ) -> None:
        self._repository = user_segment_assignment_repository

    def select_unassigned(
        self,
        *,
        promotion_run_id: str,
        eligible_users: Sequence[UserVector],
    ) -> AssignmentPageSelection:
        existing_user_ids = self._repository.list_existing_user_ids(
            promotion_run_id=promotion_run_id,
            user_ids=[user.user_id for user in eligible_users],
        )
        users_to_match = tuple(
            user
            for user in eligible_users
            if user.user_id not in existing_user_ids
        )
        return AssignmentPageSelection(
            eligible_users=tuple(eligible_users),
            users_to_match=users_to_match,
            skipped_existing_count=len(existing_user_ids),
        )

    def write_page(
        self,
        *,
        project_id: str,
        promotion_run_id: str,
        match_result: AssignmentPageMatchResult,
        experiments: AssignmentExperimentSet,
        assigned_at: datetime,
        expires_in_days: int | None,
        page_number: int,
    ) -> AssignmentPageWriteOutcome:
        if match_result.fallback_count > 0 and experiments.fallback is None:
            log.warn(
                "fallback_ad_experiment_missing",
                {
                    "pageNumber": page_number,
                    "fallbackCount": match_result.fallback_count,
                },
            )
            raise SegmentAssignmentValidationError(
                "fallback ad experiment is required when fallback assignments exist"
            )
        assignments = _build_assignment_writes(
            project_id=project_id,
            promotion_run_id=promotion_run_id,
            matches=match_result.matches,
            experiments=experiments,
            assigned_at=assigned_at,
            expires_in_days=expires_in_days,
        )
        inserted_records = tuple(self._repository.insert_many(assignments))
        return AssignmentPageWriteOutcome(
            attempted_count=len(assignments),
            inserted_records=inserted_records,
        )

    def summarize_run(
        self,
        promotion_run_id: str,
    ) -> UserSegmentAssignmentRunAggregateRecord:
        return self._repository.summarize_run(promotion_run_id)


class SegmentAssignmentService:
    def __init__(
        self,
        *,
        input_loader: AssignmentInputLoader,
        page_matcher: AssignmentPageMatcher,
        result_writer: AssignmentResultWriter,
    ) -> None:
        self._input_loader = input_loader
        self._page_matcher = page_matcher
        self._result_writer = result_writer

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
        build_input = self._input_loader.load(
            promotion_run_id=promotion_run_id,
            request=request,
        )
        run = build_input.run
        audience_scope = build_input.audience_scope
        experiments = build_input.experiments
        segment_vectors = build_input.segment_vectors
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
        log.info(
            "ad_experiments_loaded",
            {
                "nonFallbackCount": len(experiments.non_fallback),
                "hasFallback": experiments.fallback is not None,
            },
        )
        log.info(
            "segment_vectors_loaded",
            {"segmentVectorCount": len(segment_vectors)},
        )
        diagnostics = AssignmentDiagnostics()
        assigned_at = datetime.now(UTC)
        for page_number, eligible_users in enumerate(
            self._input_loader.iter_user_pages(build_input),
            start=1,
        ):
            if not eligible_users:
                continue
            log.info(
                "eligible_users_page_loaded",
                {
                    "pageNumber": page_number,
                    "eligibleUserCount": len(eligible_users),
                },
            )
            selection = self._result_writer.select_unassigned(
                promotion_run_id=run.promotion_run_id,
                eligible_users=eligible_users,
            )
            diagnostics.accumulate_page(
                processed_user_count=len(selection.eligible_users),
                users_to_match_count=len(selection.users_to_match),
                skipped_existing_count=selection.skipped_existing_count,
            )
            if selection.skipped_existing_count:
                log.info(
                    "existing_assignments_skipped",
                    {
                        "pageNumber": page_number,
                        "userCount": selection.skipped_existing_count,
                    },
                )

            try:
                match_result = self._page_matcher.match_page(
                    project_id=run.project_id,
                    promotion_id=run.promotion_id,
                    analysis_id=run.analysis_id,
                    vector_version=audience_scope.effective_vector_version,
                    users=selection.users_to_match,
                    segment_vectors=segment_vectors,
                )
            except (SegmentMatchValidationError, ValueError) as exc:
                log.warn("segment_matching_invalid", {"err": exc})
                raise SegmentAssignmentValidationError(str(exc)) from exc

            log.info(
                "segment_matches_created",
                {
                    "pageNumber": page_number,
                    "matchCount": len(match_result.matches),
                    "fallbackCount": match_result.fallback_count,
                    "annCandidateCount": match_result.ann_candidate_count,
                    "exactRerankedPairCount": (
                        match_result.exact_reranked_pair_count
                    ),
                },
            )

            write_outcome = self._result_writer.write_page(
                project_id=run.project_id,
                promotion_run_id=run.promotion_run_id,
                match_result=match_result,
                experiments=experiments,
                assigned_at=assigned_at,
                expires_in_days=request.expires_in_days,
                page_number=page_number,
            )
            diagnostics.accumulate_matching(match_result)
            diagnostics.accumulate_inserted(
                attempted_count=write_outcome.attempted_count,
                inserted_records=write_outcome.inserted_records,
            )
            log.info(
                "segment_assignments_page_created",
                {
                    "pageNumber": page_number,
                    "assignmentCount": len(write_outcome.inserted_records),
                    "insertConflictCount": (
                        write_outcome.attempted_count
                        - len(write_outcome.inserted_records)
                    ),
                },
            )

        diagnostics.validate_totals()
        run_aggregate = self._result_writer.summarize_run(run.promotion_run_id)
        assignment_mode = (
            ASSIGNMENT_MODE_EXPLICIT_USER_IDS
            if audience_scope.user_ids is not None
            else ASSIGNMENT_MODE_LIVE_KEYSET
        )

        response = SegmentAssignmentBuildResponse(
            promotion_run_id=run.promotion_run_id,
            matching_mode=MATCHING_MODE,
            vector_version=audience_scope.effective_vector_version,
            ann_candidate_limit=ANN_CANDIDATE_LIMIT,
            ann_candidate_count=diagnostics.ann_candidate_count,
            exact_reranked_pair_count=diagnostics.exact_reranked_pair_count,
            page_count=diagnostics.page_count,
            processed_user_count=diagnostics.processed_user_count,
            assignment_count=diagnostics.assignment_count,
            run_assignment_count=run_aggregate.assignment_count,
            run_has_fallback=run_aggregate.fallback_count > 0,
            run_fallback_count=run_aggregate.fallback_count,
            insert_conflict_count=diagnostics.insert_conflict_count,
            segment_assignment_counts=diagnostics.segment_assignment_counts,
            batch_has_fallback=diagnostics.fallback_count > 0,
            fallback_count=diagnostics.fallback_count,
            fallback_rate=diagnostics.fallback_rate,
            fallback_reason_counts=diagnostics.fallback_reason_counts,
            below_threshold_fallback_count=(
                diagnostics.fallback_reason_counts[
                    FALLBACK_REASON_BELOW_THRESHOLD
                ]
            ),
            no_candidate_fallback_count=diagnostics.fallback_reason_counts[
                FALLBACK_REASON_NO_CANDIDATE
            ],
            invalid_user_vector_fallback_count=(
                diagnostics.fallback_reason_counts[
                    FALLBACK_REASON_INVALID_USER_VECTOR
                ]
            ),
            similarity_score_buckets=diagnostics.similarity_score_buckets,
            ann_underfilled_user_count=diagnostics.ann_underfilled_user_count,
            exact_rescue_user_count=diagnostics.exact_rescue_user_count,
            ann_applied=diagnostics.ann_applied,
            ann_not_applied_reason=diagnostics.ann_not_applied_reason,
            skipped_existing_count=diagnostics.skipped_existing_count,
            insufficient_segment_count=0,
            completion_scope="current_request",
            assignment_mode=assignment_mode,
            input_stability="not_snapshotted",
            status="completed",
        )
        log.info(
            "assignment_diagnostics",
            {"diagnostics": response.model_dump(mode="json")},
        )
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response


def _split_experiments(
    experiments: Sequence[AdExperimentRecord],
) -> AssignmentExperimentSet:
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
    return AssignmentExperimentSet(non_fallback=non_fallback, fallback=fallback)


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
) -> AssignmentAudienceScope:
    del project_id  # promotion_run.project_id is the only allowed project context.
    raw_scope = goal_snapshot_json.get("audience_scope")
    if raw_scope is not None and not isinstance(raw_scope, Mapping):
        raise SegmentAssignmentValidationError("audience_scope must be an object")

    request_vector_version_is_explicit = "vector_version" in request.model_fields_set
    request_user_ids = list(request.user_ids) if request.user_ids else None
    request_limit = request.eligible_user_limit

    if raw_scope is None:
        return AssignmentAudienceScope(
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

    return AssignmentAudienceScope(
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
    experiments: AssignmentExperimentSet,
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
        experiment = experiments_by_segment.get(result.segment_id)
        if experiment is None:
            raise SegmentAssignmentValidationError(
                "match references a segment without an ad experiment: "
                f"{result.segment_id}"
            )
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


def _validated_segment_vector_from_record(
    record: SegmentVectorRecord,
    *,
    require_embedding: bool,
) -> SegmentVector:
    segment_vector = _segment_vector_from_record(
        record,
        require_embedding=require_embedding,
    )
    try:
        normalize_values(
            segment_vector.embedding_values,
            segment_vector.vector_dim,
        )
    except ValueError as exc:
        raise SegmentAssignmentValidationError(
            f"invalid segment embedding: {record.segment_id}"
        ) from exc
    return segment_vector


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
    # The persisted Data Contract is [0, 1], so negative raw cosine values are
    # intentionally represented as 0.000000 before diagnostics are bucketed.
    clamped_score = min(1.0, max(0.0, float(score)))
    return Decimal(str(clamped_score)).quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_UP,
    )


def _similarity_score_bucket(score: Decimal | None) -> str:
    if score is None:
        return "not_available"
    if score < Decimal("0.50"):
        return "0_00_to_0_50"
    if score < Decimal("0.65"):
        return "0_50_to_0_65"
    if score < Decimal("0.80"):
        return "0_65_to_0_80"
    if score < Decimal("0.90"):
        return "0_80_to_0_90"
    return "gte_0_90"
