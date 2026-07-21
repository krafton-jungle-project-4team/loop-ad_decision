from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterator, Mapping, Sequence

from psycopg import errors as pg_errors

from app.decision.audience_snapshots import (
    AudienceSnapshotContractError,
    AudienceSnapshotBinding,
    AudienceSnapshotReader,
)
from app.audience_contract import SEGMENT_AUDIENCE_CONTRACT
from app.decision.experiment_assignment_repository import (
    AdExperimentUnitRecord,
    AdExperimentUnitWrite,
    ExperimentAssignmentWriter,
    SegmentAssignmentExecutionRecord,
    SegmentAssignmentExecutionWrite,
)
from app.decision.experiment_design import (
    EXECUTION_SCHEMA_VERSION,
    ExperimentAllocationResult,
    ExperimentAudienceMember,
    ExperimentDesign,
    ExperimentDesignConflictError,
    ExperimentDesignValidationError,
    ExperimentUnitAllocation,
    RandomizedHoldoutAudienceTooSmallError,
    RandomizedHoldoutConfigurationError,
    allocate_experiment_units,
    build_execution_id,
    build_experiment_design,
    build_experiment_unit_id,
    build_input_fingerprint,
    build_request_fingerprint,
)
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
    AdExperimentReader,
    EvaluationMetricReader,
    PromotionAnalysisReader,
    PromotionEvaluationWriter,
    PromotionRunRecord,
    PromotionRunWriter,
    SegmentVectorReader,
    SegmentVectorRecord,
    UserBehaviorVectorRecord,
    UserBehaviorVectorReader,
    UserSegmentAssignmentInsertRecord,
    UserSegmentAssignmentSourceRecord,
    UserSegmentAssignmentWrite,
    UserSegmentAssignmentWriter,
)
from app.decision.schemas import (
    AssignmentSource,
    SegmentAssignmentBuildRequest,
    SegmentAssignmentBuildResponse,
)
from app.decision.outcome_spec import require_frozen_outcome_spec
from app.logging import log, log_context_scope, now_ms, duration_ms


MATCHING_MODE = "pgvector_hnsw_rerank"
ASSIGNMENT_PAGE_SIZE = 10_000
DEFAULT_VECTOR_VERSION = "v1"
AUDIENCE_SCOPE_BASE = "user_behavior_vectors"
ASSIGNMENT_MODE_LIVE_KEYSET = "live_keyset"
ASSIGNMENT_MODE_EXPLICIT_USER_IDS = "explicit_user_ids"
ASSIGNMENT_MODE_ANALYSIS_SNAPSHOT = "analysis_snapshot"
SNAPSHOT_MATCHING_MODE = "analysis_snapshot_reuse"
ANN_NOT_APPLIED_NO_USERS = "no_users_to_match"
ANN_NOT_APPLIED_NO_VALID_VECTORS = "no_valid_user_vectors"
FALLBACK_REASON_KEYS = (
    FALLBACK_REASON_BELOW_THRESHOLD,
    FALLBACK_REASON_NO_CANDIDATE,
    FALLBACK_REASON_INVALID_USER_VECTOR,
)
SIMILARITY_BUCKET_KEYS = (
    "not_available",
    "lt_0_00",
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


class SegmentAssignmentAudienceContractError(SegmentAssignmentValidationError):
    def __init__(self, *, code: str, segment_id: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.segment_id = segment_id
        self.reason = reason

    def to_detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "segment_id": self.segment_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _ExperimentSet:
    non_fallback: list[AdExperimentRecord]
    fallback: AdExperimentRecord | None


@dataclass(frozen=True)
class _NextLoopRetryContext:
    source_promotion_run_id: str
    failed_ad_experiment_ids: tuple[str, ...]


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


@dataclass
class _BuildDiagnostics:
    page_count: int = 0
    processed_user_count: int = 0
    users_to_match_count: int = 0
    ann_candidate_count: int = 0
    exact_reranked_pair_count: int = 0
    ann_underfilled_user_count: int = 0
    ann_applied: bool = False
    assignment_count: int = 0
    insert_conflict_count: int = 0
    segment_assignment_counts: dict[str, int] = field(default_factory=dict)
    fallback_count: int = 0
    fallback_reason_counts: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in FALLBACK_REASON_KEYS}
    )
    unassigned_count: int = 0
    unassigned_reason_counts: dict[str, int] = field(
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

    def accumulate_matching(self, result: _BuildMatchResult) -> None:
        self.ann_candidate_count += result.ann_candidate_count
        self.exact_reranked_pair_count += result.exact_reranked_pair_count
        self.ann_underfilled_user_count += result.ann_underfilled_user_count
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

    def accumulate_unassigned(self, matches: Mapping[str, MatchResult]) -> None:
        for result in matches.values():
            if not result.fallback:
                continue
            self.unassigned_count += 1
            if result.fallback_reason in self.unassigned_reason_counts:
                self.unassigned_reason_counts[result.fallback_reason] += 1
            bucket = _similarity_score_bucket(
                _score_to_decimal(result.similarity_score)
            )
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
            + self.unassigned_count
        )
        if self.processed_user_count != expected_processed_count:
            raise SegmentAssignmentValidationError(
                "assignment diagnostics totals are inconsistent"
            )


class SegmentAssignmentService:
    def __init__(
        self,
        *,
        promotion_run_repository: PromotionRunWriter,
        ad_experiment_repository: AdExperimentReader,
        segment_vector_repository: SegmentVectorReader,
        user_behavior_vector_repository: UserBehaviorVectorReader,
        user_segment_assignment_repository: UserSegmentAssignmentWriter,
        reranker: SegmentCandidateReranker,
        audience_snapshot_repository: AudienceSnapshotReader,
        promotion_analysis_repository: PromotionAnalysisReader | None = None,
        promotion_evaluation_repository: PromotionEvaluationWriter | None = None,
        evaluation_metric_repository: EvaluationMetricReader | None = None,
        experiment_assignment_repository: ExperimentAssignmentWriter | None = None,
        randomization_salt: str | None = None,
    ) -> None:
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._segment_vector_repository = segment_vector_repository
        self._user_behavior_vector_repository = user_behavior_vector_repository
        self._user_segment_assignment_repository = user_segment_assignment_repository
        self._reranker = reranker
        self._audience_snapshot_repository = audience_snapshot_repository
        self._promotion_analysis_repository = promotion_analysis_repository
        self._promotion_evaluation_repository = promotion_evaluation_repository
        self._evaluation_metric_repository = evaluation_metric_repository
        self._experiment_assignment_repository = experiment_assignment_repository
        self._randomization_salt = randomization_salt

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

        experiments = _split_experiments(
            self._ad_experiment_repository.list_by_run(promotion_run_id)
        )
        if not experiments.non_fallback:
            log.warn("ad_experiments_empty", {"promotionRunId": promotion_run_id})
            raise SegmentAssignmentValidationError(
                "at least one non-fallback ad experiment is required"
            )
        log.info("ad_experiments_loaded", {"nonFallbackCount": len(experiments.non_fallback), "hasFallback": experiments.fallback is not None})

        retry_context = self._resolve_next_loop_retry_context(run)
        if retry_context is not None:
            if _requested_experiment_mode(request) == "randomized_holdout":
                raise SegmentAssignmentValidationError(
                    "randomized_holdout requires a Segment Audience V2 final snapshot"
                )
            return self._build_next_loop_retry_assignments(
                run=run,
                experiments=experiments,
                request=request,
                retry_context=retry_context,
                started_at=started_at,
            )

        audience_contract = self._resolve_target_audience_contract(
            promotion_run_id=run.promotion_run_id,
            analysis_id=run.analysis_id,
            segment_ids=[
                experiment.segment_id for experiment in experiments.non_fallback
            ],
        )
        if audience_contract == SEGMENT_AUDIENCE_CONTRACT:
            return self._build_snapshot_assignments(
                run=run,
                experiments=experiments,
                request=request,
                started_at=started_at,
            )

        if _requested_experiment_mode(request) == "randomized_holdout":
            raise SegmentAssignmentValidationError(
                "randomized_holdout requires a Segment Audience V2 final snapshot"
            )

        audience_scope = _build_effective_audience_scope(
            goal_snapshot_json=run.goal_snapshot_json,
            request=request,
            project_id=run.project_id,
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
            if not eligible_users:
                continue
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
            users_to_match = [
                user
                for user in eligible_users
                if user.user_id not in existing_user_ids
            ]
            diagnostics.accumulate_page(
                processed_user_count=len(eligible_users),
                users_to_match_count=len(users_to_match),
                skipped_existing_count=len(existing_user_ids),
            )
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

            assignment_matches = build_result.matches
            if experiments.fallback is None:
                unassigned_matches = {
                    user_id: result
                    for user_id, result in build_result.matches.items()
                    if result.fallback
                }
                diagnostics.accumulate_unassigned(unassigned_matches)
                assignment_matches = {
                    user_id: result
                    for user_id, result in build_result.matches.items()
                    if not result.fallback
                }
            log.info(
                "segment_matches_created",
                {
                    "pageNumber": page_number,
                    "matchCount": len(build_result.matches),
                    "fallbackCount": build_result.fallback_count,
                    "unassignedCount": len(unassigned_matches)
                    if experiments.fallback is None
                    else 0,
                    "annCandidateCount": build_result.ann_candidate_count,
                    "exactRerankedPairCount": build_result.exact_reranked_pair_count,
                },
            )

            assignments = _build_assignment_writes(
                project_id=run.project_id,
                promotion_run_id=run.promotion_run_id,
                matches=assignment_matches,
                experiments=experiments,
                assigned_at=assigned_at,
                expires_in_days=request.expires_in_days,
            )
            inserted_records = (
                self._user_segment_assignment_repository.insert_many(assignments)
            )
            diagnostics.accumulate_matching(build_result)
            diagnostics.accumulate_inserted(
                attempted_count=len(assignments),
                inserted_records=inserted_records,
            )
            log.info(
                "segment_assignments_page_created",
                {
                    "pageNumber": page_number,
                    "assignmentCount": len(inserted_records),
                    "insertConflictCount": len(assignments) - len(inserted_records),
                },
            )

        diagnostics.validate_totals()
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
            unassigned_count=diagnostics.unassigned_count,
            unassigned_reason_counts=diagnostics.unassigned_reason_counts,
            below_threshold_unassigned_count=(
                diagnostics.unassigned_reason_counts[
                    FALLBACK_REASON_BELOW_THRESHOLD
                ]
            ),
            no_candidate_unassigned_count=diagnostics.unassigned_reason_counts[
                FALLBACK_REASON_NO_CANDIDATE
            ],
            invalid_user_vector_unassigned_count=(
                diagnostics.unassigned_reason_counts[
                    FALLBACK_REASON_INVALID_USER_VECTOR
                ]
            ),
            similarity_score_buckets=diagnostics.similarity_score_buckets,
            ann_underfilled_user_count=diagnostics.ann_underfilled_user_count,
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

    def _resolve_next_loop_retry_context(
        self,
        run: PromotionRunRecord,
    ) -> _NextLoopRetryContext | None:
        repository = self._promotion_analysis_repository
        if repository is None:
            return None
        analysis = repository.get_by_id(run.analysis_id)
        if analysis is None:
            return None
        raw_context = analysis.input_snapshot_json.get("next_loop")
        if raw_context is None:
            return None
        if not isinstance(raw_context, Mapping):
            raise SegmentAssignmentValidationError(
                "next-loop analysis context is invalid"
            )
        source_run_id = raw_context.get("source_promotion_run_id")
        failed_experiment_ids = raw_context.get(
            "source_failed_ad_experiment_ids"
        )
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise SegmentAssignmentValidationError(
                "next-loop source promotion run is invalid"
            )
        if (
            not isinstance(failed_experiment_ids, list)
            or not failed_experiment_ids
            or any(
                not isinstance(experiment_id, str)
                or not experiment_id.strip()
                for experiment_id in failed_experiment_ids
            )
        ):
            raise SegmentAssignmentValidationError(
                "next-loop failed ad experiments are invalid"
            )
        normalized_experiment_ids = tuple(
            sorted({experiment_id.strip() for experiment_id in failed_experiment_ids})
        )
        return _NextLoopRetryContext(
            source_promotion_run_id=source_run_id.strip(),
            failed_ad_experiment_ids=normalized_experiment_ids,
        )

    def _build_next_loop_retry_assignments(
        self,
        *,
        run: PromotionRunRecord,
        experiments: _ExperimentSet,
        request: SegmentAssignmentBuildRequest,
        retry_context: _NextLoopRetryContext,
        started_at: int,
    ) -> SegmentAssignmentBuildResponse:
        evaluation_repository = self._promotion_evaluation_repository
        metric_repository = self._evaluation_metric_repository
        if evaluation_repository is None or metric_repository is None:
            raise SegmentAssignmentValidationError(
                "next-loop retry assignment repositories are not configured"
            )
        if request.user_ids or request.eligible_user_limit is not None:
            raise SegmentAssignmentValidationError(
                "next-loop retry assignment does not accept user_ids or a limit"
            )

        source_run = self._promotion_run_repository.get_by_id(
            retry_context.source_promotion_run_id
        )
        if source_run is None:
            raise SegmentAssignmentValidationError(
                "next-loop source promotion run was not found"
            )
        if (
            source_run.project_id != run.project_id
            or source_run.promotion_id != run.promotion_id
            or source_run.loop_count + 1 != run.loop_count
        ):
            raise SegmentAssignmentValidationError(
                "next-loop source promotion run does not match the target run"
            )

        source_experiments = {
            experiment.ad_experiment_id: experiment
            for experiment in self._ad_experiment_repository.list_by_run(
                source_run.promotion_run_id
            )
            if experiment.ad_experiment_id
            in retry_context.failed_ad_experiment_ids
        }
        if set(source_experiments) != set(retry_context.failed_ad_experiment_ids):
            raise SegmentAssignmentValidationError(
                "next-loop source ad experiments were not found"
            )
        target_experiments = {
            experiment.segment_id: experiment
            for experiment in experiments.non_fallback
        }
        missing_target_segments = sorted(
            {
                experiment.segment_id
                for experiment in source_experiments.values()
                if experiment.segment_id not in target_experiments
            }
        )
        if missing_target_segments:
            raise SegmentAssignmentValidationError(
                "next-loop target experiments are missing segments: "
                + ", ".join(missing_target_segments)
            )

        evaluations = {
            evaluation.ad_experiment_id: evaluation
            for evaluation in evaluation_repository.list_latest_by_run_ad_experiments(
                source_run.promotion_run_id
            )
            if evaluation.ad_experiment_id
            in retry_context.failed_ad_experiment_ids
        }
        if set(evaluations) != set(retry_context.failed_ad_experiment_ids):
            raise SegmentAssignmentValidationError(
                "next-loop source evaluations were not found"
            )
        evaluation_cutoffs = {
            experiment_id: _parse_evaluation_cutoff(
                evaluations[experiment_id].result_json
            )
            for experiment_id in retry_context.failed_ad_experiment_ids
        }

        assigned_at = datetime.now(UTC)
        expires_at = (
            assigned_at + timedelta(days=request.expires_in_days)
            if request.expires_in_days is not None
            else None
        )
        page_count = 0
        processed_count = 0
        retry_user_count = 0
        assignment_count = 0
        conflict_count = 0
        skipped_existing_count = 0
        successful_user_count = 0
        segment_counts: dict[str, int] = {}
        score_buckets = {bucket: 0 for bucket in SIMILARITY_BUCKET_KEYS}
        after_user_id: str | None = None

        while True:
            source_assignments = (
                self._user_segment_assignment_repository.list_source_page(
                    promotion_run_id=source_run.promotion_run_id,
                    ad_experiment_ids=retry_context.failed_ad_experiment_ids,
                    after_user_id=after_user_id,
                    limit=ASSIGNMENT_PAGE_SIZE,
                )
            )
            if not source_assignments:
                break
            page_count += 1
            processed_count += len(source_assignments)
            assignments_by_experiment: dict[
                str, list[UserSegmentAssignmentSourceRecord]
            ] = defaultdict(list)
            for source_assignment in source_assignments:
                assignments_by_experiment[
                    source_assignment.ad_experiment_id
                ].append(source_assignment)

            successful_user_ids: set[str] = set()
            for experiment_id, experiment_assignments in assignments_by_experiment.items():
                successful_user_ids.update(
                    metric_repository.list_successful_user_ids(
                        source_experiments[experiment_id],
                        user_ids=[
                            assignment.user_id
                            for assignment in experiment_assignments
                        ],
                        evaluation_cutoff_at=evaluation_cutoffs[experiment_id],
                    )
                )
            successful_user_count += len(successful_user_ids)
            retry_assignments = [
                assignment
                for assignment in source_assignments
                if assignment.user_id not in successful_user_ids
            ]
            retry_user_count += len(retry_assignments)
            existing_user_ids = (
                self._user_segment_assignment_repository.list_existing_user_ids(
                    promotion_run_id=run.promotion_run_id,
                    user_ids=[assignment.user_id for assignment in retry_assignments],
                )
            )
            skipped_existing_count += len(existing_user_ids)
            writes: list[UserSegmentAssignmentWrite] = []
            for source_assignment in retry_assignments:
                if source_assignment.user_id in existing_user_ids:
                    continue
                source_experiment = source_experiments[
                    source_assignment.ad_experiment_id
                ]
                if source_assignment.segment_id != source_experiment.segment_id:
                    raise SegmentAssignmentValidationError(
                        "next-loop source assignment segment does not match "
                        "its experiment"
                    )
                target_experiment = target_experiments[
                    source_experiment.segment_id
                ]
                writes.append(
                    UserSegmentAssignmentWrite(
                        project_id=run.project_id,
                        promotion_run_id=run.promotion_run_id,
                        user_id=source_assignment.user_id,
                        segment_id=source_experiment.segment_id,
                        ad_experiment_id=target_experiment.ad_experiment_id,
                        content_id=target_experiment.content_id,
                        content_option_id=target_experiment.content_option_id,
                        similarity_score=source_assignment.similarity_score,
                        fallback=False,
                        fallback_reason=None,
                        assignment_source=AssignmentSource.DECISION_BATCH.value,
                        assigned_at=assigned_at,
                        expires_at=expires_at,
                    )
                )
            inserted = self._user_segment_assignment_repository.insert_many(writes)
            assignment_count += len(inserted)
            conflict_count += len(writes) - len(inserted)
            for record in inserted:
                segment_counts[record.segment_id] = (
                    segment_counts.get(record.segment_id, 0) + 1
                )
                score_buckets[_similarity_score_bucket(record.similarity_score)] += 1
            after_user_id = source_assignments[-1].user_id
            if len(source_assignments) < ASSIGNMENT_PAGE_SIZE:
                break

        if processed_count == 0:
            raise SegmentAssignmentValidationError(
                "next-loop source assignments were not found"
            )
        if retry_user_count == 0:
            raise SegmentAssignmentValidationError(
                "next-loop retry audience is empty because every assigned user succeeded"
            )

        response = SegmentAssignmentBuildResponse(
            promotion_run_id=run.promotion_run_id,
            matching_mode=SNAPSHOT_MATCHING_MODE,
            vector_version=request.vector_version,
            ann_candidate_limit=0,
            ann_candidate_count=0,
            exact_reranked_pair_count=0,
            page_count=page_count,
            processed_user_count=processed_count,
            assignment_count=assignment_count,
            insert_conflict_count=conflict_count,
            segment_assignment_counts=segment_counts,
            batch_has_fallback=False,
            fallback_count=0,
            fallback_rate=0.0 if assignment_count else None,
            fallback_reason_counts={reason: 0 for reason in FALLBACK_REASON_KEYS},
            below_threshold_fallback_count=0,
            no_candidate_fallback_count=0,
            invalid_user_vector_fallback_count=0,
            unassigned_count=0,
            unassigned_reason_counts={reason: 0 for reason in FALLBACK_REASON_KEYS},
            below_threshold_unassigned_count=0,
            no_candidate_unassigned_count=0,
            invalid_user_vector_unassigned_count=0,
            similarity_score_buckets=score_buckets,
            ann_underfilled_user_count=0,
            ann_applied=False,
            ann_not_applied_reason="analysis_snapshot_reuse",
            skipped_existing_count=skipped_existing_count,
            insufficient_segment_count=0,
            completion_scope="current_request",
            assignment_mode=ASSIGNMENT_MODE_ANALYSIS_SNAPSHOT,
            input_stability="snapshotted",
            status="completed",
        )
        log.info(
            "next_loop_retry_assignments_completed",
            {
                "sourcePromotionRunId": source_run.promotion_run_id,
                "successfulUserCount": successful_user_count,
                "retryUserCount": retry_user_count,
                "response": response.model_dump(mode="json"),
                "durationMs": duration_ms(started_at),
            },
        )
        return response

    def _resolve_target_audience_contract(
        self,
        *,
        promotion_run_id: str,
        analysis_id: str,
        segment_ids: Sequence[str],
    ) -> str:
        try:
            resolver = getattr(
                self._audience_snapshot_repository,
                "resolve_run_contract",
                None,
            )
            if callable(resolver):
                return resolver(
                    promotion_run_id=promotion_run_id,
                    analysis_id=analysis_id,
                    segment_ids=segment_ids,
                ).contract
            return self._audience_snapshot_repository.resolve_target_contract(
                analysis_id=analysis_id,
                segment_ids=segment_ids,
            ).contract
        except (pg_errors.UndefinedTable, pg_errors.UndefinedColumn) as exc:
            raise SegmentAssignmentAudienceContractError(
                code="segment_audience_exclusion_contract_missing",
                segment_id=",".join(segment_ids),
                reason="V2 run-target binding Data Contract is missing",
            ) from exc
        except AudienceSnapshotContractError as exc:
            raise SegmentAssignmentAudienceContractError(
                code="segment_audience_assignment_contract_invalid",
                segment_id=",".join(segment_ids),
                reason=str(exc),
            ) from exc

    def _build_snapshot_assignments(
        self,
        *,
        run: PromotionRunRecord,
        experiments: _ExperimentSet,
        request: SegmentAssignmentBuildRequest,
        started_at: int,
    ) -> SegmentAssignmentBuildResponse:
        repository = self._experiment_assignment_repository
        has_frozen_outcome = isinstance(
            run.goal_snapshot_json.get("outcome_spec"), Mapping
        )
        if repository is None or not has_frozen_outcome:
            if _requested_experiment_mode(request) == "randomized_holdout":
                raise SegmentAssignmentValidationError(
                    "randomized_holdout requires an uplift-ready promotion run"
                )
            return self._build_legacy_snapshot_assignments(
                run=run,
                experiments=experiments,
                request=request,
                started_at=started_at,
            )

        lock_run = getattr(
            self._promotion_run_repository,
            "get_by_id_for_update",
            None,
        )
        if not callable(lock_run):
            raise SegmentAssignmentValidationError(
                "promotion run row locking is not configured"
            )
        locked_run = lock_run(run.promotion_run_id)
        if locked_run is None:
            raise SegmentAssignmentRunNotFoundError(
                f"promotion run not found: {run.promotion_run_id}"
            )

        try:
            outcome_spec, frozen_outcome_spec_hash = require_frozen_outcome_spec(
                locked_run.goal_snapshot_json
            )
        except ValueError as exc:
            raise SegmentAssignmentValidationError(str(exc)) from exc

        design_request = request.experiment_design
        mode = _requested_experiment_mode(request)
        treatment_ratio = (
            design_request.treatment_ratio if design_request is not None else None
        )
        outcome_window_days = (
            design_request.outcome_window_days if design_request is not None else 30
        )
        design = build_experiment_design(
            mode=mode,
            treatment_ratio=treatment_ratio,
            outcome_window_days=outcome_window_days,
            outcome_spec_hash=frozen_outcome_spec_hash,
            randomization_salt=self._randomization_salt,
        )

        if request.user_ids or request.eligible_user_limit is not None:
            raise SegmentAssignmentValidationError(
                "analysis snapshot assignment does not accept user_ids or a limit"
            )
        segment_ids = [
            experiment.segment_id for experiment in experiments.non_fallback
        ]
        try:
            snapshot_set = self._audience_snapshot_repository.require_run_binding_set(
                promotion_run_id=locked_run.promotion_run_id,
                segment_ids=segment_ids,
            )
        except (pg_errors.UndefinedTable, pg_errors.UndefinedColumn) as exc:
            raise SegmentAssignmentAudienceContractError(
                code="segment_audience_exclusion_contract_missing",
                segment_id=",".join(segment_ids),
                reason="V2 run-target binding Data Contract is missing",
            ) from exc
        except AudienceSnapshotContractError as exc:
            raise SegmentAssignmentAudienceContractError(
                code="segment_audience_snapshot_binding_invalid",
                segment_id=",".join(segment_ids),
                reason=str(exc),
            ) from exc
        if len(snapshot_set.bindings) != len(experiments.non_fallback):
            raise SegmentAssignmentAudienceContractError(
                code="segment_audience_snapshot_binding_invalid",
                segment_id=",".join(segment_ids),
                reason="final snapshot binding metadata is incomplete",
            )

        experiment_by_segment = {
            experiment.segment_id: experiment
            for experiment in experiments.non_fallback
        }
        binding_by_segment = {
            binding.segment_id: binding for binding in snapshot_set.bindings
        }
        audience_bindings = [
            _binding_manifest(
                binding=binding,
                ad_experiment_id=experiment_by_segment[
                    binding.segment_id
                ].ad_experiment_id,
            )
            for binding in sorted(
                snapshot_set.bindings,
                key=lambda item: item.segment_id,
            )
        ]
        input_fingerprint = build_input_fingerprint(
            audience_bindings=audience_bindings
        )
        request_fingerprint = build_request_fingerprint(
            promotion_run_id=locked_run.promotion_run_id,
            design_fingerprint=design.fingerprint,
            expires_in_days=request.expires_in_days,
        )

        existing_executions = repository.list_uplift_ready_executions(
            locked_run.promotion_run_id
        )
        for existing in existing_executions:
            existing_design_fingerprint = existing.input_manifest_json.get(
                "experiment_design_fingerprint"
            )
            if existing_design_fingerprint != design.fingerprint:
                raise ExperimentDesignConflictError(
                    "promotion run already uses a different experiment design"
                )
            if existing.input_fingerprint != input_fingerprint:
                raise ExperimentDesignConflictError(
                    "promotion run final audience differs from its existing execution"
                )
        if existing_executions:
            existing = existing_executions[0]
            units = repository.list_units_by_execution(
                existing.segment_assignment_execution_id
            )
            return _build_uplift_ready_response(
                run=locked_run,
                vector_version=snapshot_set.vector_version,
                execution=existing,
                design=design,
                outcome_spec=outcome_spec,
                units=units,
                allocation_results=_manifest_allocation_results(
                    existing.input_manifest_json
                ),
                score_by_user={},
                started_at=started_at,
            )

        raw_members = []
        after_user_id: str | None = None
        page_count = 0
        while True:
            page = self._audience_snapshot_repository.list_run_members(
                promotion_run_id=locked_run.promotion_run_id,
                segment_ids=segment_ids,
                after_user_id=after_user_id,
                limit=ASSIGNMENT_PAGE_SIZE,
            )
            if not page:
                break
            page_count += 1
            raw_members.extend(page)
            after_user_id = page[-1].user_id
            if len(page) < ASSIGNMENT_PAGE_SIZE:
                break

        actual_counts: dict[str, int] = defaultdict(int)
        members: list[ExperimentAudienceMember] = []
        for member in raw_members:
            binding = binding_by_segment.get(member.segment_id)
            experiment = experiment_by_segment.get(member.segment_id)
            if binding is None or experiment is None:
                raise SegmentAssignmentValidationError(
                    "snapshot member references an unknown experiment segment"
                )
            actual_counts[member.segment_id] += 1
            members.append(
                ExperimentAudienceMember(
                    user_id=member.user_id,
                    segment_id=member.segment_id,
                    ad_experiment_id=experiment.ad_experiment_id,
                    audience_snapshot_id=binding.audience_snapshot_id,
                    vector_generation_id=binding.vector_generation_id,
                    behavior_fit_score=member.behavior_fit_score,
                )
            )
        for binding in snapshot_set.bindings:
            if actual_counts[binding.segment_id] != binding.member_count:
                raise SegmentAssignmentAudienceContractError(
                    code="segment_audience_snapshot_binding_invalid",
                    segment_id=binding.segment_id,
                    reason="final snapshot member count changed during assignment",
                )

        experiment_bindings = {
            experiment.ad_experiment_id: (
                experiment.segment_id,
                binding_by_segment[experiment.segment_id].audience_snapshot_id,
            )
            for experiment in experiments.non_fallback
        }
        allocations, allocation_results = allocate_experiment_units(
            project_id=locked_run.project_id,
            promotion_run_id=locked_run.promotion_run_id,
            design=design,
            members=members,
            randomization_salt=self._randomization_salt,
            experiment_bindings=experiment_bindings,
        )

        assigned_at = repository.database_clock()
        for binding in snapshot_set.bindings:
            _validate_binding_time(binding, assigned_at=assigned_at)
        outcome_window_end = assigned_at + timedelta(
            days=design.outcome_window_days
        )
        source_cutoff_at = max(
            cutoff
            for binding in snapshot_set.bindings
            for cutoff in (
                binding.source_cutoff,
                binding.generation_source_revision_cutoff,
            )
        )
        execution_id = build_execution_id(
            locked_run.promotion_run_id,
            request_fingerprint,
        )
        score_by_user = {
            allocation.member.user_id: allocation.member.behavior_fit_score
            for allocation in allocations
            if allocation.arm == "treatment"
        }
        manifest = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "experiment_design": design.as_manifest(),
            "experiment_design_fingerprint": design.fingerprint,
            "outcome_spec": outcome_spec,
            "audience_bindings": audience_bindings,
            "allocation_results": [
                result.as_manifest() for result in allocation_results
            ],
            "assignment_diagnostics": {
                "page_count": page_count,
                "similarity_score_buckets": _score_buckets(score_by_user.values()),
            },
        }
        execution = repository.insert_execution(
            SegmentAssignmentExecutionWrite(
                segment_assignment_execution_id=execution_id,
                promotion_run_id=locked_run.promotion_run_id,
                request_fingerprint=request_fingerprint,
                input_fingerprint=input_fingerprint,
                matcher_strategy="analysis_snapshot_complete_randomization",
                matcher_version="uplift-ready-assignment.v1",
                vector_version=snapshot_set.vector_version,
                source_cutoff_at=source_cutoff_at,
                input_manifest_json=manifest,
            )
        )
        unit_writes = [
            _unit_write(
                run=locked_run,
                allocation=allocation,
                execution_id=execution_id,
                assigned_at=assigned_at,
                outcome_window_end=outcome_window_end,
            )
            for allocation in allocations
        ]
        repository.insert_units(unit_writes)

        expires_at = (
            assigned_at + timedelta(days=request.expires_in_days)
            if request.expires_in_days is not None
            else None
        )
        treatment_writes = [
            _serving_assignment_write(
                run=locked_run,
                experiments=experiment_by_segment,
                allocation=allocation,
                execution_id=execution_id,
                assigned_at=assigned_at,
                expires_at=expires_at,
            )
            for allocation in allocations
            if allocation.arm == "treatment"
        ]
        self._audience_snapshot_repository.consume_run_members(
            promotion_run_id=locked_run.promotion_run_id,
            segment_ids=segment_ids,
        )
        inserted = self._user_segment_assignment_repository.insert_many(
            treatment_writes
        )
        if len(inserted) != len(treatment_writes):
            raise ExperimentDesignConflictError(
                "serving assignments already exist for the promotion run"
            )
        repository.finalize_execution(execution_id)

        units = [
            _unit_record_from_write(unit)
            for unit in unit_writes
        ]
        return _build_uplift_ready_response(
            run=locked_run,
            vector_version=snapshot_set.vector_version,
            execution=execution,
            design=design,
            outcome_spec=outcome_spec,
            units=units,
            allocation_results=allocation_results,
            score_by_user=score_by_user,
            started_at=started_at,
        )

    def _build_legacy_snapshot_assignments(
        self,
        *,
        run: Any,
        experiments: _ExperimentSet,
        request: SegmentAssignmentBuildRequest,
        started_at: int,
    ) -> SegmentAssignmentBuildResponse:
        if request.user_ids or request.eligible_user_limit is not None:
            raise SegmentAssignmentValidationError(
                "analysis snapshot assignment does not accept user_ids or a limit"
            )
        segment_ids = [
            experiment.segment_id for experiment in experiments.non_fallback
        ]
        try:
            snapshot_set = self._audience_snapshot_repository.require_run_binding_set(
                promotion_run_id=run.promotion_run_id,
                segment_ids=segment_ids,
            )
            self._audience_snapshot_repository.consume_run_members(
                promotion_run_id=run.promotion_run_id,
                segment_ids=segment_ids,
            )
        except (pg_errors.UndefinedTable, pg_errors.UndefinedColumn) as exc:
            raise SegmentAssignmentAudienceContractError(
                code="segment_audience_exclusion_contract_missing",
                segment_id=",".join(segment_ids),
                reason="V2 run-target binding Data Contract is missing",
            ) from exc
        except AudienceSnapshotContractError as exc:
            raise SegmentAssignmentAudienceContractError(
                code="segment_audience_snapshot_binding_invalid",
                segment_id=",".join(segment_ids),
                reason=str(exc),
            ) from exc

        experiment_by_segment = {
            experiment.segment_id: experiment
            for experiment in experiments.non_fallback
        }
        assigned_at = datetime.now(UTC)
        expires_at = (
            assigned_at + timedelta(days=request.expires_in_days)
            if request.expires_in_days is not None
            else None
        )
        page_count = 0
        processed_count = 0
        assignment_count = 0
        conflict_count = 0
        skipped_existing_count = 0
        segment_counts: dict[str, int] = {}
        score_buckets = {bucket: 0 for bucket in SIMILARITY_BUCKET_KEYS}
        after_user_id: str | None = None

        while True:
            members = self._audience_snapshot_repository.list_run_members(
                promotion_run_id=run.promotion_run_id,
                segment_ids=segment_ids,
                after_user_id=after_user_id,
                limit=ASSIGNMENT_PAGE_SIZE,
            )
            if not members:
                break
            page_count += 1
            processed_count += len(members)
            existing_user_ids = (
                self._user_segment_assignment_repository.list_existing_user_ids(
                    promotion_run_id=run.promotion_run_id,
                    user_ids=[member.user_id for member in members],
                )
            )
            skipped_existing_count += len(existing_user_ids)
            writes: list[UserSegmentAssignmentWrite] = []
            for member in members:
                if member.user_id in existing_user_ids:
                    continue
                experiment = experiment_by_segment.get(member.segment_id)
                if experiment is None:
                    raise SegmentAssignmentValidationError(
                        "snapshot member references an unknown experiment segment"
                    )
                writes.append(
                    UserSegmentAssignmentWrite(
                        project_id=run.project_id,
                        promotion_run_id=run.promotion_run_id,
                        user_id=member.user_id,
                        segment_id=member.segment_id,
                        ad_experiment_id=experiment.ad_experiment_id,
                        content_id=experiment.content_id,
                        content_option_id=experiment.content_option_id,
                        similarity_score=_score_to_decimal(
                            member.behavior_fit_score
                        ),
                        fallback=False,
                        fallback_reason=None,
                        assignment_source=AssignmentSource.ANALYSIS_SNAPSHOT.value,
                        assigned_at=assigned_at,
                        expires_at=expires_at,
                    )
                )
            inserted = self._user_segment_assignment_repository.insert_many(writes)
            assignment_count += len(inserted)
            conflict_count += len(writes) - len(inserted)
            for record in inserted:
                segment_counts[record.segment_id] = (
                    segment_counts.get(record.segment_id, 0) + 1
                )
                score_buckets[_similarity_score_bucket(record.similarity_score)] += 1
            after_user_id = members[-1].user_id
            if len(members) < ASSIGNMENT_PAGE_SIZE:
                break

        response = SegmentAssignmentBuildResponse(
            promotion_run_id=run.promotion_run_id,
            matching_mode=SNAPSHOT_MATCHING_MODE,
            vector_version=snapshot_set.vector_version,
            ann_candidate_limit=0,
            ann_candidate_count=0,
            exact_reranked_pair_count=0,
            page_count=page_count,
            processed_user_count=processed_count,
            assignment_count=assignment_count,
            insert_conflict_count=conflict_count,
            segment_assignment_counts=segment_counts,
            batch_has_fallback=False,
            fallback_count=0,
            fallback_rate=0.0 if assignment_count else None,
            fallback_reason_counts={reason: 0 for reason in FALLBACK_REASON_KEYS},
            below_threshold_fallback_count=0,
            no_candidate_fallback_count=0,
            invalid_user_vector_fallback_count=0,
            unassigned_count=0,
            unassigned_reason_counts={reason: 0 for reason in FALLBACK_REASON_KEYS},
            below_threshold_unassigned_count=0,
            no_candidate_unassigned_count=0,
            invalid_user_vector_unassigned_count=0,
            similarity_score_buckets=score_buckets,
            ann_underfilled_user_count=0,
            ann_applied=False,
            ann_not_applied_reason="analysis_snapshot_reuse",
            skipped_existing_count=skipped_existing_count,
            insufficient_segment_count=0,
            completion_scope="current_request",
            assignment_mode=ASSIGNMENT_MODE_ANALYSIS_SNAPSHOT,
            input_stability="snapshotted",
            status="completed",
        )
        log.info(
            "analysis_snapshot_assignments_completed",
            {
                "response": response.model_dump(mode="json"),
                "durationMs": duration_ms(started_at),
            },
        )
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
            ann_query_user_count=len(normalized_users),
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

def _requested_experiment_mode(
    request: SegmentAssignmentBuildRequest,
) -> str:
    if request.experiment_design is None:
        return "all_treatment"
    return request.experiment_design.mode.value


def _binding_manifest(
    *,
    binding: AudienceSnapshotBinding,
    ad_experiment_id: str,
) -> dict[str, Any]:
    return {
        "segment_id": binding.segment_id,
        "ad_experiment_id": ad_experiment_id,
        "audience_snapshot_id": binding.audience_snapshot_id,
        "vector_generation_id": binding.vector_generation_id,
        "vector_version": binding.vector_version,
        "member_count": binding.member_count,
        "source_cutoff": binding.source_cutoff.isoformat(),
        "generation_window_end": binding.generation_window_end.isoformat(),
        "generation_source_revision_cutoff": (
            binding.generation_source_revision_cutoff.isoformat()
        ),
    }


def _validate_binding_time(
    binding: AudienceSnapshotBinding,
    *,
    assigned_at: datetime,
) -> None:
    if binding.generation_window_end > assigned_at:
        raise SegmentAssignmentValidationError(
            "feature generation window_end must not follow assigned_at"
        )
    if binding.generation_source_revision_cutoff > assigned_at:
        raise SegmentAssignmentValidationError(
            "feature generation source_revision_cutoff must not follow assigned_at"
        )
    if binding.source_cutoff > assigned_at:
        raise SegmentAssignmentValidationError(
            "audience snapshot source_cutoff must not follow assigned_at"
        )


def _unit_write(
    *,
    run: PromotionRunRecord,
    allocation: ExperimentUnitAllocation,
    execution_id: str,
    assigned_at: datetime,
    outcome_window_end: datetime,
) -> AdExperimentUnitWrite:
    member = allocation.member
    return AdExperimentUnitWrite(
        experiment_unit_id=build_experiment_unit_id(
            promotion_run_id=run.promotion_run_id,
            user_id=member.user_id,
        ),
        project_id=run.project_id,
        promotion_run_id=run.promotion_run_id,
        ad_experiment_id=member.ad_experiment_id,
        segment_id=member.segment_id,
        audience_snapshot_id=member.audience_snapshot_id,
        vector_generation_id=member.vector_generation_id,
        segment_assignment_execution_id=execution_id,
        user_id=member.user_id,
        arm=allocation.arm,
        treatment_probability=allocation.treatment_probability,
        assigned_at=assigned_at,
        outcome_window_start=assigned_at,
        outcome_window_end=outcome_window_end,
    )


def _unit_record_from_write(unit: AdExperimentUnitWrite) -> AdExperimentUnitRecord:
    return AdExperimentUnitRecord(
        experiment_unit_id=unit.experiment_unit_id,
        project_id=unit.project_id,
        promotion_run_id=unit.promotion_run_id,
        ad_experiment_id=unit.ad_experiment_id,
        segment_id=unit.segment_id,
        audience_snapshot_id=unit.audience_snapshot_id,
        vector_generation_id=unit.vector_generation_id,
        segment_assignment_execution_id=unit.segment_assignment_execution_id,
        user_id=unit.user_id,
        arm=unit.arm,
        treatment_probability=unit.treatment_probability,
        assigned_at=unit.assigned_at,
        outcome_window_start=unit.outcome_window_start,
        outcome_window_end=unit.outcome_window_end,
    )


def _serving_assignment_write(
    *,
    run: PromotionRunRecord,
    experiments: Mapping[str, AdExperimentRecord],
    allocation: ExperimentUnitAllocation,
    execution_id: str,
    assigned_at: datetime,
    expires_at: datetime | None,
) -> UserSegmentAssignmentWrite:
    member = allocation.member
    experiment = experiments[member.segment_id]
    return UserSegmentAssignmentWrite(
        project_id=run.project_id,
        promotion_run_id=run.promotion_run_id,
        user_id=member.user_id,
        segment_id=member.segment_id,
        ad_experiment_id=member.ad_experiment_id,
        content_id=experiment.content_id,
        content_option_id=experiment.content_option_id,
        similarity_score=_score_to_decimal(member.behavior_fit_score),
        fallback=False,
        fallback_reason=None,
        assignment_source=AssignmentSource.ANALYSIS_SNAPSHOT.value,
        assigned_at=assigned_at,
        expires_at=expires_at,
        segment_assignment_execution_id=execution_id,
    )


def _manifest_allocation_results(
    manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw_results = manifest.get("allocation_results")
    if not isinstance(raw_results, list) or any(
        not isinstance(result, Mapping) for result in raw_results
    ):
        raise ExperimentDesignConflictError(
            "stored assignment execution allocation results are invalid"
        )
    return [dict(result) for result in raw_results]


def _score_buckets(scores: Sequence[Decimal | None] | Any) -> dict[str, int]:
    buckets = {bucket: 0 for bucket in SIMILARITY_BUCKET_KEYS}
    for score in scores:
        buckets[_similarity_score_bucket(score)] += 1
    return buckets


def _build_uplift_ready_response(
    *,
    run: PromotionRunRecord,
    vector_version: str,
    execution: SegmentAssignmentExecutionRecord,
    design: ExperimentDesign,
    outcome_spec: Mapping[str, Any],
    units: Sequence[AdExperimentUnitRecord],
    allocation_results: Sequence[ExperimentAllocationResult | Mapping[str, Any]],
    score_by_user: Mapping[str, Decimal | None],
    started_at: int,
) -> SegmentAssignmentBuildResponse:
    result_payloads = [
        result.as_manifest()
        if isinstance(result, ExperimentAllocationResult)
        else dict(result)
        for result in allocation_results
    ]
    treatment_units = [unit for unit in units if unit.arm == "treatment"]
    segment_counts: dict[str, int] = defaultdict(int)
    for unit in treatment_units:
        segment_counts[unit.segment_id] += 1
    unit_count = len(units)
    treatment_count = len(treatment_units)
    actual_treatment_ratio = treatment_count / unit_count if unit_count else 0.0
    diagnostics = execution.input_manifest_json.get("assignment_diagnostics")
    stored_buckets = (
        diagnostics.get("similarity_score_buckets")
        if isinstance(diagnostics, Mapping)
        else None
    )
    score_buckets = (
        _score_buckets(list(score_by_user.values()))
        if score_by_user
        else {
            key: int(stored_buckets.get(key, 0))
            for key in SIMILARITY_BUCKET_KEYS
        }
        if isinstance(stored_buckets, Mapping)
        else {key: 0 for key in SIMILARITY_BUCKET_KEYS}
    )
    response = SegmentAssignmentBuildResponse(
        promotion_run_id=run.promotion_run_id,
        matching_mode=SNAPSHOT_MATCHING_MODE,
        vector_version=vector_version,
        ann_candidate_limit=0,
        ann_candidate_count=0,
        exact_reranked_pair_count=0,
        page_count=(unit_count + ASSIGNMENT_PAGE_SIZE - 1) // ASSIGNMENT_PAGE_SIZE,
        processed_user_count=unit_count,
        assignment_count=treatment_count,
        insert_conflict_count=0,
        segment_assignment_counts=dict(segment_counts),
        batch_has_fallback=False,
        fallback_count=0,
        fallback_rate=0.0 if treatment_count else None,
        fallback_reason_counts={reason: 0 for reason in FALLBACK_REASON_KEYS},
        below_threshold_fallback_count=0,
        no_candidate_fallback_count=0,
        invalid_user_vector_fallback_count=0,
        unassigned_count=0,
        unassigned_reason_counts={reason: 0 for reason in FALLBACK_REASON_KEYS},
        below_threshold_unassigned_count=0,
        no_candidate_unassigned_count=0,
        invalid_user_vector_unassigned_count=0,
        similarity_score_buckets=score_buckets,
        ann_underfilled_user_count=0,
        ann_applied=False,
        ann_not_applied_reason="analysis_snapshot_reuse",
        skipped_existing_count=0,
        insufficient_segment_count=0,
        completion_scope="current_request",
        assignment_mode=ASSIGNMENT_MODE_ANALYSIS_SNAPSHOT,
        input_stability="snapshotted",
        status="completed",
        segment_assignment_execution_id=(
            execution.segment_assignment_execution_id
        ),
        request_fingerprint=execution.request_fingerprint,
        input_fingerprint=execution.input_fingerprint,
        experiment_design_fingerprint=design.fingerprint,
        experiment_design={
            "mode": design.mode,
            "requested_treatment_ratio": float(
                design.requested_treatment_ratio
            ),
            "actual_treatment_ratio": actual_treatment_ratio,
            "outcome_window_days": design.outcome_window_days,
            "randomization_version": design.randomization_version,
            "quota_policy_version": design.quota_policy_version,
        },
        outcome_spec=dict(outcome_spec),
        allocation_results=result_payloads,
    )
    log.info(
        "uplift_ready_snapshot_assignments_completed",
        {
            "response": response.model_dump(mode="json"),
            "durationMs": duration_ms(started_at),
        },
    )
    return response


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


def _score_to_decimal(score: float | Decimal | None) -> Decimal | None:
    if score is None:
        return None
    clamped_score = min(1.0, max(0.0, float(score)))
    return Decimal(str(clamped_score)).quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_UP,
    )


def _parse_evaluation_cutoff(result_json: Mapping[str, Any]) -> datetime:
    raw_cutoff = result_json.get("evaluation_cutoff_at")
    if not isinstance(raw_cutoff, str) or not raw_cutoff.strip():
        raise SegmentAssignmentValidationError(
            "next-loop source evaluation cutoff is missing"
        )
    normalized = raw_cutoff.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        cutoff = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SegmentAssignmentValidationError(
            "next-loop source evaluation cutoff is invalid"
        ) from exc
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise SegmentAssignmentValidationError(
            "next-loop source evaluation cutoff must include a timezone"
        )
    return cutoff.astimezone(UTC)


def _similarity_score_bucket(score: Decimal | None) -> str:
    if score is None:
        return "not_available"
    if score < Decimal("0.00"):
        return "lt_0_00"
    if score < Decimal("0.50"):
        return "0_00_to_0_50"
    if score < Decimal("0.65"):
        return "0_50_to_0_65"
    if score < Decimal("0.80"):
        return "0_65_to_0_80"
    if score < Decimal("0.90"):
        return "0_80_to_0_90"
    return "gte_0_90"
