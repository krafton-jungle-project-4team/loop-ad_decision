from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist
from typing import Mapping, Protocol, Sequence

from app.audience_contract import CUSTOM_STRUCTURED_TEMPLATE_ID
from app.analysis.behavior_vector_schema import CandidateBehaviorSpec


EXACT_USER_LIMIT = 50_000
TRANSITION_USER_LIMIT = 500_000
MIN_ANN_CANDIDATES = 10_000
ANN_K_SAFETY_FACTOR = 1.5
MAX_ANN_CORPUS_FRACTION = 0.25
TARGET_THRESHOLD_RECALL = 0.95
RECALL_CONFIDENCE = 0.95
MIN_AUDIT_SAMPLE = 10_000
MAX_AUDIT_SAMPLE = 50_000


class AudienceSearchMethod(StrEnum):
    EXACT = "exact"
    TRANSITION = "transition"
    ANN = "ann"
    EXACT_FALLBACK = "exact_fallback"


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    exact_user_limit: int = EXACT_USER_LIMIT
    transition_user_limit: int = TRANSITION_USER_LIMIT
    min_ann_candidates: int = MIN_ANN_CANDIDATES
    ann_k_safety_factor: float = ANN_K_SAFETY_FACTOR
    max_ann_corpus_fraction: float = MAX_ANN_CORPUS_FRACTION
    target_threshold_recall: float = TARGET_THRESHOLD_RECALL
    recall_confidence: float = RECALL_CONFIDENCE
    min_audit_sample: int = MIN_AUDIT_SAMPLE
    max_audit_sample: int = MAX_AUDIT_SAMPLE
    version: str = "audience_search.v2"


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    user_id: str
    behavior_fit_score: float
    retrieval_rank: int


@dataclass(frozen=True, slots=True)
class RecallAudit:
    retrieved_positive_count: int
    audited_nonretrieved_count: int
    audited_missed_positive_count: int
    estimated_recall: float
    recall_lower_bound: float
    confidence: float
    target_recall: float

    @property
    def passed(self) -> bool:
        return self.recall_lower_bound >= self.target_recall


@dataclass(frozen=True, slots=True)
class AudienceSearchResult:
    method: AudienceSearchMethod
    members: tuple[SearchCandidate, ...]
    corpus_user_count: int
    hard_match_user_count: int
    requested_k: int
    recall_audit: RecallAudit | None
    policy_version: str
    materialized_member_count: int | None = None
    members_relation: str | None = None

    @property
    def final_user_count(self) -> int:
        if self.materialized_member_count is not None:
            return self.materialized_member_count
        return len(self.members)


class AudienceVectorSearchRepository(Protocol):
    def exact_search(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str,
        query_vector: Sequence[float],
        score_threshold: float,
        apply_score_threshold: bool,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
    ) -> list[SearchCandidate]:
        ...

    def ann_search(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[SearchCandidate]:
        ...

    def exact_filter_candidates(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str,
        query_vector: Sequence[float],
        score_threshold: float,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        user_ids: Sequence[str],
    ) -> list[SearchCandidate]:
        ...

    def audit_nonretrieved(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        vector_version: str,
        source_cutoff: str,
        query_vector: Sequence[float],
        score_threshold: float,
        hard_predicate_keys: Sequence[str],
        predicate_parameters: Mapping[str, Sequence[str] | Sequence[int]],
        excluded_user_ids: Sequence[str],
        sample_size: int,
    ) -> tuple[int, int]:
        """Return (audited_count, missed_positive_count)."""
        ...


class CandidateAudienceSearchService:
    def __init__(
        self,
        repository: AudienceVectorSearchRepository,
        *,
        policy: SearchPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._policy = policy or SearchPolicy()

    def search(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        source_cutoff: str,
        spec: CandidateBehaviorSpec,
        corpus_user_count: int,
        hard_match_user_count: int,
        estimated_score_pass_rate: float,
    ) -> AudienceSearchResult:
        if corpus_user_count < 0 or hard_match_user_count < 0:
            raise ValueError("audience counts must not be negative")
        if hard_match_user_count > corpus_user_count:
            raise ValueError("hard match count must not exceed corpus count")
        if not 0.0 <= estimated_score_pass_rate <= 1.0:
            raise ValueError("estimated score pass rate must be between 0 and 1")

        if spec.template_id == CUSTOM_STRUCTURED_TEMPLATE_ID:
            return self._exact(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                source_cutoff=source_cutoff,
                spec=spec,
                corpus_user_count=corpus_user_count,
                hard_match_user_count=hard_match_user_count,
                method=AudienceSearchMethod.EXACT,
            )

        if corpus_user_count <= self._policy.exact_user_limit:
            return self._exact(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                source_cutoff=source_cutoff,
                spec=spec,
                corpus_user_count=corpus_user_count,
                hard_match_user_count=hard_match_user_count,
                method=AudienceSearchMethod.EXACT,
            )

        if corpus_user_count <= self._policy.transition_user_limit:
            exact = self._exact(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                source_cutoff=source_cutoff,
                spec=spec,
                corpus_user_count=corpus_user_count,
                hard_match_user_count=hard_match_user_count,
                method=AudienceSearchMethod.TRANSITION,
            )
            requested_k = min(
                corpus_user_count,
                max(
                    self._policy.min_ann_candidates,
                    math.ceil(
                        self._policy.ann_k_safety_factor
                        * hard_match_user_count
                        * estimated_score_pass_rate
                    ),
                ),
            )
            if exact.members_relation and _supports_materialized_search(
                self._repository
            ):
                self._repository.materialize_ann_candidates(
                    project_id=project_id,
                    vector_generation_id=vector_generation_id,
                    vector_version=spec.vector_version,
                    source_cutoff=source_cutoff,
                    query_vector=spec.query_vector,
                    limit=requested_k,
                )
                self._repository.materialize_ann_members(
                    project_id=project_id,
                    vector_generation_id=vector_generation_id,
                    vector_version=spec.vector_version,
                    source_cutoff=source_cutoff,
                    score_threshold=spec.score_threshold,
                    hard_predicate_keys=spec.hard_predicate_keys,
                    predicate_parameters=spec.predicate_parameters,
                )
                retrieved_count, missed_count = (
                    self._repository.compare_materialized_members(
                        authoritative_relation=exact.members_relation,
                        shadow_relation="audience_ann_members",
                    )
                )
                exact_positive_count = exact.final_user_count
            else:
                ann_candidates = self._repository.ann_search(
                    project_id=project_id,
                    vector_generation_id=vector_generation_id,
                    vector_version=spec.vector_version,
                    source_cutoff=source_cutoff,
                    query_vector=spec.query_vector,
                    limit=requested_k,
                )
                shadow_members = self._repository.exact_filter_candidates(
                    project_id=project_id,
                    vector_generation_id=vector_generation_id,
                    vector_version=spec.vector_version,
                    source_cutoff=source_cutoff,
                    query_vector=spec.query_vector,
                    score_threshold=spec.score_threshold,
                    hard_predicate_keys=spec.hard_predicate_keys,
                    predicate_parameters=spec.predicate_parameters,
                    user_ids=[candidate.user_id for candidate in ann_candidates],
                )
                exact_user_ids = {member.user_id for member in exact.members}
                shadow_user_ids = {member.user_id for member in shadow_members}
                missed_count = len(exact_user_ids - shadow_user_ids)
                retrieved_count = len(shadow_user_ids)
                exact_positive_count = len(exact_user_ids)
            audit = RecallAudit(
                retrieved_positive_count=retrieved_count,
                audited_nonretrieved_count=max(0, exact_positive_count),
                audited_missed_positive_count=missed_count,
                estimated_recall=_safe_recall(
                    retrieved_count,
                    float(missed_count),
                ),
                recall_lower_bound=_safe_recall(
                    retrieved_count,
                    float(missed_count),
                ),
                confidence=1.0,
                target_recall=self._policy.target_threshold_recall,
            )
            return AudienceSearchResult(
                method=AudienceSearchMethod.TRANSITION,
                members=exact.members,
                corpus_user_count=corpus_user_count,
                hard_match_user_count=hard_match_user_count,
                requested_k=requested_k,
                recall_audit=audit,
                policy_version=self._policy.version,
                materialized_member_count=exact.materialized_member_count,
                members_relation=exact.members_relation,
            )

        expected_members = hard_match_user_count * estimated_score_pass_rate
        requested_k = min(
            corpus_user_count,
            max(
                self._policy.min_ann_candidates,
                math.ceil(self._policy.ann_k_safety_factor * expected_members),
            ),
        )
        max_ann_k = max(
            self._policy.min_ann_candidates,
            math.ceil(corpus_user_count * self._policy.max_ann_corpus_fraction),
        )
        requested_k = min(requested_k, corpus_user_count)

        while requested_k <= max_ann_k and requested_k < corpus_user_count:
            if _supports_materialized_search(self._repository):
                retrieved_candidate_count = (
                    self._repository.materialize_ann_candidates(
                        project_id=project_id,
                        vector_generation_id=vector_generation_id,
                        vector_version=spec.vector_version,
                        source_cutoff=source_cutoff,
                        query_vector=spec.query_vector,
                        limit=requested_k,
                    )
                )
                member_count = self._repository.materialize_ann_members(
                    project_id=project_id,
                    vector_generation_id=vector_generation_id,
                    vector_version=spec.vector_version,
                    source_cutoff=source_cutoff,
                    score_threshold=spec.score_threshold,
                    hard_predicate_keys=spec.hard_predicate_keys,
                    predicate_parameters=spec.predicate_parameters,
                )
                audit_sample_size = min(
                    self._policy.max_audit_sample,
                    max(
                        self._policy.min_audit_sample,
                        math.ceil(corpus_user_count * 0.01),
                    ),
                )
                audited_count, missed_count = (
                    self._repository.audit_materialized_nonretrieved(
                        project_id=project_id,
                        vector_generation_id=vector_generation_id,
                        vector_version=spec.vector_version,
                        source_cutoff=source_cutoff,
                        query_vector=spec.query_vector,
                        score_threshold=spec.score_threshold,
                        hard_predicate_keys=spec.hard_predicate_keys,
                        predicate_parameters=spec.predicate_parameters,
                        sample_size=audit_sample_size,
                    )
                )
                audit = threshold_recall_audit(
                    retrieved_positive_count=member_count,
                    audited_nonretrieved_count=audited_count,
                    audited_missed_positive_count=missed_count,
                    nonretrieved_population_count=max(
                        0,
                        corpus_user_count - retrieved_candidate_count,
                    ),
                    target_recall=self._policy.target_threshold_recall,
                    confidence=self._policy.recall_confidence,
                )
                if audit.passed:
                    return AudienceSearchResult(
                        method=AudienceSearchMethod.ANN,
                        members=(),
                        corpus_user_count=corpus_user_count,
                        hard_match_user_count=hard_match_user_count,
                        requested_k=requested_k,
                        recall_audit=audit,
                        policy_version=self._policy.version,
                        materialized_member_count=member_count,
                        members_relation="audience_ann_members",
                    )
                next_k = min(corpus_user_count, requested_k * 2)
                if next_k == requested_k or next_k > max_ann_k:
                    break
                requested_k = next_k
                continue
            ann_candidates = self._repository.ann_search(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                vector_version=spec.vector_version,
                source_cutoff=source_cutoff,
                query_vector=spec.query_vector,
                limit=requested_k,
            )
            members = self._repository.exact_filter_candidates(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                vector_version=spec.vector_version,
                source_cutoff=source_cutoff,
                query_vector=spec.query_vector,
                score_threshold=spec.score_threshold,
                hard_predicate_keys=spec.hard_predicate_keys,
                predicate_parameters=spec.predicate_parameters,
                user_ids=[candidate.user_id for candidate in ann_candidates],
            )
            audit_sample_size = min(
                self._policy.max_audit_sample,
                max(
                    self._policy.min_audit_sample,
                    math.ceil(corpus_user_count * 0.01),
                ),
            )
            audited_count, missed_count = self._repository.audit_nonretrieved(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                vector_version=spec.vector_version,
                source_cutoff=source_cutoff,
                query_vector=spec.query_vector,
                score_threshold=spec.score_threshold,
                hard_predicate_keys=spec.hard_predicate_keys,
                predicate_parameters=spec.predicate_parameters,
                excluded_user_ids=[candidate.user_id for candidate in ann_candidates],
                sample_size=audit_sample_size,
            )
            audit = threshold_recall_audit(
                retrieved_positive_count=len(members),
                audited_nonretrieved_count=audited_count,
                audited_missed_positive_count=missed_count,
                nonretrieved_population_count=max(0, corpus_user_count - len(ann_candidates)),
                target_recall=self._policy.target_threshold_recall,
                confidence=self._policy.recall_confidence,
            )
            if audit.passed:
                return AudienceSearchResult(
                    method=AudienceSearchMethod.ANN,
                    members=tuple(_dedupe_members(members)),
                    corpus_user_count=corpus_user_count,
                    hard_match_user_count=hard_match_user_count,
                    requested_k=requested_k,
                    recall_audit=audit,
                    policy_version=self._policy.version,
                )
            next_k = min(corpus_user_count, requested_k * 2)
            if next_k == requested_k or next_k > max_ann_k:
                break
            requested_k = next_k

        return self._exact(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            source_cutoff=source_cutoff,
            spec=spec,
            corpus_user_count=corpus_user_count,
            hard_match_user_count=hard_match_user_count,
            method=AudienceSearchMethod.EXACT_FALLBACK,
        )

    def _exact(
        self,
        *,
        project_id: str,
        vector_generation_id: str,
        source_cutoff: str,
        spec: CandidateBehaviorSpec,
        corpus_user_count: int,
        hard_match_user_count: int,
        method: AudienceSearchMethod,
    ) -> AudienceSearchResult:
        apply_score_threshold = spec.template_id != CUSTOM_STRUCTURED_TEMPLATE_ID
        materialize_exact = getattr(
            self._repository,
            "materialize_exact_members",
            None,
        )
        if callable(materialize_exact):
            member_count = materialize_exact(
                project_id=project_id,
                vector_generation_id=vector_generation_id,
                vector_version=spec.vector_version,
                source_cutoff=source_cutoff,
                query_vector=spec.query_vector,
                score_threshold=spec.score_threshold,
                apply_score_threshold=apply_score_threshold,
                hard_predicate_keys=spec.hard_predicate_keys,
                predicate_parameters=spec.predicate_parameters,
            )
            return AudienceSearchResult(
                method=method,
                members=(),
                corpus_user_count=corpus_user_count,
                hard_match_user_count=hard_match_user_count,
                requested_k=corpus_user_count,
                recall_audit=None,
                policy_version=self._policy.version,
                materialized_member_count=member_count,
                members_relation="audience_exact_members",
            )
        members = self._repository.exact_search(
            project_id=project_id,
            vector_generation_id=vector_generation_id,
            vector_version=spec.vector_version,
            source_cutoff=source_cutoff,
            query_vector=spec.query_vector,
            score_threshold=spec.score_threshold,
            apply_score_threshold=apply_score_threshold,
            hard_predicate_keys=spec.hard_predicate_keys,
            predicate_parameters=spec.predicate_parameters,
        )
        return AudienceSearchResult(
            method=method,
            members=tuple(_dedupe_members(members)),
            corpus_user_count=corpus_user_count,
            hard_match_user_count=hard_match_user_count,
            requested_k=corpus_user_count,
            recall_audit=None,
            policy_version=self._policy.version,
        )


def threshold_recall_audit(
    *,
    retrieved_positive_count: int,
    audited_nonretrieved_count: int,
    audited_missed_positive_count: int,
    nonretrieved_population_count: int,
    target_recall: float = TARGET_THRESHOLD_RECALL,
    confidence: float = RECALL_CONFIDENCE,
) -> RecallAudit:
    if min(
        retrieved_positive_count,
        audited_nonretrieved_count,
        audited_missed_positive_count,
        nonretrieved_population_count,
    ) < 0:
        raise ValueError("recall audit counts must not be negative")
    if audited_missed_positive_count > audited_nonretrieved_count:
        raise ValueError("missed positives must not exceed audited users")

    if audited_nonretrieved_count == 0:
        missed_upper = float(nonretrieved_population_count)
    else:
        upper_rate = _wilson_upper_bound(
            successes=audited_missed_positive_count,
            trials=audited_nonretrieved_count,
            confidence=confidence,
        )
        missed_upper = upper_rate * nonretrieved_population_count
    estimated_missed = (
        (audited_missed_positive_count / audited_nonretrieved_count)
        * nonretrieved_population_count
        if audited_nonretrieved_count
        else nonretrieved_population_count
    )
    estimated_recall = _safe_recall(retrieved_positive_count, estimated_missed)
    recall_lower_bound = _safe_recall(retrieved_positive_count, missed_upper)
    return RecallAudit(
        retrieved_positive_count=retrieved_positive_count,
        audited_nonretrieved_count=audited_nonretrieved_count,
        audited_missed_positive_count=audited_missed_positive_count,
        estimated_recall=estimated_recall,
        recall_lower_bound=recall_lower_bound,
        confidence=confidence,
        target_recall=target_recall,
    )


def _wilson_upper_bound(*, successes: int, trials: int, confidence: float) -> float:
    if trials <= 0:
        return 1.0
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = proportion + (z * z / (2.0 * trials))
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) / trials)
        + (z * z / (4.0 * trials * trials))
    )
    return min(1.0, (center + margin) / denominator)


def _safe_recall(retrieved: int, missed: float) -> float:
    denominator = retrieved + missed
    return 1.0 if denominator == 0 else retrieved / denominator


def _dedupe_members(members: Sequence[SearchCandidate]) -> list[SearchCandidate]:
    best_by_user: dict[str, SearchCandidate] = {}
    for member in members:
        current = best_by_user.get(member.user_id)
        if current is None or (
            member.behavior_fit_score,
            -member.retrieval_rank,
        ) > (
            current.behavior_fit_score,
            -current.retrieval_rank,
        ):
            best_by_user[member.user_id] = member
    return sorted(
        best_by_user.values(),
        key=lambda member: (-member.behavior_fit_score, member.user_id),
    )


def _supports_materialized_search(repository: object) -> bool:
    return all(
        callable(getattr(repository, name, None))
        for name in (
            "materialize_exact_members",
            "materialize_ann_candidates",
            "materialize_ann_members",
            "audit_materialized_nonretrieved",
            "compare_materialized_members",
        )
    )
