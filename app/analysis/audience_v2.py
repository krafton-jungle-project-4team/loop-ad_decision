from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from app.audience_contract import (
    SegmentAudienceContractError,
    SegmentAudienceSpec,
    SegmentDefinitionAudienceAdapter,
)
from app.analysis.audience_search import CandidateAudienceSearchService
from app.analysis.audience_search_repository import (
    AudienceSearchContext,
    HardMatchAggregateRequest,
    PgClickHouseAudienceVectorSearchRepository,
    hard_predicates_support_batch_aggregate,
)
from app.analysis.audience_snapshot_repository import (
    AudienceSnapshotRepository,
    AudienceSnapshotWrite,
)
from app.analysis.behavior_vector_schema import (
    CandidateBehaviorSpec,
    HotelBookingBehaviorSchemaV2,
)
from app.analysis.repositories import PromotionRecord, SegmentDefinitionRecord
from app.analysis.semantic_selection import (
    BUNDLED_SEMANTIC_SELECTION_PATH,
    BUNDLED_SEMANTIC_SELECTION_SHA256,
    BundledSemanticSelectionProvider,
    SemanticSelectionArtifact,
    load_bundled_semantic_selection,
    load_semantic_selection_artifact,
)
from app.analysis.vector_service import (
    SegmentVectorBuildRequest,
    SegmentVectorBuildResult,
)


BUNDLED_AUDIENCE_CALIBRATION_PATH = BUNDLED_SEMANTIC_SELECTION_PATH
BUNDLED_AUDIENCE_CALIBRATION_SHA256 = BUNDLED_SEMANTIC_SELECTION_SHA256


class SegmentVectorPreparer(Protocol):
    def prepare_segment_vector(
        self,
        request: SegmentVectorBuildRequest,
    ) -> SegmentVectorBuildResult:
        ...


@dataclass(frozen=True, slots=True)
class AudienceV2Preparation:
    audience_snapshot_id: str
    segment_vector_id: str
    vector_generation_id: str
    vector_version: str
    total_eligible_user_count: int
    matching_user_count: int
    selected_user_count: int
    selection_method: str
    estimated_recall: float
    recall_lower_bound: float
    recall_target: float
    meets_min_sample_size: bool
    source_audience_snapshot_id: str | None = None
    allocation_plan_id: str | None = None
    promotion_exclusion_revision: int | None = None
    excluded_user_count: int = 0


@dataclass(frozen=True, slots=True)
class _CompiledAudience:
    segment: SegmentDefinitionRecord
    spec: CandidateBehaviorSpec
    observation_window_days: int
    context: AudienceSearchContext
    segment_vector_id: str


CandidateCalibrationRegistry = SemanticSelectionArtifact
BundledCandidateCalibrationProvider = BundledSemanticSelectionProvider


class AudienceV2Coordinator:
    def __init__(
        self,
        *,
        search_repository: PgClickHouseAudienceVectorSearchRepository,
        snapshot_repository: AudienceSnapshotRepository,
        segment_vector_service: SegmentVectorPreparer,
        calibration_provider: BundledCandidateCalibrationProvider | None = None,
        schema: HotelBookingBehaviorSchemaV2 | None = None,
        audience_adapter: SegmentDefinitionAudienceAdapter | None = None,
    ) -> None:
        self._search_repository = search_repository
        self._search_service = CandidateAudienceSearchService(search_repository)
        self._snapshot_repository = snapshot_repository
        self._segment_vector_service = segment_vector_service
        self._calibration_provider = (
            calibration_provider or BundledCandidateCalibrationProvider()
        )
        self._schema = schema or HotelBookingBehaviorSchemaV2()
        self._audience_adapter = audience_adapter or SegmentDefinitionAudienceAdapter()

    def prepare(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        segment: SegmentDefinitionRecord,
        audience_snapshot_id: str | None = None,
    ) -> AudienceV2Preparation:
        if audience_snapshot_id is None:
            return self.prepare_many(
                analysis_id=analysis_id,
                promotion=promotion,
                segments=(segment,),
            )[segment.segment_id]

        _audience_spec, spec = self._compile_segment(segment)
        bound = self._snapshot_repository.require_binding(
            snapshot_id=audience_snapshot_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_id=segment.segment_id,
            spec=spec,
        )
        return AudienceV2Preparation(
            audience_snapshot_id=bound.snapshot_id,
            segment_vector_id=bound.segment_vector_id,
            vector_generation_id=bound.vector_generation_id,
            vector_version=spec.vector_version,
            total_eligible_user_count=bound.eligible_user_count,
            matching_user_count=bound.behavior_match_count,
            selected_user_count=bound.final_user_count,
            selection_method=bound.selection_method,
            estimated_recall=bound.estimated_recall,
            recall_lower_bound=bound.recall_lower_bound,
            recall_target=bound.recall_target,
            meets_min_sample_size=bound.meets_min_sample_size,
            promotion_exclusion_revision=bound.promotion_exclusion_revision,
            excluded_user_count=bound.excluded_user_count,
        )

    def prepare_many(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        segments: Sequence[SegmentDefinitionRecord],
    ) -> Mapping[str, AudienceV2Preparation]:
        if not segments:
            return {}
        contexts: dict[str, AudienceSearchContext] = {}
        compiled: list[_CompiledAudience] = []
        for segment in segments:
            audience_spec, spec = self._compile_segment(segment)
            context = contexts.get(spec.vector_version)
            if context is None:
                context = self._active_context(
                    project_id=promotion.project_id,
                    campaign_id=promotion.campaign_id,
                    promotion_id=promotion.promotion_id,
                    segment_id=segment.segment_id,
                    vector_version=spec.vector_version,
                )
                contexts[spec.vector_version] = context
            self._validate_context(
                segment_id=segment.segment_id,
                spec=spec,
                context=context,
                observation_window_days=audience_spec.observation_window_days,
            )
            vector_result = self._segment_vector_service.prepare_segment_vector(
                SegmentVectorBuildRequest(
                    project_id=promotion.project_id,
                    promotion_id=promotion.promotion_id,
                    analysis_id=analysis_id,
                    segment_id=segment.segment_id,
                    vector_version=spec.vector_version,
                    query_vector=spec.query_vector,
                )
            )
            compiled.append(
                _CompiledAudience(
                    segment=segment,
                    spec=spec,
                    observation_window_days=audience_spec.observation_window_days,
                    context=context,
                    segment_vector_id=vector_result.segment_vector_id,
                )
            )

        hard_match_counts = self._count_hard_matches_by_scope(
            project_id=promotion.project_id,
            compiled=compiled,
        )
        return {
            item.segment.segment_id: self._prepare_compiled(
                analysis_id=analysis_id,
                promotion=promotion,
                item=item,
                hard_match_count=hard_match_counts[item.segment.segment_id],
            )
            for item in compiled
        }

    def _compile_segment(
        self,
        segment: SegmentDefinitionRecord,
    ) -> tuple[SegmentAudienceSpec, CandidateBehaviorSpec]:
        resolution = self._audience_adapter.resolve(
            segment_id=segment.segment_id,
            rule_json=segment.rule_json,
        )
        if not resolution.is_v2 or resolution.spec is None:
            raise SegmentAudienceContractError(
                code="segment_audience_contract_unsupported",
                segment_id=segment.segment_id,
                reason="AudienceV2Coordinator requires segment_audience.v1",
            )
        audience_spec = resolution.spec
        if audience_spec.is_custom_structured:
            try:
                return (
                    audience_spec,
                    self._schema.compile_custom_segment_audience(
                        spec=audience_spec,
                    ),
                )
            except ValueError as exc:
                raise SegmentAudienceContractError(
                    code="segment_audience_manifest_mismatch",
                    segment_id=segment.segment_id,
                    reason=str(exc),
                ) from exc
        calibration = self._calibration_provider.require(
            segment_id=segment.segment_id,
            spec=audience_spec,
            schema=self._schema,
        )
        try:
            compiled = self._schema.compile_segment_audience(
                spec=audience_spec,
                calibration=calibration,
            )
        except ValueError as exc:
            raise SegmentAudienceContractError(
                code="segment_audience_manifest_mismatch",
                segment_id=segment.segment_id,
                reason=str(exc),
            ) from exc
        return audience_spec, compiled

    def _active_context(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        segment_id: str,
        vector_version: str,
    ) -> AudienceSearchContext:
        try:
            return self._search_repository.get_context(
                project_id=project_id,
                campaign_id=campaign_id,
                promotion_id=promotion_id,
                vector_version=vector_version,
            )
        except RuntimeError as exc:
            raise SegmentAudienceContractError(
                code="segment_audience_active_generation_missing",
                segment_id=segment_id,
                reason=str(exc),
            ) from exc

    @staticmethod
    def _validate_context(
        *,
        segment_id: str,
        spec: CandidateBehaviorSpec,
        context: AudienceSearchContext,
        observation_window_days: int,
    ) -> None:
        if context.manifest_hash != spec.manifest_hash:
            raise SegmentAudienceContractError(
                code="segment_audience_generation_incompatible",
                segment_id=segment_id,
                reason="active vector generation manifest hash is incompatible",
            )
        window_days = int(
            (context.source_cutoff - context.window_start).total_seconds() / 86400
        )
        if window_days != observation_window_days:
            raise SegmentAudienceContractError(
                code="segment_audience_generation_incompatible",
                segment_id=segment_id,
                reason=(
                    "active vector generation window does not match "
                    "observation_window_days"
                ),
            )

    def _count_hard_matches_by_scope(
        self,
        *,
        project_id: str,
        compiled: Sequence[_CompiledAudience],
    ) -> Mapping[str, int]:
        grouped: dict[tuple[Any, ...], list[_CompiledAudience]] = {}
        for item in compiled:
            context = item.context
            key = (
                context.vector_generation_id,
                item.spec.vector_version,
                context.window_start,
                context.source_cutoff,
                context.manifest_hash,
            )
            grouped.setdefault(key, []).append(item)

        counts: dict[str, int] = {}
        for items in grouped.values():
            batchable = [
                item
                for item in items
                if hard_predicates_support_batch_aggregate(
                    item.spec.hard_predicate_keys
                )
            ]
            if batchable:
                context = batchable[0].context
                counts.update(
                    self._search_repository.count_hard_matches_batch(
                        project_id=project_id,
                        vector_version=batchable[0].spec.vector_version,
                        source_revision_cutoff=context.source_revision_cutoff,
                        window_start=context.window_start,
                        window_end=context.source_cutoff,
                        requests=tuple(
                            HardMatchAggregateRequest(
                                segment_id=item.segment.segment_id,
                                hard_predicate_keys=item.spec.hard_predicate_keys,
                                predicate_parameters=item.spec.predicate_parameters,
                            )
                            for item in batchable
                        ),
                        exclusion_context=context.exclusion_context,
                    )
                )
            for item in items:
                if item in batchable:
                    continue
                context = item.context
                counts[item.segment.segment_id] = (
                    self._search_repository.count_hard_matches(
                        project_id=project_id,
                        vector_version=item.spec.vector_version,
                        source_revision_cutoff=context.source_revision_cutoff,
                        window_start=context.window_start,
                        window_end=context.source_cutoff,
                        hard_predicate_keys=item.spec.hard_predicate_keys,
                        predicate_parameters=item.spec.predicate_parameters,
                        exclusion_context=context.exclusion_context,
                    )
                )
        return counts

    def _prepare_compiled(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        item: _CompiledAudience,
        hard_match_count: int,
    ) -> AudienceV2Preparation:
        segment = item.segment
        spec = item.spec
        context = item.context
        estimated_score_pass_rate = (
            self._search_repository.estimate_score_pass_rate(
                project_id=promotion.project_id,
                vector_generation_id=context.vector_generation_id,
                vector_version=spec.vector_version,
                source_revision_cutoff=context.source_revision_cutoff,
                source_cutoff=context.source_cutoff,
                window_start=context.window_start,
                query_vector=spec.query_vector,
                score_threshold=spec.score_threshold,
                hard_predicate_keys=spec.hard_predicate_keys,
                predicate_parameters=spec.predicate_parameters,
                exclusion_context=context.exclusion_context,
            )
        )
        search_result = self._search_service.search(
            project_id=promotion.project_id,
            vector_generation_id=context.vector_generation_id,
            source_cutoff=context.source_cutoff,
            spec=spec,
            corpus_user_count=context.corpus_user_count,
            hard_match_user_count=hard_match_count,
            estimated_score_pass_rate=estimated_score_pass_rate,
        )
        snapshot_id = self._snapshot_repository.save_completed(
            AudienceSnapshotWrite(
                analysis_id=analysis_id,
                project_id=promotion.project_id,
                campaign_id=promotion.campaign_id,
                promotion_id=promotion.promotion_id,
                segment_id=segment.segment_id,
                segment_vector_id=item.segment_vector_id,
                vector_generation_id=context.vector_generation_id,
                source_cutoff=context.source_cutoff,
                window_start=context.window_start,
                window_end=context.source_cutoff,
                spec=spec,
                search_result=search_result,
                min_sample_size=promotion.min_sample_size,
                exclusion_context=context.exclusion_context,
            )
        )
        audit = search_result.recall_audit
        final_user_count = search_result.final_user_count
        return AudienceV2Preparation(
            audience_snapshot_id=snapshot_id,
            segment_vector_id=item.segment_vector_id,
            vector_generation_id=context.vector_generation_id,
            vector_version=spec.vector_version,
            total_eligible_user_count=context.corpus_user_count,
            matching_user_count=hard_match_count,
            selected_user_count=final_user_count,
            selection_method=search_result.method.value,
            estimated_recall=audit.estimated_recall if audit else 1.0,
            recall_lower_bound=audit.recall_lower_bound if audit else 1.0,
            recall_target=audit.target_recall if audit else 1.0,
            meets_min_sample_size=(
                final_user_count > 0
                and final_user_count >= promotion.min_sample_size
            ),
            promotion_exclusion_revision=(
                context.exclusion_context.revision
                if context.exclusion_context is not None
                else 0
            ),
            excluded_user_count=(
                context.exclusion_context.excluded_user_count
                if context.exclusion_context is not None
                else 0
            ),
        )


def load_bundled_candidate_calibrations(
    *,
    segment_id: str,
) -> CandidateCalibrationRegistry:
    return load_bundled_semantic_selection(
        segment_id=segment_id,
    )


def _load_candidate_calibrations(
    *,
    path: Any,
    expected_sha256: str,
    segment_id: str,
) -> CandidateCalibrationRegistry:
    return load_semantic_selection_artifact(
        path=path,
        expected_sha256=expected_sha256,
        segment_id=segment_id,
    )
