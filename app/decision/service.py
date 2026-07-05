from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from decimal import Decimal
from typing import Sequence

from app.decision.repositories import (
    AdExperimentWrite,
    AdExperimentWriter,
    ContentCandidateReader,
    ContentCandidateRecord,
    GenerationRunReader,
    GenerationRunRecord,
    PromotionAnalysisReader,
    PromotionAnalysisRecord,
    PromotionReader,
    PromotionRecord,
    PromotionRunWrite,
    PromotionRunWriter,
    PromotionTargetSegmentReader,
    PromotionTargetSegmentRecord,
)
from app.decision.schemas import (
    AdExperimentCreateResponse,
    AdExperimentStatus,
    PromotionRunStatus,
    RunCreateRequest,
    RunCreateResponse,
)


COMPLETED_STATUS = "completed"
MAX_CONTRACT_ID_LENGTH = 100


class PromotionNotFoundError(Exception):
    pass


class RunValidationError(Exception):
    pass


class RunConflictError(Exception):
    pass


class PromotionRunService:
    def __init__(
        self,
        *,
        promotion_repository: PromotionReader,
        promotion_analysis_repository: PromotionAnalysisReader,
        promotion_target_segment_repository: PromotionTargetSegmentReader,
        generation_run_repository: GenerationRunReader,
        content_candidate_repository: ContentCandidateReader,
        promotion_run_repository: PromotionRunWriter,
        ad_experiment_repository: AdExperimentWriter,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._promotion_analysis_repository = promotion_analysis_repository
        self._promotion_target_segment_repository = promotion_target_segment_repository
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository

    def create_run(
        self,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        promotion = self._get_promotion(promotion_id)
        analysis = self._select_analysis(
            promotion=promotion,
            analysis_id=request.analysis_id,
        )
        generation = self._select_generation(
            promotion=promotion,
            analysis=analysis,
            generation_id=request.generation_id,
        )

        promotion_run_id = build_bounded_decision_id(
            "prun",
            promotion.promotion_id,
            f"loop_{request.loop_count}",
        )
        if self._promotion_run_repository.exists_for_promotion_loop(
            promotion_id=promotion.promotion_id,
            loop_count=request.loop_count,
        ):
            raise RunConflictError(
                "promotion_run already exists for promotion_id and loop_count"
            )

        target_segments = self._load_target_segments(analysis, promotion)
        content_by_segment = self._load_content_by_segment(generation.generation_id)
        selected_content = self._select_content_for_segments(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            target_segments=target_segments,
            content_by_segment=content_by_segment,
        )

        run = self._build_promotion_run(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            promotion_run_id=promotion_run_id,
            loop_count=request.loop_count,
        )
        ad_experiments = self._build_ad_experiments(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            promotion_run_id=promotion_run_id,
            target_segments=target_segments,
            selected_content=selected_content,
            loop_count=request.loop_count,
        )

        for experiment in ad_experiments:
            if self._ad_experiment_repository.exists_for_run_segment(
                promotion_run_id=experiment.promotion_run_id,
                segment_id=experiment.segment_id,
            ):
                raise RunConflictError(
                    "ad_experiment already exists for promotion_run_id and segment_id"
                )

        self._promotion_run_repository.insert(run)
        self._ad_experiment_repository.insert_many(ad_experiments)

        return RunCreateResponse(
            promotion_run_id=run.promotion_run_id,
            project_id=run.project_id,
            campaign_id=run.campaign_id,
            promotion_id=run.promotion_id,
            analysis_id=run.analysis_id,
            generation_id=run.generation_id,
            loop_count=run.loop_count,
            status=PromotionRunStatus(run.status),
            goal_snapshot_json=dict(run.goal_snapshot_json),
            ad_experiments=[
                AdExperimentCreateResponse(
                    ad_experiment_id=experiment.ad_experiment_id,
                    segment_id=experiment.segment_id,
                    segment_name=experiment.segment_name,
                    content_id=experiment.content_id,
                    content_option_id=experiment.content_option_id,
                    channel=experiment.channel,
                    loop_count=experiment.loop_count,
                    status=AdExperimentStatus(experiment.status),
                )
                for experiment in ad_experiments
            ],
        )

    def _get_promotion(self, promotion_id: str) -> PromotionRecord:
        promotion = self._promotion_repository.get_by_id(promotion_id)
        if promotion is None:
            raise PromotionNotFoundError(f"promotion not found: {promotion_id}")
        return promotion

    def _select_analysis(
        self,
        *,
        promotion: PromotionRecord,
        analysis_id: str | None,
    ) -> PromotionAnalysisRecord:
        if analysis_id is None:
            analysis = self._promotion_analysis_repository.get_latest_completed_for_promotion(
                promotion.promotion_id,
            )
            if analysis is None:
                raise RunValidationError(
                    "completed promotion analysis is required before creating a run"
                )
        else:
            analysis = self._promotion_analysis_repository.get_by_id(analysis_id)
            if analysis is None:
                raise RunValidationError(f"promotion analysis not found: {analysis_id}")

        if analysis.status != COMPLETED_STATUS:
            raise RunValidationError("promotion analysis must be completed")
        _validate_project_campaign_promotion(
            label="promotion analysis",
            project_id=analysis.project_id,
            campaign_id=analysis.campaign_id,
            promotion_id=analysis.promotion_id,
            promotion=promotion,
        )
        return analysis

    def _select_generation(
        self,
        *,
        promotion: PromotionRecord,
        analysis: PromotionAnalysisRecord,
        generation_id: str | None,
    ) -> GenerationRunRecord:
        if generation_id is None:
            generation = self._generation_run_repository.get_latest_completed_for_promotion(
                promotion.promotion_id,
            )
            if generation is None:
                raise RunValidationError(
                    "completed generation run is required before creating a run"
                )
        else:
            generation = self._generation_run_repository.get_by_id(generation_id)
            if generation is None:
                raise RunValidationError(f"generation run not found: {generation_id}")

        if generation.status != COMPLETED_STATUS:
            raise RunValidationError("generation run must be completed")
        _validate_project_campaign_promotion(
            label="generation run",
            project_id=generation.project_id,
            campaign_id=generation.campaign_id,
            promotion_id=generation.promotion_id,
            promotion=promotion,
        )
        if generation.analysis_id != analysis.analysis_id:
            raise RunValidationError(
                "generation run must belong to the selected promotion analysis"
            )
        return generation

    def _load_target_segments(
        self,
        analysis: PromotionAnalysisRecord,
        promotion: PromotionRecord,
    ) -> list[PromotionTargetSegmentRecord]:
        target_segments = self._promotion_target_segment_repository.list_for_analysis(
            analysis.analysis_id,
        )
        if not target_segments:
            raise RunValidationError("at least one target segment is required")

        seen_segment_ids: set[str] = set()
        for segment in target_segments:
            _validate_project_campaign_promotion(
                label="target segment",
                project_id=segment.project_id,
                campaign_id=segment.campaign_id,
                promotion_id=segment.promotion_id,
                promotion=promotion,
            )
            if segment.analysis_id != analysis.analysis_id:
                raise RunValidationError(
                    "target segment must belong to the selected promotion analysis"
                )
            if segment.segment_id in seen_segment_ids:
                raise RunValidationError(
                    f"duplicate target segment is not allowed: {segment.segment_id}"
                )
            seen_segment_ids.add(segment.segment_id)
        return target_segments

    def _load_content_by_segment(
        self,
        generation_id: str,
    ) -> dict[str, list[ContentCandidateRecord]]:
        content_candidates = (
            self._content_candidate_repository.list_approved_or_active_for_generation(
                generation_id,
            )
        )
        content_by_segment: dict[str, list[ContentCandidateRecord]] = defaultdict(list)
        for candidate in content_candidates:
            content_by_segment[candidate.segment_id].append(candidate)
        return content_by_segment

    def _select_content_for_segments(
        self,
        *,
        promotion: PromotionRecord,
        analysis: PromotionAnalysisRecord,
        generation: GenerationRunRecord,
        target_segments: Sequence[PromotionTargetSegmentRecord],
        content_by_segment: dict[str, list[ContentCandidateRecord]],
    ) -> dict[str, ContentCandidateRecord]:
        selected_content: dict[str, ContentCandidateRecord] = {}
        for segment in target_segments:
            candidates = content_by_segment.get(segment.segment_id, [])
            if len(candidates) != 1:
                raise RunValidationError(
                    "each target segment must have exactly one approved or active "
                    f"content candidate: {segment.segment_id}"
                )
            candidate = candidates[0]
            _validate_project_campaign_promotion(
                label="content candidate",
                project_id=candidate.project_id,
                campaign_id=candidate.campaign_id,
                promotion_id=candidate.promotion_id,
                promotion=promotion,
            )
            if candidate.analysis_id != analysis.analysis_id:
                raise RunValidationError(
                    "content candidate must belong to the selected promotion analysis"
                )
            if candidate.generation_id != generation.generation_id:
                raise RunValidationError(
                    "content candidate must belong to the selected generation run"
                )
            if candidate.channel != promotion.channel:
                raise RunValidationError("content candidate channel must match promotion")
            selected_content[segment.segment_id] = candidate
        return selected_content

    def _build_promotion_run(
        self,
        *,
        promotion: PromotionRecord,
        analysis: PromotionAnalysisRecord,
        generation: GenerationRunRecord,
        promotion_run_id: str,
        loop_count: int,
    ) -> PromotionRunWrite:
        return PromotionRunWrite(
            promotion_run_id=promotion_run_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            analysis_id=analysis.analysis_id,
            generation_id=generation.generation_id,
            loop_count=loop_count,
            status=PromotionRunStatus.PLANNED.value,
            goal_snapshot_json=_build_goal_snapshot(
                promotion=promotion,
                analysis=analysis,
                generation=generation,
                loop_count=loop_count,
            ),
        )

    def _build_ad_experiments(
        self,
        *,
        promotion: PromotionRecord,
        analysis: PromotionAnalysisRecord,
        generation: GenerationRunRecord,
        promotion_run_id: str,
        target_segments: Sequence[PromotionTargetSegmentRecord],
        selected_content: dict[str, ContentCandidateRecord],
        loop_count: int,
    ) -> list[AdExperimentWrite]:
        experiments: list[AdExperimentWrite] = []
        for segment in target_segments:
            content = selected_content[segment.segment_id]
            experiments.append(
                AdExperimentWrite(
                    ad_experiment_id=build_bounded_decision_id(
                        "adexp",
                        promotion_run_id,
                        segment.segment_id,
                    ),
                    project_id=promotion.project_id,
                    campaign_id=promotion.campaign_id,
                    promotion_id=promotion.promotion_id,
                    promotion_run_id=promotion_run_id,
                    analysis_id=analysis.analysis_id,
                    generation_id=generation.generation_id,
                    segment_id=segment.segment_id,
                    segment_name=segment.segment_name,
                    content_id=content.content_id,
                    content_option_id=content.content_option_id,
                    channel=promotion.channel,
                    loop_count=loop_count,
                    status=AdExperimentStatus.PLANNED.value,
                    goal_metric=promotion.goal_metric,
                    goal_target_value=promotion.goal_target_value,
                    goal_basis=promotion.goal_basis,
                )
            )
        return experiments


def build_bounded_decision_id(prefix: str, *parts: str) -> str:
    seed = "::".join(parts)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", "_".join(parts)).strip("_").lower()
    if not slug:
        slug = "id"

    max_slug_length = MAX_CONTRACT_ID_LENGTH - len(prefix) - len(digest) - 2
    slug = slug[:max_slug_length].rstrip("_") or "id"
    return f"{prefix}_{slug}_{digest}"


def _build_goal_snapshot(
    *,
    promotion: PromotionRecord,
    analysis: PromotionAnalysisRecord,
    generation: GenerationRunRecord,
    loop_count: int,
) -> dict[str, str | int]:
    return {
        "source": "promotions",
        "promotion_id": promotion.promotion_id,
        "channel": promotion.channel,
        "goal_metric": promotion.goal_metric,
        "goal_target_value": _decimal_to_snapshot_string(promotion.goal_target_value),
        "goal_basis": promotion.goal_basis,
        "min_sample_size": promotion.min_sample_size,
        "max_loop_count": promotion.max_loop_count,
        "analysis_id": analysis.analysis_id,
        "generation_id": generation.generation_id,
        "loop_count": loop_count,
    }


def _decimal_to_snapshot_string(value: Decimal) -> str:
    return str(value)


def _validate_project_campaign_promotion(
    *,
    label: str,
    project_id: str,
    campaign_id: str,
    promotion_id: str,
    promotion: PromotionRecord,
) -> None:
    if project_id != promotion.project_id:
        raise RunValidationError(f"{label} project_id must match promotion")
    if campaign_id != promotion.campaign_id:
        raise RunValidationError(f"{label} campaign_id must match promotion")
    if promotion_id != promotion.promotion_id:
        raise RunValidationError(f"{label} promotion_id must match promotion")
