from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal
from typing import Sequence

from app.decision.repositories import (
    AdExperimentRecord,
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
    PromotionRunRecord,
    PromotionRunWriter,
    PromotionTargetSegmentReader,
    PromotionTargetSegmentRecord,
)
from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.schemas import (
    AdExperimentCreateResponse,
    AdExperimentStatus,
    PromotionRunStatus,
    RunCreateRequest,
    RunCreateResponse,
)
from app.logging import log, log_context_scope, now_ms, duration_ms


COMPLETED_STATUS = "completed"
FALLBACK_SEGMENT_NAME = "Existing users fallback"
MAX_CONTRACT_ID_LENGTH = 100


class PromotionNotFoundError(Exception):
    pass


class RunValidationError(Exception):
    pass


class RunSegmentScopeValidationError(RunValidationError):
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
        partial_segment_scope_enabled: bool = False,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._promotion_analysis_repository = promotion_analysis_repository
        self._promotion_target_segment_repository = promotion_target_segment_repository
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._partial_segment_scope_enabled = partial_segment_scope_enabled

    @log_context_scope
    def create_run(
        self,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        started_at = now_ms()
        log.assign_context({"promotionId": promotion_id})
        log.info("started", {"promotionId": promotion_id, "request": request})
        requested_segment_ids = normalize_explicit_segment_ids(request.segment_ids)
        if (
            requested_segment_ids is not None
            and not self._partial_segment_scope_enabled
        ):
            raise RunConflictError(
                "explicit segment scope is disabled until Dashboard scope lineage is ready"
            )
        promotion = self._get_promotion(promotion_id)
        log.assign_context(
            {
                "projectId": promotion.project_id,
                "campaignId": promotion.campaign_id,
            }
        )
        log.info("promotion_loaded", {"promotion": promotion})
        analysis = self._select_analysis(
            promotion=promotion,
            analysis_id=request.analysis_id,
        )
        log.assign_context({"analysisId": analysis.analysis_id})
        log.info("promotion_analysis_loaded", {"analysis": analysis})
        generation = self._select_generation(
            promotion=promotion,
            analysis=analysis,
            generation_id=request.generation_id,
        )
        log.assign_context({"generationId": generation.generation_id})
        log.info("generation_run_loaded", {"generation": generation})

        snapshot_segment_ids = normalize_generation_segment_snapshot(
            generation.input_json.get("target_segment_ids"),
            required=requested_segment_ids is not None,
        )
        if requested_segment_ids is None:
            effective_segment_ids = snapshot_segment_ids
        else:
            if not set(requested_segment_ids).issubset(
                set(snapshot_segment_ids or ())
            ):
                raise RunValidationError(
                    "segment_ids must be a subset of the generation target_segment_ids snapshot"
                )
            effective_segment_ids = requested_segment_ids

        target_segments = self._load_target_segments(
            analysis,
            promotion,
            segment_ids=effective_segment_ids,
        )
        segment_ids = tuple(
            sorted({target_segment.segment_id for target_segment in target_segments})
        )
        segment_scope_fingerprint = build_segment_scope_fingerprint(
            segment_ids=segment_ids,
        )
        promotion_run_id = build_promotion_run_id(
            project_id=promotion.project_id,
            promotion_id=promotion.promotion_id,
            analysis_id=analysis.analysis_id,
            generation_id=generation.generation_id,
            loop_count=request.loop_count,
            segment_scope_fingerprint=segment_scope_fingerprint,
        )
        log.assign_context(
            {
                "segmentScopeFingerprint": segment_scope_fingerprint[:12],
                "segmentScopeCount": len(segment_ids),
            }
        )
        log.info("promotion_run_scope_resolved")
        existing_run = self._promotion_run_repository.get_by_scope(
            project_id=promotion.project_id,
            promotion_id=promotion.promotion_id,
            analysis_id=analysis.analysis_id,
            generation_id=generation.generation_id,
            segment_scope_fingerprint=segment_scope_fingerprint,
            loop_count=request.loop_count,
        )
        if existing_run is not None:
            response = self._reuse_existing_run(existing_run)
            log.info(
                "completed",
                {"response": response, "durationMs": duration_ms(started_at)},
            )
            return response

        log.info("target_segments_loaded", {"targetSegmentCount": len(target_segments)})
        content_by_segment = self._load_content_by_segment(generation.generation_id)
        log.info("content_candidates_loaded", {"segmentCount": len(content_by_segment)})
        selected_content = self._select_content_for_segments(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            target_segments=target_segments,
            content_by_segment=content_by_segment,
        )
        log.info("content_candidates_selected", {"selectedSegmentCount": len(selected_content)})

        run = self._build_promotion_run(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            promotion_run_id=promotion_run_id,
            loop_count=request.loop_count,
            segment_ids=segment_ids,
            segment_scope_fingerprint=segment_scope_fingerprint,
        )
        ad_experiments = self._build_ad_experiments(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            promotion_run_id=promotion_run_id,
            target_segments=target_segments,
            selected_content=selected_content,
            content_by_segment=content_by_segment,
            loop_count=request.loop_count,
        )
        log.assign_context({"promotionRunId": run.promotion_run_id})
        log.info("promotion_run_prepared", {"promotionRun": run, "adExperimentCount": len(ad_experiments)})

        inserted = self._promotion_run_repository.insert_if_absent(run)
        if not inserted:
            concurrent_run = self._promotion_run_repository.get_by_scope(
                project_id=run.project_id,
                promotion_id=run.promotion_id,
                analysis_id=run.analysis_id,
                generation_id=run.generation_id,
                segment_scope_fingerprint=run.segment_scope_fingerprint,
                loop_count=run.loop_count,
            )
            if concurrent_run is None:
                log.warn("promotion_run_id_collision")
                raise RunConflictError(
                    "promotion_run_id collided with a different segment scope"
                )
            response = self._reuse_existing_run(concurrent_run)
            log.info(
                "completed",
                {"response": response, "durationMs": duration_ms(started_at)},
            )
            return response

        self._ad_experiment_repository.insert_many(ad_experiments)
        log.info("promotion_run_created", {"promotionRun": run, "adExperiments": ad_experiments})
        response = self._build_response(run, ad_experiments)
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response

    def _reuse_existing_run(self, run: PromotionRunRecord) -> RunCreateResponse:
        ad_experiments = self._ad_experiment_repository.list_by_run(
            run.promotion_run_id
        )
        self._validate_existing_run_integrity(run, ad_experiments)
        log.assign_context({"promotionRunId": run.promotion_run_id})
        log.info("promotion_run_reused")
        return self._build_response(run, ad_experiments)

    def _validate_existing_run_integrity(
        self,
        run: PromotionRunRecord,
        ad_experiments: Sequence[AdExperimentRecord],
    ) -> None:
        experiment_segment_ids = [
            experiment.segment_id
            for experiment in ad_experiments
            if experiment.segment_id != FALLBACK_SEGMENT_ID
        ]
        fallback_count = sum(
            experiment.segment_id == FALLBACK_SEGMENT_ID
            for experiment in ad_experiments
        )
        if (
            tuple(sorted(experiment_segment_ids))
            != tuple(run.segment_scope_json)
            or len(experiment_segment_ids) != len(set(experiment_segment_ids))
            or fallback_count != 1
        ):
            log.warn(
                "promotion_run_scope_corrupted",
                {
                    "storedSegmentCount": len(run.segment_scope_json),
                    "experimentSegmentCount": len(experiment_segment_ids),
                    "fallbackCount": fallback_count,
                },
            )
            raise RunConflictError(
                "stored promotion_run experiments do not match its segment scope"
            )

    def _build_response(
        self,
        run: PromotionRunWrite | PromotionRunRecord,
        ad_experiments: Sequence[AdExperimentWrite | AdExperimentRecord],
    ) -> RunCreateResponse:
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
            segment_ids=list(run.segment_scope_json),
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
                    is_fallback=experiment.segment_id == FALLBACK_SEGMENT_ID,
                )
                for experiment in ad_experiments
            ],
        )

    def _get_promotion(self, promotion_id: str) -> PromotionRecord:
        promotion = self._promotion_repository.get_by_id(promotion_id)
        if promotion is None:
            log.warn("promotion_not_found", {"promotionId": promotion_id})
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
                log.warn("promotion_analysis_not_found", {"promotionId": promotion.promotion_id})
                raise RunValidationError(
                    "completed promotion analysis is required before creating a run"
                )
        else:
            analysis = self._promotion_analysis_repository.get_by_id(analysis_id)
            if analysis is None:
                log.warn("promotion_analysis_not_found", {"analysisId": analysis_id})
                raise RunValidationError(f"promotion analysis not found: {analysis_id}")

        if analysis.status != COMPLETED_STATUS:
            log.warn("promotion_analysis_invalid", {"analysisId": analysis.analysis_id, "status": analysis.status})
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
                log.warn("generation_run_not_found", {"promotionId": promotion.promotion_id})
                raise RunValidationError(
                    "completed generation run is required before creating a run"
                )
        else:
            generation = self._generation_run_repository.get_by_id(generation_id)
            if generation is None:
                log.warn("generation_run_not_found", {"generationId": generation_id})
                raise RunValidationError(f"generation run not found: {generation_id}")

        if generation.status != COMPLETED_STATUS:
            log.warn("generation_run_invalid", {"generationId": generation.generation_id, "status": generation.status})
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
        *,
        segment_ids: Sequence[str] | None,
    ) -> list[PromotionTargetSegmentRecord]:
        target_segments = (
            self._promotion_target_segment_repository.list_approved_for_analysis(
                analysis.analysis_id,
                segment_ids,
            )
        )
        target_segments = [
            segment
            for segment in target_segments
            if segment.segment_id != FALLBACK_SEGMENT_ID
        ]
        if not target_segments:
            log.warn("target_segments_empty", {"analysisId": analysis.analysis_id})
            if segment_ids is not None:
                raise RunValidationError(
                    "segment_ids must match approved promotion_target_segments"
                )
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
                log.warn("target_segment_mismatch", {"segmentId": segment.segment_id, "analysisId": segment.analysis_id})
                raise RunValidationError(
                    "target segment must belong to the selected promotion analysis"
                )
            if segment.segment_id in seen_segment_ids:
                log.warn("target_segment_conflict", {"segmentId": segment.segment_id})
                raise RunValidationError(
                    f"duplicate target segment is not allowed: {segment.segment_id}"
                )
            seen_segment_ids.add(segment.segment_id)

        if segment_ids is not None and seen_segment_ids != set(segment_ids):
            raise RunValidationError(
                "segment_ids must match approved promotion_target_segments"
            )
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
                log.warn("content_candidate_invalid", {"segmentId": segment.segment_id, "contentCandidateCount": len(candidates)})
                raise RunValidationError(
                    "each target segment must have exactly one approved or active "
                    f"content candidate: {segment.segment_id}"
                )
            candidate = candidates[0]
            _validate_content_candidate(
                candidate=candidate,
                promotion=promotion,
                analysis=analysis,
                generation=generation,
            )
            selected_content[segment.segment_id] = candidate
        return selected_content

    def _select_fallback_content(
        self,
        *,
        promotion: PromotionRecord,
        analysis: PromotionAnalysisRecord,
        generation: GenerationRunRecord,
        target_segments: Sequence[PromotionTargetSegmentRecord],
        selected_content: dict[str, ContentCandidateRecord],
        content_by_segment: dict[str, list[ContentCandidateRecord]],
    ) -> ContentCandidateRecord:
        fallback_candidates = content_by_segment.get(FALLBACK_SEGMENT_ID, [])
        if len(fallback_candidates) > 1:
            raise RunValidationError(
                "fallback segment must have at most one approved or active "
                "content candidate"
            )
        if fallback_candidates:
            candidate = fallback_candidates[0]
            _validate_content_candidate(
                candidate=candidate,
                promotion=promotion,
                analysis=analysis,
                generation=generation,
            )
            return candidate

        for segment in target_segments:
            if segment.segment_id != FALLBACK_SEGMENT_ID:
                return selected_content[segment.segment_id]

        raise RunValidationError("at least one non-fallback target segment is required")

    def _build_promotion_run(
        self,
        *,
        promotion: PromotionRecord,
        analysis: PromotionAnalysisRecord,
        generation: GenerationRunRecord,
        promotion_run_id: str,
        loop_count: int,
        segment_ids: Sequence[str],
        segment_scope_fingerprint: str,
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
            segment_scope_json=tuple(segment_ids),
            segment_scope_fingerprint=segment_scope_fingerprint,
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
        content_by_segment: dict[str, list[ContentCandidateRecord]],
        loop_count: int,
    ) -> list[AdExperimentWrite]:
        experiments: list[AdExperimentWrite] = []
        for segment in target_segments:
            content = selected_content[segment.segment_id]
            experiments.append(
                _build_ad_experiment(
                    promotion=promotion,
                    analysis=analysis,
                    generation=generation,
                    promotion_run_id=promotion_run_id,
                    segment_id=segment.segment_id,
                    segment_name=segment.segment_name,
                    content=content,
                    loop_count=loop_count,
                )
            )
        if not any(
            experiment.segment_id == FALLBACK_SEGMENT_ID for experiment in experiments
        ):
            fallback_content = self._select_fallback_content(
                promotion=promotion,
                analysis=analysis,
                generation=generation,
                target_segments=target_segments,
                selected_content=selected_content,
                content_by_segment=content_by_segment,
            )
            experiments.append(
                _build_ad_experiment(
                    promotion=promotion,
                    analysis=analysis,
                    generation=generation,
                    promotion_run_id=promotion_run_id,
                    segment_id=FALLBACK_SEGMENT_ID,
                    segment_name=FALLBACK_SEGMENT_NAME,
                    content=fallback_content,
                    loop_count=loop_count,
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


def build_promotion_run_id(
    *,
    project_id: str,
    promotion_id: str,
    analysis_id: str,
    generation_id: str,
    loop_count: int,
    segment_scope_fingerprint: str,
) -> str:
    if loop_count < 1:
        raise ValueError("loop_count must be at least 1")
    if re.fullmatch(r"[0-9a-f]{64}", segment_scope_fingerprint) is None:
        raise ValueError(
            "segment_scope_fingerprint must be a 64-character lowercase hex value"
        )

    prefix = "prun"
    loop_slug = f"loop_{loop_count}"
    scope_slug = segment_scope_fingerprint[:24]
    identity_seed = "::".join(
        (
            project_id,
            promotion_id,
            analysis_id,
            generation_id,
            str(loop_count),
            segment_scope_fingerprint,
        )
    )
    identity_digest = hashlib.sha256(identity_seed.encode("utf-8")).hexdigest()[:12]
    promotion_slug = re.sub(r"[^a-zA-Z0-9]+", "_", promotion_id).strip("_").lower()
    if not promotion_slug:
        promotion_slug = "id"

    max_promotion_slug_length = (
        MAX_CONTRACT_ID_LENGTH
        - len(prefix)
        - len(loop_slug)
        - len(scope_slug)
        - len(identity_digest)
        - 4
    )
    promotion_slug = (
        promotion_slug[:max_promotion_slug_length].rstrip("_") or "id"
    )
    return f"{prefix}_{promotion_slug}_{loop_slug}_{scope_slug}_{identity_digest}"


def build_segment_scope_fingerprint(
    *,
    segment_ids: Sequence[str],
) -> str:
    serialized = json.dumps(
        sorted(
            segment_id
            for segment_id in set(segment_ids)
            if segment_id != FALLBACK_SEGMENT_ID
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_explicit_segment_ids(
    segment_ids: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if segment_ids is None:
        return None
    normalized: set[str] = set()
    for segment_id in segment_ids:
        value = segment_id.strip()
        if not value:
            raise RunSegmentScopeValidationError(
                "segment_ids must not contain blank values"
            )
        if value == FALLBACK_SEGMENT_ID:
            raise RunSegmentScopeValidationError(
                "segment_ids must not include the fallback segment"
            )
        normalized.add(value)
    if not normalized:
        raise RunSegmentScopeValidationError(
            "segment_ids must contain at least one segment"
        )
    return tuple(sorted(normalized))


def normalize_generation_segment_snapshot(
    snapshot: object,
    *,
    required: bool,
) -> tuple[str, ...] | None:
    if snapshot is None and not required:
        return None
    if not isinstance(snapshot, list):
        raise RunValidationError(
            "generation run must include a valid target_segment_ids snapshot"
        )
    normalized: set[str] = set()
    for segment_id in snapshot:
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise RunValidationError(
                "generation run must include a valid target_segment_ids snapshot"
            )
        value = segment_id.strip()
        if value != FALLBACK_SEGMENT_ID:
            normalized.add(value)
    if not normalized:
        raise RunValidationError(
            "generation run must include a valid target_segment_ids snapshot"
        )
    return tuple(sorted(normalized))


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


def _build_ad_experiment(
    *,
    promotion: PromotionRecord,
    analysis: PromotionAnalysisRecord,
    generation: GenerationRunRecord,
    promotion_run_id: str,
    segment_id: str,
    segment_name: str | None,
    content: ContentCandidateRecord,
    loop_count: int,
) -> AdExperimentWrite:
    return AdExperimentWrite(
        ad_experiment_id=build_bounded_decision_id(
            "adexp",
            promotion_run_id,
            segment_id,
        ),
        project_id=promotion.project_id,
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        promotion_run_id=promotion_run_id,
        analysis_id=analysis.analysis_id,
        generation_id=generation.generation_id,
        segment_id=segment_id,
        segment_name=segment_name,
        content_id=content.content_id,
        content_option_id=content.content_option_id,
        channel=promotion.channel,
        loop_count=loop_count,
        status=AdExperimentStatus.PLANNED.value,
        goal_metric=promotion.goal_metric,
        goal_target_value=promotion.goal_target_value,
        goal_basis=promotion.goal_basis,
    )


def _validate_content_candidate(
    *,
    candidate: ContentCandidateRecord,
    promotion: PromotionRecord,
    analysis: PromotionAnalysisRecord,
    generation: GenerationRunRecord,
) -> None:
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
