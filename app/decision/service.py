from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal
from typing import Mapping, Sequence

from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWrite,
    AdExperimentWriter,
    ContentCandidateReader,
    ContentCandidateRecord,
    GenerationRunReader,
    GenerationRunRecord,
    NextLoopPreparationConflictError,
    NextLoopPreparationRecord,
    NextLoopPreparationWriter,
    PromotionAnalysisReader,
    PromotionAnalysisRecord,
    PromotionEvaluationRecord,
    PromotionEvaluationWriter,
    PromotionReader,
    PromotionRecord,
    PromotionRunRecord,
    PromotionRunWrite,
    PromotionRunWriter,
    PromotionTargetSegmentReader,
    PromotionTargetSegmentRecord,
)
from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.schemas import (
    AdExperimentCreateResponse,
    AdExperimentStatus,
    PromotionEvaluationStatus,
    PromotionRunStatus,
    RunCreateRequest,
    RunCreateResponse,
)
from app.logging import log, log_context_scope, now_ms, duration_ms


COMPLETED_STATUS = "completed"
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
        promotion_evaluation_repository: PromotionEvaluationWriter,
        next_loop_preparation_repository: NextLoopPreparationWriter,
        manual_activation_enabled: bool = False,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._promotion_analysis_repository = promotion_analysis_repository
        self._promotion_target_segment_repository = promotion_target_segment_repository
        self._generation_run_repository = generation_run_repository
        self._content_candidate_repository = content_candidate_repository
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._promotion_evaluation_repository = promotion_evaluation_repository
        self._next_loop_preparation_repository = next_loop_preparation_repository
        self._manual_activation_enabled = manual_activation_enabled

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
        analysis_id = request.analysis_id
        generation_id = request.generation_id
        loop_count = request.loop_count
        if request.next_loop_preparation_id is not None:
            if not self._manual_activation_enabled:
                raise RunConflictError(
                    "manual next-loop preparation activation is disabled"
                )
            response = self._activate_prepared_run(
                promotion_id=promotion_id,
                request=request,
            )
            log.info(
                "completed",
                {"response": response, "durationMs": duration_ms(started_at)},
            )
            return response
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
            analysis_id=analysis_id,
        )
        log.assign_context({"analysisId": analysis.analysis_id})
        log.info("promotion_analysis_loaded", {"analysis": analysis})
        generation = self._select_generation(
            promotion=promotion,
            analysis=analysis,
            generation_id=generation_id,
        )
        log.assign_context({"generationId": generation.generation_id})
        log.info("generation_run_loaded", {"generation": generation})

        snapshot_segment_ids = normalize_generation_segment_snapshot(
            generation.input_json.get("target_segment_ids"),
            target_segments_snapshot=generation.input_json.get("target_segments"),
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
            loop_count=loop_count,
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
            loop_count=loop_count,
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
            loop_count=loop_count,
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
            loop_count=loop_count,
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

    def _activate_preparation(
        self,
        preparation: NextLoopPreparationRecord | None,
        promotion_run_id: str,
    ) -> None:
        if preparation is None or preparation.status == "activated":
            return
        repository = self._next_loop_preparation_repository
        if repository is None:
            raise RunConflictError(
                "next-loop preparation activation is not configured"
            )
        try:
            activated = repository.mark_activated(
                next_loop_preparation_id=preparation.next_loop_preparation_id,
                activated_promotion_run_id=promotion_run_id,
            )
        except NextLoopPreparationConflictError as exc:
            raise RunConflictError(
                "canonical promotion run is already activated by another "
                "next-loop preparation"
            ) from exc
        if activated is not None:
            return
        current = repository.get_by_id(preparation.next_loop_preparation_id)
        if (
            current is None
            or current.status != "activated"
            or current.activated_promotion_run_id != promotion_run_id
        ):
            raise RunConflictError(
                "next-loop preparation could not be activated"
            )

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
            or fallback_count > 1
            or build_segment_scope_fingerprint(run.segment_scope_json)
            != run.segment_scope_fingerprint
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
        segment_order = {
            segment_id: index
            for index, segment_id in enumerate(run.segment_scope_json)
        }
        ordered_experiments = sorted(
            ad_experiments,
            key=lambda experiment: (
                segment_order.get(experiment.segment_id, len(segment_order)),
                experiment.segment_id,
            ),
        )
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
                for experiment in ordered_experiments
            ],
        )

    def _activate_prepared_run(
        self,
        *,
        promotion_id: str,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        preparation_id = request.next_loop_preparation_id
        if preparation_id is None:
            raise AssertionError("preparation activation requires an id")

        preparation = self._next_loop_preparation_repository.get_by_id_for_update(
            preparation_id
        )
        if preparation is None:
            raise RunValidationError(
                f"next-loop preparation not found: {preparation_id}"
            )

        if preparation.status == "activated":
            promotion = self._get_promotion(promotion_id)
            return self._load_canonical_activated_run(
                promotion=promotion,
                preparation=preparation,
                request=request,
            )
        if preparation.status == "rejected":
            raise RunValidationError("next-loop preparation is rejected")
        if preparation.status != "awaiting_content_approval":
            raise RunConflictError("next-loop preparation has an invalid status")

        promotion = self._get_promotion(promotion_id)
        self._validate_activation_request(
            preparation=preparation,
            request=request,
        )
        source_run = self._require_source_run(
            preparation=preparation,
            promotion=promotion,
        )
        expected_loop_count = source_run.loop_count + 1
        if request.loop_count != expected_loop_count:
            raise RunValidationError(
                "loop_count must be the source promotion run loop_count plus one"
            )
        if request.loop_count > promotion.max_loop_count:
            raise RunValidationError("promotion max_loop_count exceeded")

        analysis = self._select_analysis(
            promotion=promotion,
            analysis_id=request.analysis_id,
        )
        generation = self._select_generation(
            promotion=promotion,
            analysis=analysis,
            generation_id=request.generation_id,
        )
        normalized_preparation_scope = normalize_explicit_segment_ids(
            preparation.failed_segment_ids_json
        )
        if normalized_preparation_scope is None:
            raise RunValidationError("next-loop preparation segment scope is empty")
        expected_segment_ids = list(normalized_preparation_scope)
        segment_scope_fingerprint = build_segment_scope_fingerprint(
            segment_ids=expected_segment_ids
        )
        self._promotion_run_repository.lock_activation_scope(
            project_id=promotion.project_id,
            promotion_id=promotion.promotion_id,
            analysis_id=analysis.analysis_id,
            generation_id=generation.generation_id,
            segment_scope_fingerprint=segment_scope_fingerprint,
            loop_count=request.loop_count,
        )
        target_segments = self._load_target_segments(
            analysis,
            promotion,
            segment_ids=expected_segment_ids,
        )
        self._validate_generation_segment_snapshot(
            generation=generation,
            requested_segment_ids=expected_segment_ids,
        )

        content_by_segment = self._load_content_by_segment(generation.generation_id)
        if set(content_by_segment) != set(expected_segment_ids):
            raise RunValidationError(
                "approved or active content candidates must match every expected "
                "segment and no unexpected segment"
            )
        selected_content = self._select_content_for_segments(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            target_segments=target_segments,
            content_by_segment=content_by_segment,
        )
        lineage_by_segment = self._load_validated_lineage(
            preparation=preparation,
            source_run=source_run,
            promotion=promotion,
        )

        promotion_run_id = build_promotion_run_id(
            project_id=promotion.project_id,
            promotion_id=promotion.promotion_id,
            analysis_id=analysis.analysis_id,
            generation_id=generation.generation_id,
            loop_count=request.loop_count,
            segment_scope_fingerprint=segment_scope_fingerprint,
        )
        existing_run = self._promotion_run_repository.get_by_scope(
            project_id=promotion.project_id,
            promotion_id=promotion.promotion_id,
            analysis_id=analysis.analysis_id,
            generation_id=generation.generation_id,
            segment_scope_fingerprint=segment_scope_fingerprint,
            loop_count=request.loop_count,
        )
        if existing_run is not None:
            experiments = self._ad_experiment_repository.list_by_run(
                existing_run.promotion_run_id
            )
            self._validate_existing_run_integrity(existing_run, experiments)
            self._validate_canonical_experiments(
                preparation=preparation,
                run=existing_run,
                experiments=experiments,
            )
            self._activate_preparation(preparation, existing_run.promotion_run_id)
            return self._build_response(existing_run, experiments)

        run = self._build_promotion_run(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            promotion_run_id=promotion_run_id,
            loop_count=request.loop_count,
            segment_ids=expected_segment_ids,
            segment_scope_fingerprint=segment_scope_fingerprint,
        )
        ad_experiments = self._build_ad_experiments(
            promotion=promotion,
            analysis=analysis,
            generation=generation,
            promotion_run_id=promotion_run_id,
            target_segments=target_segments,
            selected_content=selected_content,
            loop_count=request.loop_count,
            lineage_by_segment=lineage_by_segment,
        )
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
                raise RunConflictError(
                    "promotion_run_id collided with a different segment scope"
                )
            experiments = self._ad_experiment_repository.list_by_run(
                concurrent_run.promotion_run_id
            )
            self._validate_existing_run_integrity(concurrent_run, experiments)
            self._validate_canonical_experiments(
                preparation=preparation,
                run=concurrent_run,
                experiments=experiments,
            )
            self._activate_preparation(preparation, concurrent_run.promotion_run_id)
            return self._build_response(concurrent_run, experiments)

        self._ad_experiment_repository.insert_many(ad_experiments)
        self._activate_preparation(preparation, run.promotion_run_id)
        return self._build_response(run, ad_experiments)

    def _load_canonical_activated_run(
        self,
        *,
        promotion: PromotionRecord,
        preparation: NextLoopPreparationRecord,
        request: RunCreateRequest,
    ) -> RunCreateResponse:
        self._validate_activation_request(
            preparation=preparation,
            request=request,
        )
        source_run = self._require_source_run(
            preparation=preparation,
            promotion=promotion,
            conflict_on_invalid=True,
        )
        activated_run_id = preparation.activated_promotion_run_id
        if activated_run_id is None:
            raise RunConflictError(
                "activated next-loop preparation has no canonical promotion run"
            )
        run = self._promotion_run_repository.get_by_id(activated_run_id)
        if run is None:
            raise RunConflictError(
                "activated next-loop preparation canonical promotion run was not found"
            )
        if (
            run.project_id != promotion.project_id
            or run.campaign_id != promotion.campaign_id
            or run.promotion_id != promotion.promotion_id
            or run.analysis_id != preparation.analysis_id
            or run.generation_id != preparation.generation_id
            or run.loop_count != source_run.loop_count + 1
            or request.loop_count != run.loop_count
            or tuple(sorted(run.segment_scope_json))
            != tuple(sorted(preparation.failed_segment_ids_json))
            or run.segment_scope_fingerprint
            != build_segment_scope_fingerprint(
                segment_ids=preparation.failed_segment_ids_json
            )
        ):
            raise RunConflictError(
                "activated next-loop preparation canonical promotion run is invalid"
            )

        experiments = self._ad_experiment_repository.list_by_run(run.promotion_run_id)
        self._validate_canonical_experiments(
            preparation=preparation,
            run=run,
            experiments=experiments,
        )
        self._validate_existing_run_integrity(run, experiments)
        return self._build_response(run, experiments)

    def _validate_activation_request(
        self,
        *,
        preparation: NextLoopPreparationRecord,
        request: RunCreateRequest,
    ) -> None:
        if request.analysis_id is None or request.generation_id is None:
            raise RunValidationError(
                "analysis_id and generation_id are required for preparation activation"
            )
        if request.analysis_id != preparation.analysis_id:
            raise RunValidationError(
                "analysis_id must match the next-loop preparation"
            )
        if request.generation_id != preparation.generation_id:
            raise RunValidationError(
                "generation_id must match the next-loop preparation"
            )
        if (
            request.segment_ids is not None
            and set(request.segment_ids) != set(preparation.failed_segment_ids_json)
        ):
            raise RunValidationError(
                "segment_ids must match the next-loop preparation failed segments"
            )

    def _require_source_run(
        self,
        *,
        preparation: NextLoopPreparationRecord,
        promotion: PromotionRecord,
        conflict_on_invalid: bool = False,
    ) -> PromotionRunRecord:
        source_run = self._promotion_run_repository.get_by_id(
            preparation.source_promotion_run_id
        )
        invalid = source_run is None or (
            source_run.project_id != promotion.project_id
            or source_run.campaign_id != promotion.campaign_id
            or source_run.promotion_id != promotion.promotion_id
        )
        if invalid:
            message = "next-loop preparation source promotion run is invalid"
            if conflict_on_invalid:
                raise RunConflictError(message)
            raise RunValidationError(message)
        source_experiments = self._ad_experiment_repository.list_by_run(
            source_run.promotion_run_id
        )
        try:
            self._validate_existing_run_integrity(source_run, source_experiments)
        except RunConflictError as exc:
            message = "next-loop preparation source promotion run is invalid"
            if conflict_on_invalid:
                raise RunConflictError(message) from exc
            raise RunValidationError(message) from exc
        if not set(preparation.failed_segment_ids_json).issubset(
            set(source_run.segment_scope_json)
        ):
            message = "next-loop preparation exceeds its source run segment scope"
            if conflict_on_invalid:
                raise RunConflictError(message)
            raise RunValidationError(message)
        return source_run

    def _load_validated_lineage(
        self,
        *,
        preparation: NextLoopPreparationRecord,
        source_run: PromotionRunRecord,
        promotion: PromotionRecord,
    ) -> dict[str, tuple[str, str]]:
        expected_segment_ids = set(preparation.failed_segment_ids_json)
        expected_experiment_ids = set(preparation.failed_ad_experiment_ids_json)
        expected_evaluation_ids = set(preparation.source_evaluation_ids_json)
        expected_count = len(expected_segment_ids)
        if (
            expected_count == 0
            or len(preparation.failed_segment_ids_json) != expected_count
            or len(preparation.failed_ad_experiment_ids_json) != expected_count
            or len(preparation.source_evaluation_ids_json) != expected_count
            or len(expected_experiment_ids) != expected_count
            or len(expected_evaluation_ids) != expected_count
        ):
            raise RunValidationError(
                "preparation lineage ids must map one-to-one to expected segments"
            )

        source_experiments = [
            experiment
            for experiment in self._ad_experiment_repository.list_by_run(
                source_run.promotion_run_id
            )
            if experiment.ad_experiment_id in expected_experiment_ids
        ]
        if {
            experiment.ad_experiment_id for experiment in source_experiments
        } != expected_experiment_ids:
            raise RunValidationError(
                "failed ad experiments must belong to the preparation source run"
            )

        experiments_by_segment: dict[str, AdExperimentRecord] = {}
        for experiment in source_experiments:
            if (
                experiment.project_id != promotion.project_id
                or experiment.campaign_id != promotion.campaign_id
                or experiment.promotion_id != promotion.promotion_id
                or experiment.promotion_run_id != source_run.promotion_run_id
                or experiment.analysis_id != source_run.analysis_id
                or experiment.generation_id != source_run.generation_id
                or experiment.loop_count != source_run.loop_count
                or experiment.segment_id not in expected_segment_ids
                or experiment.segment_id in experiments_by_segment
            ):
                raise RunValidationError(
                    "source ad experiments must map one-to-one to expected segments"
                )
            experiments_by_segment[experiment.segment_id] = experiment
        if set(experiments_by_segment) != expected_segment_ids:
            raise RunValidationError(
                "source ad experiments must cover every expected segment"
            )

        latest_evaluations = (
            self._promotion_evaluation_repository.list_latest_by_run_ad_experiments(
                source_run.promotion_run_id
            )
        )
        evaluations_by_experiment: dict[str, PromotionEvaluationRecord] = {}
        for evaluation in latest_evaluations:
            ad_experiment_id = evaluation.ad_experiment_id
            if ad_experiment_id not in expected_experiment_ids:
                continue
            if ad_experiment_id is None or ad_experiment_id in evaluations_by_experiment:
                raise RunValidationError(
                    "source evaluations must map one-to-one to parent experiments"
                )
            evaluations_by_experiment[ad_experiment_id] = evaluation

        lineage_by_segment: dict[str, tuple[str, str]] = {}
        consumed_evaluation_ids: set[str] = set()
        for segment_id, experiment in experiments_by_segment.items():
            evaluation = evaluations_by_experiment.get(experiment.ad_experiment_id)
            if evaluation is None:
                raise RunValidationError(
                    "latest individual evaluation is required for each parent experiment"
                )
            if (
                evaluation.evaluation_id not in expected_evaluation_ids
                or evaluation.project_id != promotion.project_id
                or evaluation.campaign_id != promotion.campaign_id
                or evaluation.promotion_id != promotion.promotion_id
                or evaluation.promotion_run_id != source_run.promotion_run_id
                or evaluation.ad_experiment_id != experiment.ad_experiment_id
                or evaluation.segment_id != segment_id
                or evaluation.status
                != PromotionEvaluationStatus.GOAL_NOT_MET.value
            ):
                raise RunValidationError(
                    "source evaluation must be the latest individual goal_not_met "
                    "evaluation for the parent experiment and segment"
                )
            consumed_evaluation_ids.add(evaluation.evaluation_id)
            lineage_by_segment[segment_id] = (
                experiment.ad_experiment_id,
                evaluation.evaluation_id,
            )

        if consumed_evaluation_ids != expected_evaluation_ids:
            raise RunValidationError(
                "source evaluation ids must map one-to-one to expected segments"
            )
        return lineage_by_segment

    def _validate_canonical_experiments(
        self,
        *,
        preparation: NextLoopPreparationRecord,
        run: PromotionRunRecord,
        experiments: Sequence[AdExperimentRecord],
    ) -> None:
        expected_segment_ids = set(preparation.failed_segment_ids_json)
        expected_count = len(expected_segment_ids)
        if (
            expected_count == 0
            or len(preparation.failed_segment_ids_json) != expected_count
            or len(preparation.failed_ad_experiment_ids_json) != expected_count
            or len(set(preparation.failed_ad_experiment_ids_json)) != expected_count
            or len(preparation.source_evaluation_ids_json) != expected_count
            or len(set(preparation.source_evaluation_ids_json)) != expected_count
        ):
            raise RunConflictError(
                "activated next-loop preparation canonical lineage is invalid"
            )
        non_fallback = [
            experiment
            for experiment in experiments
            if experiment.segment_id != FALLBACK_SEGMENT_ID
        ]
        fallback = [
            experiment
            for experiment in experiments
            if experiment.segment_id == FALLBACK_SEGMENT_ID
        ]
        if (
            len(non_fallback) != len(expected_segment_ids)
            or {experiment.segment_id for experiment in non_fallback}
            != expected_segment_ids
            or len(fallback) > 1
        ):
            raise RunConflictError(
                "activated next-loop preparation canonical experiments are invalid"
            )

        parent_ids: set[str] = set()
        evaluation_ids: set[str] = set()
        for experiment in experiments:
            if (
                experiment.project_id != run.project_id
                or experiment.campaign_id != run.campaign_id
                or experiment.promotion_id != run.promotion_id
                or experiment.promotion_run_id != run.promotion_run_id
                or experiment.analysis_id != run.analysis_id
                or experiment.generation_id != run.generation_id
                or experiment.loop_count != run.loop_count
            ):
                raise RunConflictError(
                    "activated next-loop preparation canonical experiment context is invalid"
                )
            if experiment.segment_id == FALLBACK_SEGMENT_ID:
                if (
                    experiment.parent_ad_experiment_id is not None
                    or experiment.source_evaluation_id is not None
                ):
                    raise RunConflictError(
                        "canonical fallback experiment must not synthesize lineage"
                    )
                continue
            if (
                experiment.parent_ad_experiment_id is None
                or experiment.source_evaluation_id is None
            ):
                raise RunConflictError(
                    "canonical child experiments must preserve stored lineage"
                )
            parent_ids.add(experiment.parent_ad_experiment_id)
            evaluation_ids.add(experiment.source_evaluation_id)

        if (
            parent_ids != set(preparation.failed_ad_experiment_ids_json)
            or evaluation_ids != set(preparation.source_evaluation_ids_json)
        ):
            raise RunConflictError(
                "canonical child experiment lineage does not match preparation"
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

    def _validate_generation_segment_snapshot(
        self,
        *,
        generation: GenerationRunRecord,
        requested_segment_ids: Sequence[str] | None,
    ) -> None:
        if requested_segment_ids is None:
            return

        snapshot = normalize_generation_segment_snapshot(
            generation.input_json.get("target_segment_ids"),
            target_segments_snapshot=generation.input_json.get("target_segments"),
            required=True,
        )
        if set(snapshot or ()) != set(requested_segment_ids):
            raise RunValidationError(
                "segment_ids must match the generation target_segment_ids snapshot"
            )

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
        segment_scope_json = tuple(
            sorted(
                segment_id
                for segment_id in set(segment_ids)
                if segment_id != FALLBACK_SEGMENT_ID
            )
        )
        if not segment_scope_json:
            raise RunValidationError(
                "promotion run segment scope must contain a non-fallback segment"
            )
        if build_segment_scope_fingerprint(segment_scope_json) != (
            segment_scope_fingerprint
        ):
            raise RunValidationError(
                "promotion run segment scope fingerprint does not match its scope"
            )
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
            segment_scope_json=segment_scope_json,
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
        loop_count: int,
        lineage_by_segment: dict[str, tuple[str, str]] | None = None,
    ) -> list[AdExperimentWrite]:
        experiments: list[AdExperimentWrite] = []
        for segment in target_segments:
            content = selected_content[segment.segment_id]
            parent_ad_experiment_id: str | None = None
            source_evaluation_id: str | None = None
            if lineage_by_segment is not None:
                lineage = lineage_by_segment.get(segment.segment_id)
                if lineage is None:
                    raise RunValidationError(
                        "lineage is required for every prepared child segment"
                    )
                parent_ad_experiment_id, source_evaluation_id = lineage
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
                    parent_ad_experiment_id=parent_ad_experiment_id,
                    source_evaluation_id=source_evaluation_id,
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


def build_segment_scope_fingerprint(segment_ids: Sequence[str]) -> str:
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
    target_segments_snapshot: object = None,
    required: bool,
) -> tuple[str, ...] | None:
    if snapshot is None:
        snapshot = _segment_ids_from_target_segments_snapshot(
            target_segments_snapshot
        )
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


def _segment_ids_from_target_segments_snapshot(
    snapshot: object,
) -> list[str] | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, list) or not snapshot:
        raise RunValidationError(
            "generation run must include a valid target_segments snapshot"
        )

    segment_ids: list[str] = []
    for target_segment in snapshot:
        if not isinstance(target_segment, Mapping):
            raise RunValidationError(
                "generation run must include a valid target_segments snapshot"
            )
        segment_id = target_segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise RunValidationError(
                "generation run must include a valid target_segments snapshot"
            )
        segment_ids.append(segment_id)
    return segment_ids


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
    parent_ad_experiment_id: str | None = None,
    source_evaluation_id: str | None = None,
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
        parent_ad_experiment_id=parent_ad_experiment_id,
        source_evaluation_id=source_evaluation_id,
        channel=promotion.channel,
        loop_count=loop_count,
        status=AdExperimentStatus.PLANNED.value,
        goal_metric=promotion.goal_metric,
        goal_target_value=promotion.goal_target_value,
        goal_basis=promotion.goal_basis,
    )


def _run_create_response(
    *,
    run: PromotionRunRecord | PromotionRunWrite,
    ad_experiments: Sequence[AdExperimentRecord | AdExperimentWrite],
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


def _validate_content_candidate(
    *,
    candidate: ContentCandidateRecord,
    promotion: PromotionRecord,
    analysis: PromotionAnalysisRecord,
    generation: GenerationRunRecord,
) -> None:
    if candidate.status not in {"approved", "active"}:
        raise RunValidationError(
            "content candidate must be approved or active"
        )
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
