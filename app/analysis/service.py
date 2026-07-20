from __future__ import annotations

import hashlib
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Literal, Mapping, Protocol, Sequence

from app.audience_contract import (
    LEGACY_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_CONTRACT,
    SegmentAudienceContractError,
    SegmentDefinitionAudienceAdapter,
)
from app.analysis.audience_v2 import AudienceV2Coordinator, AudienceV2Preparation
from app.audience_allocation import AudienceAllocationService
from app.analysis.audience_snapshot_repository import AudienceSnapshotBindingError
from app.analysis.booking_model import (
    BookingPropensityModel,
    BookingPropensityPrediction,
    train_booking_propensity_model,
)
from app.analysis.repositories import (
    BookingTrainingRecord,
    HotelMarketingProfileRecord,
    PromotionAnalysisWrite,
    PromotionRecord,
    PromotionSegmentSuggestionWrite,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
    SegmentSuggestionAudienceBindingRecord,
)
from app.analysis.report_generator import (
    DeterministicSegmentSuggestionReportGenerator,
    SegmentSuggestionReportGenerator,
    SegmentSuggestionReportInput,
)
from app.analysis.segment_suggester import DEFAULT_MAX_SUGGESTED_SEGMENTS
from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisStatus,
    Channel,
    GoalMetric,
    SegmentAnalysisRequest,
)
from app.analysis.vector_service import (
    SegmentVectorBuildRequest,
    SegmentVectorBuildResult,
)
from app.content_brief import build_content_brief_v2
from app.logging import log, log_context_scope, now_ms, duration_ms


MAX_DEFAULT_TARGET_SEGMENTS = DEFAULT_MAX_SUGGESTED_SEGMENTS
MAX_SEGMENT_REPORT_WORKERS = 4

TargetSegmentStatus = Literal["planned", "approved"]

CUSTOM_SEGMENT_SOURCES = {"custom_chatkit", "manual_rule"}

DEFAULT_SEGMENT_IDS_BY_CHANNEL = {
    Channel.EMAIL.value: (
        "seg_mobile_user",
        "seg_family_trip",
        "seg_near_checkin",
        "seg_existing_all",
    ),
    Channel.ONSITE_BANNER.value: (
        "seg_family_trip",
        "seg_mobile_user",
        "seg_repeat_hotel_no_booking",
        "seg_near_checkin",
    ),
    Channel.SMS.value: (
        "seg_near_checkin",
        "seg_mobile_user",
        "seg_family_trip",
        "seg_existing_all",
    ),
}

SEGMENT_CONTENT_HINTS = {
    "seg_mobile_user": (
        "Reduce steps and emphasize mobile-friendly booking.",
        ("mobile booking", "quick checkout", "easy reservation"),
    ),
    "seg_family_trip": (
        "Highlight family rooms, breakfast, and flexible cancellation.",
        ("family room", "breakfast included", "flexible cancellation"),
    ),
    "seg_couple_trip": (
        "Highlight two-person stays and convenient hotel conditions.",
        ("couple stay", "hotel deal", "late checkout"),
    ),
    "seg_package_trip": (
        "Emphasize bundled stay benefits and clear package value.",
        ("package deal", "bundled stay", "travel value"),
    ),
    "seg_long_stay": (
        "Highlight benefits for longer stays and stable availability.",
        ("long stay", "weekly rate", "room availability"),
    ),
    "seg_near_checkin": (
        "Emphasize near check-in availability and low-friction booking.",
        ("near check-in", "same-day availability", "free cancellation"),
    ),
    "seg_repeat_hotel_no_booking": (
        "Emphasize free cancellation, same-day availability, and breakfast benefits.",
        ("free cancellation", "same-day availability", "breakfast included"),
    ),
    "seg_existing_all": (
        "Use a broad hotel booking message for existing users.",
        ("hotel deal", "seasonal stay", "booking benefit"),
    ),
}

RELATED_TERMS_BY_GOAL = {
    GoalMetric.INFLOW_RATE.value: (
        "inflow",
        "visit",
        "traffic",
        "click",
        "mobile",
        "banner",
        "email",
        "sms",
    ),
    GoalMetric.BOOKING_CONVERSION_RATE.value: (
        "booking",
        "book",
        "hotel",
        "stay",
        "checkin",
        "reservation",
    ),
    GoalMetric.FUNNEL_STEP_RATE.value: (
        "funnel",
        "step",
        "search",
        "detail",
        "booking",
        "checkin",
    ),
}

SIGNAL_COPY_BY_FEATURE = {
    "Booking conversion ready users": {
        "key": "booking_conversion_ready",
        "chip": "예약 가능성 높음",
    },
    "Promotion-engaged hotel users": {
        "key": "promotion_engaged",
        "chip": "프로모션 반응",
    },
    "Campaign redirect users": {
        "key": "campaign_redirect",
        "chip": "이메일 링크 클릭",
    },
    "Campaign landing users": {
        "key": "campaign_landing",
        "chip": "캠페인 랜딩",
    },
    "Experiment-exposed hotel users": {
        "key": "experiment_exposed",
        "chip": "광고 노출 이력",
    },
    "Hotel page viewers": {
        "key": "hotel_browsing",
        "chip": "호텔 탐색",
    },
    "Hotel search users": {
        "key": "hotel_search",
        "chip": "숙소 검색",
    },
    "Hotel click users": {
        "key": "hotel_click",
        "chip": "숙소 클릭",
    },
    "Hotel detail viewers": {
        "key": "hotel_detail",
        "chip": "상세 조회",
    },
    "Booking starters": {
        "key": "booking_start",
        "chip": "예약 시작",
    },
    "Booking converters": {
        "key": "booking_complete",
        "chip": "예약 완료",
    },
    "Booking cancellation risk users": {
        "key": "booking_cancel_risk",
        "chip": "취소 위험",
    },
    "Mixed event hotel users": {
        "key": "mixed_hotel_behavior",
        "chip": "복합 행동",
    },
    "Free cancellation seekers": {
        "key": "free_cancellation",
        "chip": "무료 취소 선호",
    },
    "Breakfast-included seekers": {
        "key": "breakfast_included",
        "chip": "조식 선호",
    },
    "Higher-price hotel shoppers": {
        "key": "higher_price",
        "chip": "고가 숙소 탐색",
    },
    "Promotion click responsive users": {
        "key": "promotion_click_responsive",
        "chip": "클릭 반응 높음",
    },
    "Mobile hotel users": {
        "key": "mobile_hotel_user",
        "chip": "모바일 이용",
    },
}

DEFAULT_SIGNAL_COPY_BY_SEGMENT_ID = {
    "seg_mobile_user": ("mobile_hotel_user", "모바일 이용"),
    "seg_family_trip": ("family_trip", "가족 여행 관심"),
    "seg_near_checkin": ("near_checkin", "임박 예약 관심"),
    "seg_existing_all": ("existing_users", "기존 사용자"),
    "seg_repeat_hotel_no_booking": ("repeat_hotel_viewer", "반복 조회"),
}

SEGMENT_TITLE_BY_SIGNAL = {
    ("booking_conversion_ready", "promotion_engaged"): "예약 가능성이 높은 프로모션 반응 고객",
    ("campaign_redirect", "promotion_engaged"): "캠페인 링크 반응이 높은 고객",
    ("campaign_landing", "promotion_engaged"): "캠페인 랜딩 후 관심 고객",
    ("hotel_browsing", "promotion_engaged"): "호텔 탐색이 활발한 프로모션 반응 고객",
    ("hotel_search", "promotion_engaged"): "숙소 검색이 활발한 프로모션 반응 고객",
    ("mobile_hotel_user",): "모바일 예약 선호 고객",
    ("family_trip",): "가족 여행 관심 고객",
    ("near_checkin",): "임박 예약 가능성이 높은 고객",
    ("existing_users",): "기존 사용자 전체 고객",
    ("repeat_hotel_viewer",): "반복 조회 후 예약 전환이 필요한 고객",
    ("booking_start",): "예약을 시작한 고객",
    ("booking_complete",): "예약 완료 경험 고객",
    ("campaign_redirect",): "이메일 링크 반응 고객",
    ("campaign_landing",): "캠페인 랜딩 고객",
    ("free_cancellation",): "무료 취소 혜택 선호 고객",
    ("breakfast_included",): "조식 혜택 관심 고객",
    ("higher_price",): "프리미엄 숙소 관심 고객",
    ("hotel_market_affinity",): "특정 지역 선호 고객",
    ("hotel_cluster_affinity",): "숙소 취향이 뚜렷한 고객",
    ("hotel_path_pattern",): "탐색 경로가 유사한 고객",
    ("promotion_click_responsive",): "프로모션 클릭 반응이 높은 고객",
}

GOAL_REASON_COPY = {
    GoalMetric.BOOKING_CONVERSION_RATE.value: (
        "예약 전환 목표에 가까운 행동 패턴을 보인 고객군입니다."
    ),
    GoalMetric.INFLOW_RATE.value: (
        "유입 확대 목표에 맞는 방문과 클릭 반응이 확인된 고객군입니다."
    ),
    GoalMetric.FUNNEL_STEP_RATE.value: (
        "예약 퍼널의 다음 단계로 이동할 가능성이 있는 행동 패턴입니다."
    ),
}

CHANNEL_ACTION_COPY = {
    Channel.EMAIL.value: "이메일 예약 혜택 메시지의 우선 타겟으로 적합합니다.",
    Channel.SMS.value: "짧은 예약 혜택 메시지로 재방문을 유도하기 적합합니다.",
    Channel.ONSITE_BANNER.value: "사이트 내 배너로 호텔 혜택을 노출하기 적합합니다.",
}


class PromotionReader(Protocol):
    def get_for_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionRecord | None:
        ...


class SegmentDefinitionReader(Protocol):
    def list_active(
        self,
        *,
        project_id: str,
        campaign_id: str | None = None,
        promotion_id: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        ...

    def save_ai_suggested(
        self,
        segments: Sequence[SegmentDefinitionRecord],
    ) -> None:
        ...


class HotelProfileReader(Protocol):
    def list_marketing_profiles(
        self,
        *,
        project_id: str,
    ) -> list[HotelMarketingProfileRecord]:
        ...

    def summarize_user_ids(
        self,
        *,
        project_id: str,
        profile_name: str,
        user_ids: Sequence[str],
    ) -> HotelMarketingProfileRecord | None:
        ...

    def list_booking_training_records(
        self,
        *,
        limit: int = 500,
    ) -> list[BookingTrainingRecord]:
        ...


class PromotionAnalysisWriter(Protocol):
    def save_analysis(self, analysis: PromotionAnalysisWrite) -> None:
        ...

    def save_target_segments(
        self,
        target_segments: Sequence[PromotionTargetSegmentWrite],
    ) -> None:
        ...

    def save_segment_suggestions(
        self,
        suggestions: Sequence[PromotionSegmentSuggestionWrite],
    ) -> None:
        ...

    def get_latest_audience_bindings(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        segment_ids: Sequence[str],
    ) -> list[SegmentSuggestionAudienceBindingRecord]:
        ...


class SegmentVectorPreparer(Protocol):
    def prepare_segment_vector(
        self,
        request: SegmentVectorBuildRequest,
    ) -> SegmentVectorBuildResult:
        ...


class SegmentDefinitionSuggester(Protocol):
    def suggest_segments(
        self,
        *,
        promotion: PromotionRecord,
        segment_instruction: str | None = None,
    ) -> list[SegmentDefinitionRecord]:
        ...


@dataclass(frozen=True)
class PromotionAnalysisResult:
    analysis: PromotionAnalysisWrite
    target_segments: list[PromotionTargetSegmentWrite]
    segment_suggestions: list[PromotionSegmentSuggestionWrite] = field(default_factory=list)


@dataclass(frozen=True)
class NextLoopFocusAnalysisRequest:
    project_id: str
    campaign_id: str
    promotion_id: str
    focus_segment_ids: Sequence[str]
    loop_count: int
    source_promotion_run_id: str
    source_failed_ad_experiment_ids: Sequence[str]
    attempt_no: int = 1
    operator_instruction: str | None = None


@dataclass(frozen=True)
class NextLoopAnalysisContext:
    loop_count: int
    source_promotion_run_id: str
    source_failed_ad_experiment_ids: Sequence[str]
    attempt_no: int


@dataclass(frozen=True)
class SegmentCandidate:
    definition: SegmentDefinitionRecord
    profile: HotelMarketingProfileRecord | None

    @property
    def segment_id(self) -> str:
        return self.definition.segment_id

    @property
    def estimated_size(self) -> int:
        return self.definition.sample_size


class PromotionNotFoundError(Exception):
    pass


class SegmentSelectionError(Exception):
    pass


class PromotionAnalysisService:
    def __init__(
        self,
        *,
        promotion_repository: PromotionReader,
        segment_definition_repository: SegmentDefinitionReader,
        hotel_profile_repository: HotelProfileReader,
        promotion_analysis_repository: PromotionAnalysisWriter,
        segment_vector_service: SegmentVectorPreparer,
        segment_suggester: SegmentDefinitionSuggester | None = None,
        segment_report_generator: SegmentSuggestionReportGenerator | None = None,
        max_default_target_segments: int = MAX_DEFAULT_TARGET_SEGMENTS,
        audience_v2_coordinator: AudienceV2Coordinator | None = None,
        audience_adapter: SegmentDefinitionAudienceAdapter | None = None,
        audience_allocation_service: AudienceAllocationService | None = None,
    ) -> None:
        self._promotion_repository = promotion_repository
        self._segment_definition_repository = segment_definition_repository
        self._hotel_profile_repository = hotel_profile_repository
        self._promotion_analysis_repository = promotion_analysis_repository
        self._segment_vector_service = segment_vector_service
        self._segment_suggester = segment_suggester
        self._segment_report_generator = (
            segment_report_generator
            or DeterministicSegmentSuggestionReportGenerator()
        )
        self._max_default_target_segments = max_default_target_segments
        self._audience_v2_coordinator = audience_v2_coordinator
        self._audience_adapter = audience_adapter or SegmentDefinitionAudienceAdapter()
        self._audience_allocation_service = audience_allocation_service

    @log_context_scope
    def recommend_segments(
        self,
        request: AnalysisRequest,
    ) -> PromotionAnalysisResult:
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": request.project_id,
                "campaignId": request.campaign_id,
                "promotionId": request.promotion_id,
            }
        )
        log.info("started", _analysis_request_log_payload(request))
        response = self._analyze(
            request=request,
            focus_segment_ids=None,
            next_loop_context=None,
            refresh_segment_suggestions=True,
            persist_target_segments=False,
            persist_segment_suggestions=True,
            target_status="planned",
        )
        log.assign_context({"analysisId": response.analysis.analysis_id})
        log.info(
            "completed",
            {
                **_analysis_result_log_payload(response),
                "durationMs": duration_ms(started_at),
            },
        )
        return response

    def analyze(self, request: AnalysisRequest) -> PromotionAnalysisResult:
        """Compatibility alias for the former combined recommendation endpoint."""
        return self.recommend_segments(request)

    @log_context_scope
    def analyze_segments(
        self,
        request: SegmentAnalysisRequest,
    ) -> PromotionAnalysisResult:
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": request.project_id,
                "campaignId": request.campaign_id,
                "promotionId": request.promotion_id,
            }
        )
        log.info("started", _analysis_request_log_payload(request))
        response = self._analyze(
            request=request,
            focus_segment_ids=request.segment_ids,
            next_loop_context=None,
            refresh_segment_suggestions=False,
            persist_target_segments=True,
            persist_segment_suggestions=False,
            target_status="approved",
        )
        log.assign_context({"analysisId": response.analysis.analysis_id})
        log.info(
            "completed",
            {
                **_analysis_result_log_payload(response),
                "durationMs": duration_ms(started_at),
            },
        )
        return response

    @log_context_scope
    def analyze_focus(
        self,
        request: NextLoopFocusAnalysisRequest,
        *,
        target_status: TargetSegmentStatus = "planned",
    ) -> PromotionAnalysisResult:
        if request.attempt_no < 1:
            raise SegmentSelectionError("attempt_no must be at least 1")
        started_at = now_ms()
        log.assign_context(
            {
                "projectId": request.project_id,
                "campaignId": request.campaign_id,
                "promotionId": request.promotion_id,
                "promotionRunId": request.source_promotion_run_id,
            }
        )
        log.info("started", _analysis_request_log_payload(request))
        response = self._analyze(
            request=AnalysisRequest(
                project_id=request.project_id,
                campaign_id=request.campaign_id,
                promotion_id=request.promotion_id,
                operator_instruction=request.operator_instruction,
            ),
            focus_segment_ids=list(request.focus_segment_ids),
            next_loop_context=NextLoopAnalysisContext(
                loop_count=request.loop_count,
                source_promotion_run_id=request.source_promotion_run_id,
                source_failed_ad_experiment_ids=list(
                    request.source_failed_ad_experiment_ids
                ),
                attempt_no=request.attempt_no,
            ),
            refresh_segment_suggestions=False,
            persist_target_segments=True,
            persist_segment_suggestions=False,
            target_status=target_status,
        )
        log.assign_context({"analysisId": response.analysis.analysis_id})
        log.info(
            "completed",
            {
                **_analysis_result_log_payload(response),
                "durationMs": duration_ms(started_at),
            },
        )
        return response

    def _analyze(
        self,
        *,
        request: AnalysisRequest,
        focus_segment_ids: Sequence[str] | None,
        next_loop_context: NextLoopAnalysisContext | None,
        refresh_segment_suggestions: bool,
        persist_target_segments: bool,
        persist_segment_suggestions: bool,
        target_status: TargetSegmentStatus,
    ) -> PromotionAnalysisResult:
        promotion = self._get_promotion(request)
        log.info(
            "promotion_loaded",
            {
                "channel": promotion.channel,
                "goalMetric": promotion.goal_metric,
                "goalBasis": promotion.goal_basis,
            },
        )
        segment_definitions = self._segment_definition_repository.list_active(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
        )
        log.info("segment_definitions_loaded", {"segmentDefinitionCount": len(segment_definitions)})
        suggested_segment_definitions: list[SegmentDefinitionRecord] = []
        if refresh_segment_suggestions:
            suggested_segment_definitions = self._suggest_segment_definitions(
                promotion,
                segment_instruction=request.segment_instruction,
            )
        else:
            log.info(
                "segment_suggestion_refresh_skipped",
                {
                    "reason": (
                        "next_loop_focus"
                        if next_loop_context is not None
                        else "existing_segments_analysis"
                    ),
                    "focusSegmentIds": list(focus_segment_ids or ()),
                },
            )
        if suggested_segment_definitions:
            self._segment_definition_repository.save_ai_suggested(
                suggested_segment_definitions
            )
            segment_definitions = _merge_segment_definitions(
                segment_definitions,
                suggested_segment_definitions,
            )
            log.info(
                "segment_definitions_created",
                {
                    "segmentDefinitionCount": len(suggested_segment_definitions),
                    "segmentIds": [
                        segment.segment_id
                        for segment in suggested_segment_definitions
                    ],
                },
            )
        hotel_profiles = self._hotel_profile_repository.list_marketing_profiles(
            project_id=request.project_id,
        )
        log.info("hotel_profiles_loaded", {"hotelProfileCount": len(hotel_profiles)})
        candidates = self._build_candidates(
            project_id=request.project_id,
            segment_definitions=segment_definitions,
            hotel_profiles=hotel_profiles,
        )
        log.info("segment_candidates_prepared", {"candidateCount": len(candidates)})
        booking_model = self._train_booking_model()
        if booking_model is None:
            log.warn("booking_model_unavailable")
        else:
            log.info("booking_model_trained", {"modelVersion": booking_model.model_version, "trainingSampleCount": booking_model.training_sample_count})
        booking_predictions = _predict_booking_propensity(
            model=booking_model,
            candidates=candidates,
        )
        selection_segment_ids = focus_segment_ids
        if (
            refresh_segment_suggestions
            and suggested_segment_definitions
            and request.segment_instruction
        ):
            selection_segment_ids = [
                segment.segment_id for segment in suggested_segment_definitions
            ]
        elif (
            refresh_segment_suggestions
            and request.segment_instruction
        ):
            raise SegmentSelectionError(
                "no segment candidates matched segment instruction"
            )
        selected_candidates = self._select_candidates(
            promotion=promotion,
            focus_segment_ids=selection_segment_ids,
            candidates=candidates,
            booking_predictions=booking_predictions,
        )
        if not selected_candidates:
            log.warn("segment_candidates_empty", {"candidateCount": len(candidates)})
            raise SegmentSelectionError("no active segment candidates matched analysis request")
        audience_resolutions = {
            candidate.segment_id: self._audience_adapter.resolve(
                segment_id=candidate.segment_id,
                rule_json=candidate.definition.rule_json,
            )
            for candidate in selected_candidates
        }
        contracts = {
            resolution.contract for resolution in audience_resolutions.values()
        }
        if persist_target_segments and len(contracts) > 1:
            raise AudienceSnapshotBindingError(
                "legacy and segment_audience.v1 segments cannot be confirmed together",
                code="segment_audience_contract_mixed",
            )

        confirmation_source_analysis_id: str | None = None
        confirmation_source_snapshot_ids: tuple[str, ...] = ()
        custom_confirmation_segment_ids = {
            candidate.segment_id
            for candidate in selected_candidates
            if (
                audience_resolutions[candidate.segment_id].spec is not None
                and audience_resolutions[
                    candidate.segment_id
                ].spec.is_custom_structured
            )
        }
        if (
            persist_target_segments
            and not refresh_segment_suggestions
            and next_loop_context is None
            and contracts == {SEGMENT_AUDIENCE_CONTRACT}
        ):
            recommendation_segment_ids = [
                candidate.segment_id
                for candidate in selected_candidates
                if candidate.segment_id not in custom_confirmation_segment_ids
            ]
            if recommendation_segment_ids:
                (
                    confirmation_source_analysis_id,
                    confirmation_source_snapshot_ids,
                ) = self._resolve_confirmation_source_batch(
                    promotion=promotion,
                    segment_ids=recommendation_segment_ids,
                )
            custom_spec_fingerprints = tuple(
                f"custom:{candidate.segment_id}:"
                f"{audience_resolutions[candidate.segment_id].spec.spec_hash}"
                for candidate in selected_candidates
                if candidate.segment_id in custom_confirmation_segment_ids
                and audience_resolutions[candidate.segment_id].spec is not None
            )
            confirmation_source_snapshot_ids += custom_spec_fingerprints

        analysis_id = (
            _confirmation_analysis_id(
                promotion_id=promotion.promotion_id,
                source_analysis_id=(
                    confirmation_source_analysis_id or "custom-structured"
                ),
                segment_ids=[
                    candidate.segment_id for candidate in selected_candidates
                ],
                source_snapshot_ids=confirmation_source_snapshot_ids,
                operator_instruction=request.operator_instruction,
            )
            if (
                persist_target_segments
                and not refresh_segment_suggestions
                and next_loop_context is None
                and contracts == {SEGMENT_AUDIENCE_CONTRACT}
            )
            else _analysis_id(
                promotion_id=promotion.promotion_id,
                next_loop_context=next_loop_context,
            )
        )
        analysis = self._build_analysis(
            analysis_id=analysis_id,
            promotion=promotion,
            request=request,
            focus_segment_ids=focus_segment_ids,
            next_loop_context=next_loop_context,
            segment_definitions=segment_definitions,
            candidates=candidates,
            selected_segment_ids=[
                candidate.segment_id for candidate in selected_candidates
            ],
        )

        self._promotion_analysis_repository.save_analysis(analysis)
        log.assign_context({"analysisId": analysis.analysis_id})
        log.info(
            "promotion_analysis_created",
            {
                "status": analysis.status,
                "selectedSegmentCount": len(selected_candidates),
                "hasOperatorInstruction": bool(request.operator_instruction),
                "hasSegmentInstruction": bool(request.segment_instruction),
            },
        )
        v2_recommendation_segments_to_prepare = [
            candidate.definition
            for candidate in selected_candidates
            if audience_resolutions[candidate.segment_id].is_v2
            and refresh_segment_suggestions
            and next_loop_context is None
        ]
        v2_custom_confirmation_segments = [
            candidate.definition
            for candidate in selected_candidates
            if candidate.segment_id in custom_confirmation_segment_ids
            and persist_target_segments
            and not refresh_segment_suggestions
            and next_loop_context is None
        ]
        v2_segments_to_prepare = (
            v2_recommendation_segments_to_prepare
            + v2_custom_confirmation_segments
        )
        v2_segments_to_reuse = [
            candidate.definition
            for candidate in selected_candidates
            if audience_resolutions[candidate.segment_id].is_v2
            and candidate.segment_id not in custom_confirmation_segment_ids
            and persist_target_segments
            and not refresh_segment_suggestions
            and next_loop_context is None
        ]
        v2_segments = v2_segments_to_prepare + v2_segments_to_reuse
        if v2_segments and self._audience_v2_coordinator is None:
            first_segment = v2_segments[0]
            raise SegmentAudienceContractError(
                code="segment_audience_calculator_unavailable",
                segment_id=first_segment.segment_id,
                reason="V2 audience calculator is unavailable",
            )
        prepared_v2 = (
            self._audience_v2_coordinator.prepare_many(
                analysis_id=analysis_id,
                promotion=promotion,
                segments=v2_recommendation_segments_to_prepare,
            )
            if v2_recommendation_segments_to_prepare
            and self._audience_v2_coordinator is not None
            else {}
        )
        allocation_source_analysis_id = confirmation_source_analysis_id or analysis_id
        if (
            v2_custom_confirmation_segments
            and self._audience_v2_coordinator is not None
        ):
            prepared_v2.update(
                self._audience_v2_coordinator.prepare_many(
                    analysis_id=allocation_source_analysis_id,
                    promotion=promotion,
                    segments=v2_custom_confirmation_segments,
                )
            )
            log.info(
                "custom_segment_audience_prepared",
                {
                    "customSegmentCount": len(v2_custom_confirmation_segments),
                    "segmentIds": [
                        segment.segment_id
                        for segment in v2_custom_confirmation_segments
                    ],
                    "sourceAnalysisId": allocation_source_analysis_id,
                },
            )
        v2_confirmation_segments = (
            v2_custom_confirmation_segments + v2_segments_to_reuse
        )
        if v2_confirmation_segments and self._audience_v2_coordinator is not None:
            prepared_v2.update(
                self._allocate_recommendation_audiences(
                    analysis_id=analysis_id,
                    promotion=promotion,
                    segments=v2_confirmation_segments,
                    source_analysis_id=allocation_source_analysis_id,
                )
            )
        target_segments: list[PromotionTargetSegmentWrite] = []
        for rank, candidate in enumerate(selected_candidates):
            resolution = audience_resolutions[candidate.segment_id]
            is_v2_candidate = resolution.is_v2
            should_prepare_v2 = (
                is_v2_candidate
                and (
                    refresh_segment_suggestions
                    or persist_target_segments
                )
                and next_loop_context is None
            )
            audience_v2 = (
                prepared_v2.get(candidate.segment_id)
                if should_prepare_v2
                else None
            )
            segment_vector_id = (
                audience_v2.segment_vector_id
                if audience_v2 is not None
                else self._prepare_segment_vector_id(
                    analysis_id=analysis_id,
                    promotion=promotion,
                    candidate=candidate,
                )
            )
            target_segment = self._build_target_segment(
                analysis_id=analysis_id,
                promotion=promotion,
                candidate=candidate,
                rank=rank,
                operator_instruction=request.operator_instruction,
                status=target_status,
                segment_vector_id=segment_vector_id,
            )
            if next_loop_context is not None and is_v2_candidate:
                retry_rule_json = dict(target_segment.rule_json)
                retry_rule_json.pop("segment_audience_spec", None)
                retry_rule_json["audience_resolution_contract"] = (
                    LEGACY_AUDIENCE_CONTRACT
                )
                retry_rule_json["next_loop_audience_source"] = (
                    "failed_source_assignments"
                )
                target_segment = replace(
                    target_segment,
                    rule_json=retry_rule_json,
                )
            if audience_v2 is not None:
                evidence = dict(target_segment.data_evidence_json)
                candidate_generation_user_count = int(
                    evidence.get("sample_size", target_segment.estimated_size) or 0
                )
                selected_user_ratio = _safe_audience_ratio(
                    audience_v2.selected_user_count,
                    audience_v2.total_eligible_user_count,
                )
                matching_user_ratio = _safe_audience_ratio(
                    audience_v2.matching_user_count,
                    audience_v2.total_eligible_user_count,
                )
                selection_ratio_within_matching = _safe_audience_ratio(
                    audience_v2.selected_user_count,
                    audience_v2.matching_user_count,
                )
                evidence.update(
                    {
                        "candidate_generation_user_count": (
                            candidate_generation_user_count
                        ),
                        "sample_size": audience_v2.selected_user_count,
                        "sample_ratio": selected_user_ratio,
                        "total_eligible_user_count": (
                            audience_v2.total_eligible_user_count
                        ),
                        "matching_user_count": audience_v2.matching_user_count,
                        "selected_user_count": audience_v2.selected_user_count,
                        "matching_user_ratio": matching_user_ratio,
                        "selected_user_ratio": selected_user_ratio,
                        "selection_ratio_within_matching": (
                            selection_ratio_within_matching
                        ),
                        "selection_limited": (
                            audience_v2.selected_user_count
                            < audience_v2.matching_user_count
                        ),
                        "selection_basis": (
                            "hard_predicate_and_exact_cosine"
                        ),
                        "selection_method": audience_v2.selection_method,
                        "estimated_recall": audience_v2.estimated_recall,
                        "recall_lower_bound": audience_v2.recall_lower_bound,
                        "recall_target": audience_v2.recall_target,
                        "selected_user_role": "final_experiment_audience",
                        "vector_version": audience_v2.vector_version,
                        "input_stability": "snapshotted",
                        "audience_snapshot_id": audience_v2.audience_snapshot_id,
                        "vector_generation_id": audience_v2.vector_generation_id,
                        "promotion_exclusion_revision": (
                            audience_v2.promotion_exclusion_revision
                        ),
                        "excluded_user_count": audience_v2.excluded_user_count,
                        "meets_min_sample_size": (
                            audience_v2.meets_min_sample_size
                        ),
                        "targetable": audience_v2.selected_user_count > 0,
                        "audience_status": (
                            "no_eligible_audience"
                            if audience_v2.selected_user_count == 0
                            else (
                                "targetable"
                                if audience_v2.meets_min_sample_size
                                else "insufficient_sample"
                            )
                        ),
                        "audience_resolution_contract": (
                            SEGMENT_AUDIENCE_CONTRACT
                        ),
                    }
                )
                target_segment = replace(
                    target_segment,
                    estimated_size=audience_v2.selected_user_count,
                    data_evidence_json=evidence,
                    audience_snapshot_id=audience_v2.audience_snapshot_id,
                    allocation_plan_id=audience_v2.allocation_plan_id,
                )
            target_segments.append(target_segment)
        segment_suggestions: list[PromotionSegmentSuggestionWrite] = []
        if persist_segment_suggestions:
            segment_suggestions = self._build_segment_suggestions(
                analysis_id=analysis_id,
                promotion=promotion,
                target_segments=target_segments,
                candidates=selected_candidates,
                booking_predictions=booking_predictions,
                booking_model=booking_model,
                report_generator=(
                    DeterministicSegmentSuggestionReportGenerator()
                    if request.segment_instruction
                    else self._segment_report_generator
                ),
            )
        if persist_target_segments:
            self._promotion_analysis_repository.save_target_segments(target_segments)
            log.info(
                "promotion_target_segments_created",
                {
                    "segmentIds": [segment.segment_id for segment in target_segments],
                    "targetSegmentCount": len(target_segments),
                },
            )
        if persist_segment_suggestions:
            self._promotion_analysis_repository.save_segment_suggestions(segment_suggestions)
            if v2_recommendation_segments_to_prepare:
                if self._audience_allocation_service is None:
                    raise SegmentAudienceContractError(
                        code="segment_audience_exclusion_contract_missing",
                        segment_id=v2_recommendation_segments_to_prepare[0].segment_id,
                        reason="audience allocation service is unavailable",
                    )
                self._audience_allocation_service.refresh_recommendation_previews(
                    analysis_id=analysis_id,
                    project_id=promotion.project_id,
                    campaign_id=promotion.campaign_id,
                    promotion_id=promotion.promotion_id,
                )
            log.info(
                "promotion_segment_suggestions_created",
                {
                    "segmentIds": [
                        suggestion.segment_id for suggestion in segment_suggestions
                    ],
                    "segmentSuggestionCount": len(segment_suggestions),
                },
            )
        return PromotionAnalysisResult(
            analysis=analysis,
            target_segments=target_segments,
            segment_suggestions=segment_suggestions,
        )

    def _allocate_recommendation_audiences(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        segments: Sequence[SegmentDefinitionRecord],
        source_analysis_id: str | None = None,
    ) -> dict[str, Any]:
        if self._audience_allocation_service is None:
            raise SegmentAudienceContractError(
                code="segment_audience_exclusion_contract_missing",
                segment_id=segments[0].segment_id,
                reason="audience allocation service is unavailable",
            )
        result = self._audience_allocation_service.confirm_selection(
            confirmation_analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_ids=[segment.segment_id for segment in segments],
            min_sample_size=promotion.min_sample_size,
            source_analysis_id=source_analysis_id,
        )
        assert self._audience_v2_coordinator is not None
        prepared: dict[str, AudienceV2Preparation] = {}
        for segment in segments:
            allocation = result.allocations[segment.segment_id]
            bound = self._audience_v2_coordinator.prepare(
                analysis_id=analysis_id,
                promotion=promotion,
                segment=segment,
                audience_snapshot_id=allocation.final_snapshot_id,
            )
            prepared[segment.segment_id] = replace(
                bound,
                source_audience_snapshot_id=allocation.source_snapshot_id,
                allocation_plan_id=allocation.allocation_plan_id,
            )
        return prepared

    def _resolve_confirmation_source_batch(
        self,
        *,
        promotion: PromotionRecord,
        segment_ids: Sequence[str],
    ) -> tuple[str, tuple[str, ...]]:
        expected = set(segment_ids)
        bindings = self._promotion_analysis_repository.get_latest_audience_bindings(
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_ids=segment_ids,
        )
        by_segment = {binding.segment_id: binding for binding in bindings}
        if set(by_segment) != expected or any(
            binding.audience_snapshot_id is None
            for binding in by_segment.values()
        ):
            missing = sorted(expected - set(by_segment))
            raise AudienceSnapshotBindingError(
                "selected V2 segment requires a completed recommendation snapshot",
                code="segment_audience_snapshot_binding_required",
                segment_id=missing[0] if missing else sorted(expected)[0],
            )
        source_analysis_ids = {
            binding.analysis_id for binding in by_segment.values()
        }
        if len(source_analysis_ids) != 1:
            raise AudienceSnapshotBindingError(
                "selected V2 segments must come from the same recommendation analysis",
                code="segment_audience_source_batch_mismatch",
            )
        return (
            next(iter(source_analysis_ids)),
            tuple(
                str(by_segment[segment_id].audience_snapshot_id)
                for segment_id in sorted(expected)
            ),
        )

    def _train_booking_model(self) -> BookingPropensityModel | None:
        training_records = self._hotel_profile_repository.list_booking_training_records()
        return train_booking_propensity_model(training_records)

    def _suggest_segment_definitions(
        self,
        promotion: PromotionRecord,
        *,
        segment_instruction: str | None = None,
    ) -> list[SegmentDefinitionRecord]:
        if self._segment_suggester is None:
            return []
        return self._segment_suggester.suggest_segments(
            promotion=promotion,
            segment_instruction=segment_instruction,
        )

    def _build_segment_suggestions(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        target_segments: Sequence[PromotionTargetSegmentWrite],
        candidates: Sequence[SegmentCandidate],
        booking_predictions: Mapping[str, BookingPropensityPrediction],
        booking_model: BookingPropensityModel | None,
        report_generator: SegmentSuggestionReportGenerator,
    ) -> list[PromotionSegmentSuggestionWrite]:
        jobs = [
            (
                rank,
                target_segment,
                candidates[rank],
                copy_context(),
            )
            for rank, target_segment in enumerate(target_segments)
        ]
        if len(jobs) <= 1:
            return [
                self._build_segment_suggestion(
                    analysis_id=analysis_id,
                    promotion=promotion,
                    target_segment=target_segment,
                    candidate=candidate,
                    booking_prediction=booking_predictions.get(candidate.segment_id),
                    booking_model=booking_model,
                    report_generator=report_generator,
                    rank=rank,
                )
                for rank, target_segment, candidate, _context in jobs
            ]

        started_at = now_ms()
        worker_count = min(len(jobs), MAX_SEGMENT_REPORT_WORKERS)
        log.info(
            "segment_suggestion_reports_started",
            {"candidateCount": len(jobs), "workerCount": worker_count},
        )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="loop-ad-segment-report",
        ) as executor:
            futures = [
                executor.submit(
                    context.run,
                    self._build_segment_suggestion,
                    analysis_id=analysis_id,
                    promotion=promotion,
                    target_segment=target_segment,
                    candidate=candidate,
                    booking_prediction=booking_predictions.get(candidate.segment_id),
                    booking_model=booking_model,
                    report_generator=report_generator,
                    rank=rank,
                )
                for rank, target_segment, candidate, context in jobs
            ]
            suggestions = [future.result() for future in futures]
        log.info(
            "segment_suggestion_reports_completed",
            {
                "candidateCount": len(suggestions),
                "workerCount": worker_count,
                "durationMs": duration_ms(started_at),
            },
        )
        return suggestions

    def _get_promotion(self, request: AnalysisRequest) -> PromotionRecord:
        promotion = self._promotion_repository.get_for_analysis(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
        )
        if promotion is None:
            log.warn("promotion_not_found", {"projectId": request.project_id, "campaignId": request.campaign_id, "promotionId": request.promotion_id})
            raise PromotionNotFoundError(
                f"promotion not found for analysis: {request.promotion_id}"
            )
        return promotion

    def _build_candidates(
        self,
        *,
        project_id: str,
        segment_definitions: Sequence[SegmentDefinitionRecord],
        hotel_profiles: Sequence[HotelMarketingProfileRecord],
    ) -> dict[str, SegmentCandidate]:
        profiles_by_segment = {profile.profile_name: profile for profile in hotel_profiles}
        candidates: dict[str, SegmentCandidate] = {}
        for segment in segment_definitions:
            profile = profiles_by_segment.get(segment.segment_id)
            if profile is None and segment.source == "ai_suggested":
                profile = self._summarize_ai_segment_profile(
                    project_id=project_id,
                    segment=segment,
                )
            candidates[segment.segment_id] = SegmentCandidate(
                definition=segment,
                profile=profile,
            )
        return candidates

    def _summarize_ai_segment_profile(
        self,
        *,
        project_id: str,
        segment: SegmentDefinitionRecord,
    ) -> HotelMarketingProfileRecord | None:
        candidate_user_ids = _candidate_user_ids(segment.rule_json)
        if not candidate_user_ids:
            return None
        return self._hotel_profile_repository.summarize_user_ids(
            project_id=project_id,
            profile_name=segment.segment_id,
            user_ids=candidate_user_ids,
        )

    def _select_candidates(
        self,
        *,
        promotion: PromotionRecord,
        focus_segment_ids: Sequence[str] | None,
        candidates: Mapping[str, SegmentCandidate],
        booking_predictions: Mapping[str, BookingPropensityPrediction],
    ) -> list[SegmentCandidate]:
        focus_segment_ids = _focus_segment_ids(focus_segment_ids)
        if focus_segment_ids is not None:
            missing_segment_ids = [
                segment_id
                for segment_id in focus_segment_ids
                if segment_id not in candidates
            ]
            if missing_segment_ids:
                log.warn("focus_segments_invalid", {"missingSegmentIds": missing_segment_ids})
                raise SegmentSelectionError(
                    "focus_segment_ids must match active segment definitions"
                )
            return [candidates[segment_id] for segment_id in focus_segment_ids]

        ordered_ids: list[str] = []
        for candidate in sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.estimated_size, candidate.segment_id),
        ):
            if self._is_related_custom_segment(candidate.definition, promotion):
                ordered_ids.append(candidate.segment_id)

        ordered_ids.extend(
            candidate.segment_id
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: _ai_candidate_sort_key(
                    candidate,
                    booking_predictions,
                ),
            )
            if candidate.definition.source == "ai_suggested"
        )
        ordered_ids.extend(DEFAULT_SEGMENT_IDS_BY_CHANNEL.get(promotion.channel, ()))
        ordered_ids.extend(
            profile.profile_name
            for profile in sorted(
                (candidate.profile for candidate in candidates.values() if candidate.profile),
                key=lambda profile: (
                    -int(profile.profile_json.get("event_count", 0)),
                    profile.profile_name,
                ),
            )
        )
        ordered_ids.extend(
            candidate.segment_id
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: (-candidate.estimated_size, candidate.segment_id),
            )
            if candidate.definition.source in {"system_default", "ai_suggested"}
        )

        selected: list[SegmentCandidate] = []
        seen: set[str] = set()
        for segment_id in ordered_ids:
            if segment_id in seen or segment_id not in candidates:
                continue
            candidate = candidates[segment_id]
            if not _has_recommendable_audience(candidate):
                continue
            selected.append(candidate)
            seen.add(segment_id)
            if len(selected) == self._max_default_target_segments:
                break
        return selected

    def _is_related_custom_segment(
        self,
        segment: SegmentDefinitionRecord,
        promotion: PromotionRecord,
    ) -> bool:
        if segment.source not in CUSTOM_SEGMENT_SOURCES:
            return False

        searchable = " ".join(
            [
                segment.segment_id,
                segment.segment_name,
                segment.natural_language_query or "",
                segment.generated_sql or "",
                str(segment.rule_json),
                str(segment.profile_json),
                promotion.message_brief or "",
                promotion.landing_url or "",
            ]
        ).lower()
        goal_terms = RELATED_TERMS_BY_GOAL.get(promotion.goal_metric, ())
        channel_terms = (promotion.channel.replace("_", " "), promotion.channel)
        return any(term in searchable for term in (*goal_terms, *channel_terms))

    def _build_target_segment(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        candidate: SegmentCandidate,
        rank: int,
        operator_instruction: str | None,
        status: TargetSegmentStatus,
        segment_vector_id: str,
    ) -> PromotionTargetSegmentWrite:
        segment = candidate.definition
        profile_json = dict(segment.profile_json)
        content_brief_json = self._build_content_brief_json(
            analysis_id=analysis_id,
            promotion=promotion,
            candidate=candidate,
            profile_json=profile_json,
            operator_instruction=operator_instruction,
            segment_vector_id=segment_vector_id,
        )
        return PromotionTargetSegmentWrite(
            analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_id=segment.segment_id,
            segment_name=segment.segment_name,
            rule_json=segment.rule_json,
            profile_json=profile_json,
            content_brief_json=content_brief_json,
            data_evidence_json=self._build_data_evidence_json(candidate),
            segment_vector_id=segment_vector_id,
            estimated_size=max(segment.sample_size, 0),
            priority=self._priority_for_segment(
                estimated_size=segment.sample_size,
                min_sample_size=promotion.min_sample_size,
                rank=rank,
            ),
            status=status,
        )

    def _build_segment_suggestion(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        target_segment: PromotionTargetSegmentWrite,
        candidate: SegmentCandidate,
        booking_prediction: BookingPropensityPrediction | None,
        booking_model: BookingPropensityModel | None,
        report_generator: SegmentSuggestionReportGenerator,
        rank: int,
    ) -> PromotionSegmentSuggestionWrite:
        segment = candidate.definition
        primary_signals = _display_primary_signals(segment)
        display_copy = _display_copy(
            promotion=promotion,
            target_segment=target_segment,
            primary_signals=primary_signals,
        )
        ai_score_details = _ai_score_details(segment)
        score_json = {
            "rank": rank + 1,
            "estimated_size": target_segment.estimated_size,
            "priority": target_segment.priority,
            "cluster_score": _ai_segment_score(segment),
            **ai_score_details,
            "booking_propensity_score": (
                booking_prediction.probability if booking_prediction else None
            ),
            "booking_propensity_model": (
                booking_prediction.model_version
                if booking_prediction
                else "unavailable"
            ),
            "booking_propensity_training_sample_count": (
                booking_prediction.training_sample_count
                if booking_prediction
                else 0
            ),
        }
        reason_json = {
            "channel": promotion.channel,
            "goal_metric": promotion.goal_metric,
            "segment_source": segment.source,
            "primary_signals": [signal["key"] for signal in primary_signals],
            "promotion_vector_basis": segment.profile_json.get(
                "promotion_vector_basis",
                {},
            ),
            "has_hotel_profile": candidate.profile is not None,
            "ml_model": (
                booking_model.model_version if booking_model else "unavailable"
            ),
            "ml_features": (
                dict(booking_prediction.feature_values)
                if booking_prediction
                else {}
            ),
        }
        metadata_json: dict[str, Any] = {
            "segment_name": target_segment.segment_name,
            "segment_vector_id": target_segment.segment_vector_id,
            "content_brief": target_segment.content_brief_json,
            "data_evidence": target_segment.data_evidence_json,
            "promotion_vector_basis": segment.profile_json.get(
                "promotion_vector_basis",
                {},
            ),
            "promotion_matched_features": segment.profile_json.get(
                "promotion_matched_features",
                [],
            ),
            "display_copy": display_copy,
        }
        if segment.source == "ai_suggested":
            ai_report = report_generator.generate_report(
                SegmentSuggestionReportInput(
                    promotion=promotion,
                    segment=segment,
                    target_segment=target_segment,
                    display_copy=display_copy,
                    primary_signals=primary_signals,
                    score_json=score_json,
                    reason_json=reason_json,
                )
            )
            if _is_raw_event_intent_segment(segment):
                display_copy = _display_copy_from_report(
                    display_copy=display_copy,
                    report=ai_report,
                )
                metadata_json["display_copy"] = display_copy
            metadata_json["ai_report"] = ai_report
        return PromotionSegmentSuggestionWrite(
            suggestion_id=_suggestion_id(
                analysis_id=analysis_id,
                segment_id=segment.segment_id,
            ),
            analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_id=segment.segment_id,
            suggested_rank=rank + 1,
            suggestion_source=_suggestion_source(segment),
            status="suggested",
            score_json=score_json,
            reason_json=reason_json,
            metadata_json=metadata_json,
            audience_snapshot_id=target_segment.audience_snapshot_id,
        )

    def _build_content_brief_json(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        candidate: SegmentCandidate,
        profile_json: Mapping[str, Any],
        operator_instruction: str | None,
        segment_vector_id: str,
    ) -> dict[str, Any]:
        segment = candidate.definition
        message_direction, keywords = SEGMENT_CONTENT_HINTS.get(
            segment.segment_id,
            (
                "Use a hotel booking message tailored to this segment.",
                ("hotel booking", "seasonal stay", "booking benefit"),
            ),
        )
        score_components = profile_json.get("score_components")
        if not isinstance(score_components, Mapping):
            score_components = None
        audience_evidence: dict[str, Any] = {
            "primary_signals": profile_json.get("primary_signals"),
            "score_components": score_components,
            "promotion_vector_basis": profile_json.get(
                "promotion_vector_basis"
            ),
            "promotion_matched_features": profile_json.get(
                "promotion_matched_features"
            ),
        }
        return build_content_brief_v2(
            analysis_id=analysis_id,
            segment_snapshot={
                "segment_id": segment.segment_id,
                "segment_name": segment.segment_name,
                "segment_source": segment.source,
                "estimated_size": max(segment.sample_size, 0),
                "segment_vector_id": segment_vector_id,
            },
            promotion_context={
                "channel": promotion.channel,
                "goal_metric": promotion.goal_metric,
                "goal_basis": promotion.goal_basis,
                "goal_target_value": _json_decimal(promotion.goal_target_value),
                "message_brief": promotion.message_brief,
                "landing_url": promotion.landing_url,
            },
            fallback_message_direction=message_direction,
            fallback_keywords=keywords,
            audience_evidence=audience_evidence,
            hotel_profile=(
                dict(candidate.profile.profile_json)
                if candidate.profile is not None
                else None
            ),
            operator_instruction=operator_instruction,
        )

    def _build_data_evidence_json(
        self,
        candidate: SegmentCandidate,
    ) -> dict[str, Any]:
        segment = candidate.definition
        evidence: dict[str, Any] = {
            "source": segment.source,
            "sample_size": segment.sample_size,
            "sample_ratio": _json_decimal(segment.sample_ratio),
            "total_eligible_user_count": segment.total_eligible_user_count,
        }
        for key in (
            "promotion_cluster_similarity",
            "recommendation_score",
            "cluster_quality_score",
            "sample_size_score",
            "candidate_type",
            "strategy_role",
            "performance_estimate",
            "matched_conditions",
            "missing_conditions",
            "signal_metrics",
            "score_components",
        ):
            value = segment.profile_json.get(key)
            if value is not None:
                evidence[key] = value
        if candidate.profile is not None:
            evidence["hotel_profile"] = dict(candidate.profile.profile_json)
        return evidence

    def _priority_for_segment(
        self,
        *,
        estimated_size: int,
        min_sample_size: int,
        rank: int,
    ) -> str:
        if estimated_size < min_sample_size:
            return "low"
        if rank < 2:
            return "high"
        return "medium"

    def _build_analysis(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        request: AnalysisRequest,
        focus_segment_ids: Sequence[str] | None,
        next_loop_context: NextLoopAnalysisContext | None,
        segment_definitions: Sequence[SegmentDefinitionRecord],
        candidates: Mapping[str, SegmentCandidate],
        selected_segment_ids: Sequence[str],
    ) -> PromotionAnalysisWrite:
        focus_segment_ids = _focus_segment_ids(focus_segment_ids)
        input_snapshot_json: dict[str, Any] = {
            "promotion": _promotion_snapshot(promotion),
            "available_segment_definitions": [
                _segment_definition_snapshot(segment)
                for segment in segment_definitions
            ],
            "focus_segment_ids": focus_segment_ids,
            "operator_instruction": request.operator_instruction,
            "segment_instruction": request.segment_instruction,
        }
        if next_loop_context is not None:
            input_snapshot_json["next_loop"] = {
                "loop_count": next_loop_context.loop_count,
                "source_promotion_run_id": next_loop_context.source_promotion_run_id,
                "source_failed_ad_experiment_ids": list(
                    next_loop_context.source_failed_ad_experiment_ids
                ),
                "attempt_no": next_loop_context.attempt_no,
            }

        output_json: dict[str, Any] = {
            "selected_segment_ids": list(selected_segment_ids),
            "target_segment_count": len(selected_segment_ids),
        }
        return PromotionAnalysisWrite(
            analysis_id=analysis_id,
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            status=AnalysisStatus.COMPLETED.value,
            focus_segment_ids_json=focus_segment_ids,
            operator_instruction=request.operator_instruction,
            input_snapshot_json=input_snapshot_json,
            profile_summary_json={
                "total_eligible_users": _total_eligible_users(segment_definitions),
                "candidate_segment_count": len(candidates),
                "selected_segment_count": len(selected_segment_ids),
                "selection_mode": "focus" if focus_segment_ids else "default",
                "reason": _analysis_reason(focus_segment_ids),
            },
            output_json=output_json,
        )

    def _prepare_segment_vector_id(
        self,
        *,
        analysis_id: str,
        promotion: PromotionRecord,
        candidate: SegmentCandidate,
    ) -> str:
        result = self._segment_vector_service.prepare_segment_vector(
            SegmentVectorBuildRequest(
                project_id=promotion.project_id,
                promotion_id=promotion.promotion_id,
                analysis_id=analysis_id,
                segment_id=candidate.segment_id,
                candidate_user_ids=_candidate_user_ids(candidate.definition.rule_json),
            )
        )
        return result.segment_vector_id


def _predict_booking_propensity(
    *,
    model: BookingPropensityModel | None,
    candidates: Mapping[str, SegmentCandidate],
) -> dict[str, BookingPropensityPrediction]:
    if model is None:
        return {}
    return {
        candidate.segment_id: prediction
        for candidate in candidates.values()
        if (prediction := model.predict_profile(candidate.profile)) is not None
    }


def _ai_candidate_sort_key(
    candidate: SegmentCandidate,
    booking_predictions: Mapping[str, BookingPropensityPrediction],
) -> tuple[float, float, int, str]:
    prediction = booking_predictions.get(candidate.segment_id)
    propensity_score = prediction.probability if prediction is not None else -1.0
    return (
        -propensity_score,
        -_ai_segment_score(candidate.definition),
        -candidate.estimated_size,
        candidate.segment_id,
    )


def _has_recommendable_audience(candidate: SegmentCandidate) -> bool:
    return candidate.estimated_size > 0


def _promotion_snapshot(promotion: PromotionRecord) -> dict[str, Any]:
    return {
        "project_id": promotion.project_id,
        "campaign_id": promotion.campaign_id,
        "promotion_id": promotion.promotion_id,
        "channel": promotion.channel,
        "goal_metric": promotion.goal_metric,
        "goal_target_value": _json_decimal(promotion.goal_target_value),
        "goal_basis": promotion.goal_basis,
        "min_sample_size": promotion.min_sample_size,
        "landing_url": promotion.landing_url,
        "message_brief": promotion.message_brief,
    }


def _segment_definition_snapshot(segment: SegmentDefinitionRecord) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "campaign_id": segment.campaign_id,
        "promotion_id": segment.promotion_id,
        "segment_name": segment.segment_name,
        "source": segment.source,
        "sample_size": segment.sample_size,
        "total_eligible_user_count": segment.total_eligible_user_count,
        "sample_ratio": _json_decimal(segment.sample_ratio),
        "status": segment.status,
    }


def _merge_segment_definitions(
    stored_segments: Sequence[SegmentDefinitionRecord],
    suggested_segments: Sequence[SegmentDefinitionRecord],
) -> list[SegmentDefinitionRecord]:
    merged = {
        segment.segment_id: segment
        for segment in stored_segments
        if segment.source != "ai_suggested"
    }
    for segment in suggested_segments:
        merged[segment.segment_id] = segment
    return list(merged.values())


def _analysis_id(
    *,
    promotion_id: str,
    next_loop_context: NextLoopAnalysisContext | None,
) -> str:
    if next_loop_context is None:
        suffix = f"_run_{uuid.uuid4().hex[:8]}"
        max_promotion_id_length = 100 - len("analysis_") - len(suffix)
        bounded_promotion_id = (
            promotion_id[:max_promotion_id_length].rstrip("_") or "promotion"
        )
        return f"analysis_{bounded_promotion_id}{suffix}"
    return _bounded_next_loop_lineage_id(
        prefix="analysis",
        promotion_id=promotion_id,
        loop_count=next_loop_context.loop_count,
        source_promotion_run_id=next_loop_context.source_promotion_run_id,
        attempt_no=next_loop_context.attempt_no,
    )


def _confirmation_analysis_id(
    *,
    promotion_id: str,
    source_analysis_id: str,
    segment_ids: Sequence[str],
    source_snapshot_ids: Sequence[str],
    operator_instruction: str | None,
) -> str:
    canonical = {
        "promotion_id": promotion_id,
        "source_analysis_id": source_analysis_id,
        "segment_ids": sorted(set(segment_ids)),
        "source_snapshot_ids": sorted(set(source_snapshot_ids)),
        "operator_instruction": operator_instruction,
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    promotion_slug = _slug_from_promotion_id(promotion_id)[:48].rstrip("_")
    return f"analysis_{promotion_slug or 'promotion'}_confirm_{digest}"


def _bounded_next_loop_lineage_id(
    *,
    prefix: str,
    promotion_id: str,
    loop_count: int,
    source_promotion_run_id: str,
    attempt_no: int = 1,
) -> str:
    lineage_digest = hashlib.sha256(
        source_promotion_run_id.encode("utf-8")
    ).hexdigest()[:12]
    suffix = f"_loop_{loop_count}_{lineage_digest}"
    if attempt_no > 1:
        suffix = f"{suffix}_attempt_{attempt_no}"
    max_slug_length = 100 - len(prefix) - len(suffix) - 1
    promotion_slug = _slug_from_promotion_id(promotion_id)
    promotion_slug = promotion_slug[:max_slug_length].rstrip("_") or "promotion"
    return f"{prefix}_{promotion_slug}{suffix}"


def _slug_from_promotion_id(promotion_id: str) -> str:
    slug = promotion_id.removeprefix("promo_")
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", slug).strip("_")
    return slug or "promotion"


def _focus_segment_ids(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    cleaned = [str(value).strip() for value in values]
    if not cleaned:
        raise SegmentSelectionError(
            "focus_segment_ids must contain at least one segment when provided"
        )
    if any(not value for value in cleaned):
        raise SegmentSelectionError("focus_segment_ids must not contain empty values")
    if len(set(cleaned)) != len(cleaned):
        raise SegmentSelectionError("focus_segment_ids must not contain duplicates")
    return cleaned


def _display_primary_signals(segment: SegmentDefinitionRecord) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    seen: set[str] = set()
    signal_chips = segment.profile_json.get("signal_chips")
    if isinstance(signal_chips, Sequence) and not isinstance(signal_chips, str):
        for chip in signal_chips:
            chip_text = str(chip).strip()
            if not chip_text:
                continue
            signal_key = _signal_key_from_chip(chip_text)
            if signal_key in seen:
                continue
            signals.append({"key": signal_key, "chip": chip_text})
            seen.add(signal_key)

    matched_features = segment.profile_json.get("promotion_matched_features")
    if isinstance(matched_features, Sequence) and not isinstance(matched_features, str):
        for feature in matched_features:
            signal = _signal_from_feature(str(feature))
            if signal is None or signal["key"] in seen:
                continue
            signals.append(signal)
            seen.add(signal["key"])

    top_features = segment.profile_json.get("top_common_features")
    if isinstance(top_features, Sequence) and not isinstance(top_features, str):
        for feature in top_features:
            signal = _signal_from_feature(str(feature))
            if signal is None or signal["key"] in seen:
                continue
            signals.append(signal)
            seen.add(signal["key"])

    default_signal = DEFAULT_SIGNAL_COPY_BY_SEGMENT_ID.get(segment.segment_id)
    if default_signal is not None and default_signal[0] not in seen:
        signals.append({"key": default_signal[0], "chip": default_signal[1]})
        seen.add(default_signal[0])

    if not signals:
        signals.append({"key": "hotel_booking_interest", "chip": "호텔 예약 관심"})
    return signals[:3]


def _signal_key_from_chip(chip: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "_", chip).strip("_") or "signal"


def _signal_from_feature(feature: str) -> dict[str, str] | None:
    signal_copy = SIGNAL_COPY_BY_FEATURE.get(feature)
    if signal_copy is not None:
        return {"key": signal_copy["key"], "chip": signal_copy["chip"]}
    if feature.startswith("Hotel page path bucket"):
        return {"key": "hotel_path_pattern", "chip": "탐색 경로"}
    if feature.startswith("Hotel cluster bucket"):
        return {"key": "hotel_cluster_affinity", "chip": "숙소 취향"}
    if feature.startswith("Hotel market bucket"):
        return {"key": "hotel_market_affinity", "chip": "지역 선호"}
    return None


def _display_copy(
    *,
    promotion: PromotionRecord,
    target_segment: PromotionTargetSegmentWrite,
    primary_signals: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    signal_keys = tuple(signal["key"] for signal in primary_signals)
    signal_chips = [signal["chip"] for signal in primary_signals]
    evidence = target_segment.data_evidence_json
    sample_size = int(evidence.get("sample_size", target_segment.estimated_size) or 0)
    total_users = int(evidence.get("total_eligible_user_count", 0) or 0)
    sample_ratio = _format_percent(evidence.get("sample_ratio", 0))
    raw_display_copy = target_segment.profile_json.get("display_copy")
    if isinstance(raw_display_copy, Mapping):
        display_copy = dict(raw_display_copy)
        display_copy.setdefault("title", target_segment.segment_name)
        strategy_role = target_segment.profile_json.get(
            "strategy_role",
            target_segment.profile_json.get("rank_role"),
        )
        if strategy_role:
            display_copy.setdefault("strategy_role", strategy_role)
        display_copy.setdefault(
            "audience_summary",
            f"분석 대상 {total_users}명 중 {sample_size}명 · {sample_ratio}",
        )
        display_copy.setdefault("signal_chips", signal_chips)
        display_copy.setdefault(
            "reason",
            GOAL_REASON_COPY.get(
                promotion.goal_metric,
                "호텔 예약 관심 행동이 확인된 고객군입니다.",
            ),
        )
        display_copy.setdefault(
            "action_hint",
            CHANNEL_ACTION_COPY.get(
                promotion.channel,
                "프로모션 메시지의 우선 타겟으로 적합합니다.",
            ),
        )
        if _is_v2_audience_evidence(evidence):
            display_copy["audience_summary"] = _v2_audience_summary(evidence)
            display_copy["audience"] = _v2_display_audience(evidence)
        return display_copy
    display_copy = {
        "title": _display_title(signal_keys),
        "audience_summary": (
            f"분석 대상 {total_users}명 중 {sample_size}명 · {sample_ratio}"
        ),
        "signal_chips": signal_chips,
        "reason": GOAL_REASON_COPY.get(
            promotion.goal_metric,
            "호텔 예약 관심 행동이 확인된 고객군입니다.",
        ),
        "action_hint": CHANNEL_ACTION_COPY.get(
            promotion.channel,
            "프로모션 메시지의 우선 타겟으로 적합합니다.",
        ),
    }
    strategy_role = target_segment.profile_json.get(
        "strategy_role",
        target_segment.profile_json.get("rank_role"),
    )
    if strategy_role:
        display_copy["strategy_role"] = strategy_role
    if _is_v2_audience_evidence(evidence):
        display_copy["audience_summary"] = _v2_audience_summary(evidence)
        display_copy["audience"] = _v2_display_audience(evidence)
    return display_copy


def _is_v2_audience_evidence(evidence: Mapping[str, Any]) -> bool:
    return (
        evidence.get("audience_resolution_contract")
        == SEGMENT_AUDIENCE_CONTRACT
        and evidence.get("selected_user_role") == "final_experiment_audience"
    )


def _v2_audience_summary(evidence: Mapping[str, Any]) -> str:
    total_eligible_user_count = int(
        evidence.get("total_eligible_user_count", 0) or 0
    )
    matching_user_count = int(evidence.get("matching_user_count", 0) or 0)
    selected_user_count = int(evidence.get("selected_user_count", 0) or 0)
    return (
        f"분석 가능 사용자 {total_eligible_user_count}명 · "
        f"행동 조건 부합 {matching_user_count}명 · "
        f"실험 대상 사용자 {selected_user_count}명"
    )


def _v2_display_audience(evidence: Mapping[str, Any]) -> dict[str, Any]:
    total_eligible_user_count = int(
        evidence.get("total_eligible_user_count", 0) or 0
    )
    matching_user_count = int(evidence.get("matching_user_count", 0) or 0)
    selected_user_count = int(evidence.get("selected_user_count", 0) or 0)
    return {
        "total_eligible_user_count": total_eligible_user_count,
        "matching_user_count": matching_user_count,
        "selected_user_count": selected_user_count,
        "matching_user_ratio": _safe_audience_ratio(
            matching_user_count,
            total_eligible_user_count,
        ),
        "selected_user_ratio": _safe_audience_ratio(
            selected_user_count,
            total_eligible_user_count,
        ),
        "selection_ratio_within_matching": _safe_audience_ratio(
            selected_user_count,
            matching_user_count,
        ),
        "selection_limited": selected_user_count < matching_user_count,
        "selection_basis": "hard_predicate_and_exact_cosine",
        "selected_user_role": "final_experiment_audience",
    }


def _safe_audience_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(numerator, 0) / denominator, 6)


def _is_raw_event_intent_segment(segment: SegmentDefinitionRecord) -> bool:
    return (
        segment.rule_json.get("source") == "raw_event_intent"
        or segment.profile_json.get("source") == "raw_event_intent"
    )


def _display_copy_from_report(
    *,
    display_copy: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    enhanced = dict(display_copy)
    why_recommended = _text_list(report.get("why_recommended"))
    if why_recommended:
        enhanced["reason"] = why_recommended[0]
    elif summary := _text_value(report.get("summary")):
        enhanced["reason"] = summary
    if (
        "strength_summary" not in enhanced
        and (strengths := _text_list(report.get("candidate_strengths")))
    ):
        enhanced["strength_summary"] = strengths[0]
    if (
        "tradeoff_summary" not in enhanced
        and (
            considerations := _text_list(
                report.get("selection_considerations")
            )
        )
    ):
        enhanced["tradeoff_summary"] = considerations[0]
    if action_hint := _text_value(report.get("action_hint")):
        enhanced["action_hint"] = action_hint
    return enhanced


def _text_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_list(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [text for item in value if (text := _text_value(item))]


def _display_title(signal_keys: Sequence[str]) -> str:
    key_set = set(signal_keys)
    if {"booking_conversion_ready", "promotion_engaged"}.issubset(key_set):
        return "예약 가능성이 높은 프로모션 반응 고객"
    if {"campaign_redirect", "promotion_engaged"}.issubset(key_set):
        return "캠페인 링크 반응이 높은 고객"
    if {"campaign_landing", "promotion_engaged"}.issubset(key_set):
        return "캠페인 랜딩 후 관심 고객"
    if {"hotel_browsing", "promotion_engaged"}.issubset(key_set):
        return "호텔 탐색이 활발한 프로모션 반응 고객"
    if {"hotel_search", "promotion_engaged"}.issubset(key_set):
        return "숙소 검색이 활발한 프로모션 반응 고객"

    for key in signal_keys:
        title = SEGMENT_TITLE_BY_SIGNAL.get((key,))
        if title is not None:
            return title
    return "호텔 예약 관심 고객"


def _format_percent(value: Any) -> str:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        ratio = 0.0
    return f"{ratio * 100:g}%"


def _analysis_reason(focus_segment_ids: Sequence[str] | None) -> str:
    if focus_segment_ids:
        return "Selected next-loop focus_segment_ids for failed segment re-analysis."
    return (
        "Selected hotel audience segments by channel, goal metric, "
        "and active segment definitions."
    )


def _ai_segment_score(segment: SegmentDefinitionRecord) -> float:
    raw_score = segment.profile_json.get("recommendation_score")
    if raw_score is None:
        score_components = segment.profile_json.get("score_components")
        if isinstance(score_components, Mapping):
            raw_score = score_components.get("final_score")
    if raw_score is None:
        raw_score = segment.profile_json.get("cluster_score", 0.0)
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def _ai_score_details(segment: SegmentDefinitionRecord) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for key in (
        "promotion_cluster_similarity",
        "recommendation_score",
        "cluster_quality_score",
        "sample_size_score",
        "candidate_type",
        "strategy_role",
        "recommendation_tier",
        "recommendation_tier_label",
        "recommendation_tier_reason",
        "rank_eligible",
        "minimum_primary_sample_size",
        "performance_estimate",
        "portfolio_position",
        "selection_basis",
    ):
        value = segment.profile_json.get(key)
        if value is not None:
            details[key] = value
    score_components = segment.profile_json.get("score_components")
    if isinstance(score_components, Mapping):
        details["score_components"] = dict(score_components)
    return details


def _suggestion_source(segment: SegmentDefinitionRecord) -> str:
    if segment.source == "ai_suggested":
        return "ai_generated"
    return "ai_ranked_existing"


def _suggestion_id(*, analysis_id: str, segment_id: str) -> str:
    digest = hashlib.sha1(  # noqa: S324 - stable non-security identifier.
        f"{analysis_id}:{segment_id}".encode("utf-8"),
    ).hexdigest()[:24]
    return f"sugg_{digest}"


def _total_eligible_users(
    segment_definitions: Sequence[SegmentDefinitionRecord],
) -> int:
    if not segment_definitions:
        return 0
    return max(segment.total_eligible_user_count for segment in segment_definitions)


def _candidate_user_ids(rule_json: Mapping[str, Any]) -> list[str]:
    raw_user_ids = rule_json.get("candidate_user_ids") or rule_json.get("user_ids")
    if isinstance(raw_user_ids, str) or not isinstance(raw_user_ids, Sequence):
        return []
    return [str(user_id) for user_id in raw_user_ids]


def _analysis_request_log_payload(request: AnalysisRequest) -> dict[str, Any]:
    segment_ids = getattr(request, "segment_ids", None)
    operator_instruction = getattr(request, "operator_instruction", None)
    segment_instruction = getattr(request, "segment_instruction", None)
    return {
        "hasOperatorInstruction": bool(operator_instruction),
        "operatorInstructionLength": len(operator_instruction or ""),
        "hasSegmentInstruction": bool(segment_instruction),
        "segmentInstructionLength": len(segment_instruction or ""),
        "segmentCount": len(segment_ids) if segment_ids is not None else None,
    }


def _analysis_result_log_payload(
    response: PromotionAnalysisResult,
) -> dict[str, Any]:
    return {
        "status": response.analysis.status,
        "targetSegmentCount": len(response.target_segments),
        "segmentSuggestionCount": len(response.segment_suggestions),
    }


def _json_decimal(value: Decimal) -> str:
    return format(value, "f")
