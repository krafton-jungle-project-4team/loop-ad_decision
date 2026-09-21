from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

from app.audience_contract import (
    CUSTOM_SOURCE_REFINEMENT_ANCHOR_POLICY_ID,
    CUSTOM_SOURCE_REFINEMENT_SELECTION_POLICY_ID,
    CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION,
    CUSTOM_STRUCTURED_ANCHOR_POLICY_ID,
    CUSTOM_STRUCTURED_SELECTION_POLICY_ID,
    CUSTOM_STRUCTURED_TEMPLATE_ID,
    LEGACY_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_QUERY_COMPILER_HASH,
    SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION,
    SegmentAudienceSpec,
    custom_structured_template_hash,
)
from app.analysis.behavior_manifest import (
    behavior_manifest_hash,
    canonical_destination_id,
    load_behavior_manifest,
    manifest_blocks,
    manifest_candidate_block_weights,
    manifest_candidate_hard_predicates,
    manifest_candidate_query_indices,
    manifest_intent_benefit_query_indices,
    manifest_season_query_indices,
)
from app.analysis.raw_event_segments import (
    PromotionIntent,
    destination_terms_from_intent,
)
from app.analysis.repositories import RawEventUserSignalRecord


_MANIFEST = load_behavior_manifest()
VECTOR_DIM = int(_MANIFEST["vector_dim"])
HOTEL_BEHAVIOR_VECTOR_VERSION = str(_MANIFEST["vector_version"])
HOTEL_BEHAVIOR_SCHEMA_VERSION = str(_MANIFEST["schema_version"])
HOTEL_BEHAVIOR_MANIFEST_HASH = behavior_manifest_hash()
USER_BEHAVIOR_VECTORIZER_VERSION = "hotel_user_behavior_vectorizer.v2"
_USER_VECTORIZER_SEMANTICS = {
    "version": USER_BEHAVIOR_VECTORIZER_VERSION,
    "manifest_hash": HOTEL_BEHAVIOR_MANIFEST_HASH,
    "normalization": _MANIFEST["normalization"],
    "missing_value_policy": _MANIFEST["missing_value_policy"],
    "dimension_calculations": [
        {
            "index": int(item["index"]),
            "name": str(item["name"]),
            "raw_calculation": str(item["raw_calculation"]),
        }
        for item in _MANIFEST["dimensions"]
    ],
    "bounded_intensity": "log1p(count)/log1p(max(event_count,1));cap=1",
    "smoothed_rate": "beta_prior_alpha=1,beta=9",
    "destination_hash": "sha256_signed_bucket_16_then_divide_event_count",
    "recency": "exp(-max(age_days,0)/30)",
    "checkin_shares": "count/checkin_date_count",
    "price_shares": "price_band_count/price_event_count",
}
USER_BEHAVIOR_VECTORIZER_SEMANTIC_HASH = hashlib.sha256(
    json.dumps(
        _USER_VECTORIZER_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
DESTINATION_BLOCK_START = 16
DESTINATION_BLOCK_SIZE = 16
CUSTOM_STRUCTURED_QUERY_COMPILER_VERSION = "custom_structured_query.v1"
_CUSTOM_STRUCTURED_QUERY_COMPILER_SEMANTICS = {
    "version": CUSTOM_STRUCTURED_QUERY_COMPILER_VERSION,
    "manifest_hash": HOTEL_BEHAVIOR_MANIFEST_HASH,
    "membership": "validated_structured_conditions_are_authoritative",
    "vector_role": "ordering_and_allocation_tiebreak_only",
    "score_threshold": -1.0,
    "semantic_margin": 2.0,
}
CUSTOM_STRUCTURED_QUERY_COMPILER_HASH = hashlib.sha256(
    json.dumps(
        _CUSTOM_STRUCTURED_QUERY_COMPILER_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
CUSTOM_SOURCE_REFINEMENT_QUERY_COMPILER_VERSION = "custom_source_refinement_query.v1"
_CUSTOM_SOURCE_REFINEMENT_QUERY_COMPILER_SEMANTICS = {
    "version": CUSTOM_SOURCE_REFINEMENT_QUERY_COMPILER_VERSION,
    "manifest_hash": HOTEL_BEHAVIOR_MANIFEST_HASH,
    "membership": "source_user_ids_with_optional_validated_structured_conditions",
    "vector_role": "ordering_and_allocation_tiebreak_only",
    "score_threshold": -1.0,
    "semantic_margin": 2.0,
}
CUSTOM_SOURCE_REFINEMENT_QUERY_COMPILER_HASH = hashlib.sha256(
    json.dumps(
        _CUSTOM_SOURCE_REFINEMENT_QUERY_COMPILER_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateBehaviorSpec:
    candidate_type: str
    schema_version: str
    vector_version: str
    hard_predicate_keys: tuple[str, ...]
    predicate_parameters: Mapping[str, tuple[str, ...] | tuple[int, ...]]
    query_vector: tuple[float, ...]
    active_blocks: tuple[str, ...]
    block_weights: Mapping[str, float]
    score_threshold: float
    calibration_version: str
    manifest_hash: str = HOTEL_BEHAVIOR_MANIFEST_HASH
    calibration_hash: str = ""
    audience_resolution_contract: str = LEGACY_AUDIENCE_CONTRACT
    segment_audience_spec_hash: str = ""
    query_compiler_version: str = SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION
    query_compiler_hash: str = SEGMENT_AUDIENCE_QUERY_COMPILER_HASH
    template_id: str = ""
    template_version: int = 0
    template_semantic_hash: str = ""
    semantic_selection_policy_id: str = ""
    semantic_anchor_policy_id: str = ""
    semantic_anchor_hash: str = ""
    semantic_margin: float = 0.0
    semantic_selection_status: str = ""
    business_lift_status: str = ""
    user_vectorizer_version: str = USER_BEHAVIOR_VECTORIZER_VERSION
    user_vectorizer_semantic_hash: str = USER_BEHAVIOR_VECTORIZER_SEMANTIC_HASH


@dataclass(frozen=True, slots=True)
class CandidateCalibration:
    score_threshold: float
    version: str
    artifact_hash: str = ""
    manifest_hash: str = HOTEL_BEHAVIOR_MANIFEST_HASH
    schema_version: str = HOTEL_BEHAVIOR_SCHEMA_VERSION
    vector_version: str = HOTEL_BEHAVIOR_VECTOR_VERSION
    labeled_user_count: int = 0
    positive_user_count: int = 0
    template_id: str = ""
    template_semantic_hash: str = ""
    semantic_selection_policy_id: str = ""
    semantic_anchor_policy_id: str = ""
    semantic_anchor_hash: str = ""
    semantic_margin: float = 0.0
    semantic_selection_status: str = ""
    business_lift_status: str = ""
    user_vectorizer_version: str = USER_BEHAVIOR_VECTORIZER_VERSION
    user_vectorizer_semantic_hash: str = USER_BEHAVIOR_VECTORIZER_SEMANTIC_HASH


_BLOCKS = manifest_blocks()
_CANDIDATE_BLOCK_WEIGHTS = manifest_candidate_block_weights()
_CANDIDATE_QUERY_INDICES = manifest_candidate_query_indices()
_HARD_PREDICATES = manifest_candidate_hard_predicates()
_SEASON_QUERY_INDICES = manifest_season_query_indices()
_INTENT_BENEFIT_QUERY_INDICES = manifest_intent_benefit_query_indices()
_QUERY_DIMENSION_INDICES = {
    str(item["name"]): int(item["index"])
    for item in _MANIFEST["dimensions"]
    if bool(item.get("query_enabled"))
}
_MONTH_QUERY_INDICES = {
    month: _SEASON_QUERY_INDICES[season]
    for season, months in {
        "spring": (3, 4, 5),
        "summer": (6, 7, 8),
        "fall": (9, 10, 11),
        "winter": (12, 1, 2),
    }.items()
    for month in months
}


class HotelBookingBehaviorSchemaV2:
    """Shared semantic coordinate system for candidate queries and users.

    A zero value means that the source signal is unavailable or absent. It is
    never replaced with a demographic or text-derived guess.
    """

    vector_dim = VECTOR_DIM
    schema_version = HOTEL_BEHAVIOR_SCHEMA_VERSION
    vector_version = HOTEL_BEHAVIOR_VECTOR_VERSION

    def vectorize_user(self, profile: RawEventUserSignalRecord) -> list[float]:
        event_count = max(1, int(profile.event_count))
        values = [0.0] * VECTOR_DIM

        counts = (
            profile.page_view_count,
            profile.hotel_search_count,
            profile.hotel_click_count,
            profile.hotel_detail_view_count,
            profile.booking_start_count,
            profile.booking_complete_count,
            profile.booking_cancel_count,
        )
        for index, count in enumerate(counts):
            values[index] = _bounded_intensity(count, event_count)

        values[7] = _recency(profile.hotel_search_recency_days)
        values[8] = _recency(profile.hotel_detail_recency_days)
        values[9] = _recency(profile.booking_start_recency_days)

        values[10] = _smoothed_rate(
            profile.hotel_click_count,
            profile.hotel_search_count,
        )
        values[11] = _smoothed_rate(
            profile.hotel_detail_view_count,
            profile.hotel_click_count,
        )
        values[12] = _smoothed_rate(
            profile.booking_start_count,
            profile.hotel_detail_view_count,
        )
        values[13] = _smoothed_rate(
            profile.booking_complete_count,
            profile.booking_start_count,
        )
        values[14] = float(
            profile.booking_start_count > profile.booking_complete_count
            and profile.booking_start_count > 0
        )
        values[15] = _bounded_intensity(
            profile.hotel_detail_view_count + profile.hotel_click_count,
            event_count,
        )

        for destination in profile.destination_values:
            _add_signed_hash(values, destination)
        for index in range(DESTINATION_BLOCK_START, DESTINATION_BLOCK_START + DESTINATION_BLOCK_SIZE):
            values[index] /= event_count

        checkin_months = [
            month
            for value in profile.checkin_dates
            if (month := _checkin_month(value)) is not None
        ]
        if checkin_months:
            total_checkins = len(checkin_months)
            values[32] = sum(month in {3, 4, 5} for month in checkin_months) / total_checkins
            values[33] = sum(month in {6, 7, 8} for month in checkin_months) / total_checkins
            values[34] = sum(month in {9, 10, 11} for month in checkin_months) / total_checkins
            values[35] = sum(month in {12, 1, 2} for month in checkin_months) / total_checkins
            values[36] = profile.lead_time_0_7_count / total_checkins
            values[37] = profile.lead_time_8_30_count / total_checkins
            values[38] = profile.lead_time_gt_30_count / total_checkins
            values[39] = profile.weekend_checkin_count / total_checkins

        values[40] = _bounded_intensity(profile.deal_event_count, event_count)
        values[41] = _recency(profile.deal_recency_days)
        values[42] = _bounded_intensity(profile.price_event_count, event_count)
        if profile.price_event_count > 0:
            values[43] = min(
                1.0,
                profile.budget_price_count / profile.price_event_count,
            )
            values[44] = min(
                1.0,
                profile.premium_price_count / profile.price_event_count,
            )
        values[45] = _bounded_intensity(
            profile.free_cancellation_count,
            event_count,
        )
        values[46] = _bounded_intensity(profile.breakfast_included_count, event_count)
        values[47] = min(
            1.0,
            (
                int(profile.deal_event_count > 0)
                + int(profile.free_cancellation_count > 0)
                + int(profile.breakfast_included_count > 0)
            )
            / 3.0,
        )

        values[48] = _bounded_intensity(profile.promotion_impression_count, event_count)
        values[49] = _bounded_intensity(profile.promotion_click_count, event_count)
        values[50] = _bounded_intensity(
            profile.campaign_redirect_click_count,
            event_count,
        )
        values[51] = _bounded_intensity(profile.campaign_landing_count, event_count)
        values[52] = _smoothed_rate(
            profile.promotion_click_count,
            profile.promotion_impression_count,
        )
        values[53] = _smoothed_rate(
            profile.campaign_landing_count,
            profile.promotion_click_count,
        )
        values[54] = _recency(profile.promotion_response_recency_days)
        values[55] = _bounded_intensity(
            profile.promotion_click_count + profile.campaign_landing_count,
            event_count,
        )

        values[56] = float(
            profile.hotel_search_count
            + profile.hotel_click_count
            + profile.hotel_detail_view_count
            > 0
        )
        canonical_destinations = {
            canonical_destination(value)
            for value in profile.destination_values
            if canonical_destination(value)
        }
        values[57] = float(len(canonical_destinations) == 1)
        values[58] = min(1.0, len(canonical_destinations) / 3.0)
        values[59] = values[14]
        values[60] = float(profile.promotion_click_count + profile.campaign_landing_count > 0)
        values[61] = float(
            profile.deal_event_count
            + profile.free_cancellation_count
            + profile.breakfast_included_count
            > 0
        )
        latest_relevant_age = min(
            (
                value
                for value in (
                    profile.hotel_search_recency_days,
                    profile.hotel_detail_recency_days,
                    profile.booking_start_recency_days,
                    profile.promotion_response_recency_days,
                )
                if value is not None
            ),
            default=None,
        )
        values[62] = float(
            latest_relevant_age is not None and latest_relevant_age <= 30
        )
        values[63] = min(
            1.0,
            (0.34 if profile.hotel_detail_view_count + profile.hotel_click_count >= 2 else 0.0)
            + (0.33 if values[14] else 0.0)
            + (0.33 if values[60] else 0.0),
        )
        return _l2_normalize(values)

    def compile_candidate(
        self,
        *,
        candidate_type: str,
        intent: PromotionIntent,
        calibration: CandidateCalibration,
    ) -> CandidateBehaviorSpec:
        configured = _CANDIDATE_BLOCK_WEIGHTS.get(candidate_type)
        if configured is None:
            raise ValueError(f"unsupported candidate_type: {candidate_type}")
        if not 0.0 <= calibration.score_threshold <= 1.0:
            raise ValueError("candidate score threshold must be between 0 and 1")
        if calibration.manifest_hash != HOTEL_BEHAVIOR_MANIFEST_HASH:
            raise ValueError("calibration manifest hash does not match behavior manifest")
        if calibration.schema_version != self.schema_version:
            raise ValueError("calibration schema version does not match behavior schema")
        if calibration.vector_version != self.vector_version:
            raise ValueError("calibration vector version does not match behavior schema")
        weights = dict(configured)
        if not intent.destinations:
            weights.pop("destination", None)
        if not intent.season:
            weights.pop("timing", None)
        weights = _renormalize_weights(weights)

        query = [0.0] * VECTOR_DIM
        _apply_candidate_signals(query, candidate_type)
        benefit_keys = _supported_intent_benefits(intent.benefits)
        if candidate_type == "benefit_value_seeker" and benefit_keys:
            _apply_intent_benefit_signals(query, benefit_keys)
        for destination in _canonical_destinations(intent.destinations):
            _add_signed_hash(query, destination)
        _apply_season_signals(query, intent.season)
        _apply_block_weights(query, weights)

        hard_predicates = list(_HARD_PREDICATES[candidate_type])
        if intent.destinations and candidate_type in {
            "intent_matched",
            "target_destination_affinity",
            "funnel_recovery",
            "benefit_value_seeker",
        }:
            hard_predicates.append("recent_destination_search")
        if intent.season and candidate_type == "intent_matched":
            hard_predicates.append("season_match")

        return CandidateBehaviorSpec(
            candidate_type=candidate_type,
            schema_version=self.schema_version,
            vector_version=self.vector_version,
            hard_predicate_keys=tuple(hard_predicates),
            predicate_parameters={
                "destinations": _canonical_destinations(
                    destination_terms_from_intent(intent)
                ),
                "season_months": tuple(_season_months(intent.season)),
                "benefit_keys": benefit_keys,
            },
            query_vector=tuple(_l2_normalize(query)),
            active_blocks=tuple(weights),
            block_weights=weights,
            score_threshold=calibration.score_threshold,
            calibration_version=calibration.version,
            manifest_hash=HOTEL_BEHAVIOR_MANIFEST_HASH,
            calibration_hash=calibration.artifact_hash,
        )

    def compile_segment_audience(
        self,
        *,
        spec: SegmentAudienceSpec,
        calibration: CandidateCalibration,
    ) -> CandidateBehaviorSpec:
        """Compile only the meaning serialized on the selected segment."""
        configured = _CANDIDATE_BLOCK_WEIGHTS.get(spec.candidate_type)
        if configured is None:
            raise ValueError(f"unsupported candidate_type: {spec.candidate_type}")
        if not 0.0 <= calibration.score_threshold <= 1.0:
            raise ValueError("candidate score threshold must be between 0 and 1")
        if calibration.manifest_hash != HOTEL_BEHAVIOR_MANIFEST_HASH:
            raise ValueError("calibration manifest hash does not match behavior manifest")
        if calibration.schema_version != self.schema_version:
            raise ValueError("calibration schema version does not match behavior schema")
        if calibration.vector_version != self.vector_version:
            raise ValueError("calibration vector version does not match behavior schema")
        if calibration.template_id != spec.template_id:
            raise ValueError("semantic selection template does not match segment spec")
        if calibration.template_semantic_hash != spec.template_semantic_hash:
            raise ValueError(
                "semantic selection template hash does not match segment spec"
            )
        if (
            calibration.semantic_selection_policy_id
            != spec.semantic_selection_policy_id
        ):
            raise ValueError("semantic selection policy does not match segment spec")
        if calibration.semantic_anchor_policy_id != spec.semantic_anchor_policy_id:
            raise ValueError("semantic anchor policy does not match segment spec")

        query = [0.0] * VECTOR_DIM
        for signal_key in spec.query_signal_keys:
            try:
                query[_QUERY_DIMENSION_INDICES[signal_key]] = 1.0
            except KeyError as exc:
                raise ValueError(
                    f"unsupported segment query signal: {signal_key}"
                ) from exc
        for destination_id in spec.destination_ids:
            _add_signed_hash(query, destination_id)
        for month in spec.season_months:
            query[_MONTH_QUERY_INDICES[month]] = 1.0
        if spec.benefit_keys:
            _apply_intent_benefit_signals(query, spec.benefit_keys)

        active_blocks = {
            block
            for block, indices in _BLOCKS.items()
            if any(query[index] != 0 for index in indices)
        }
        weights = _renormalize_weights(
            {
                block: weight
                for block, weight in configured.items()
                if block in active_blocks
            }
        )
        _apply_block_weights(query, weights)
        return CandidateBehaviorSpec(
            candidate_type=spec.candidate_type,
            schema_version=self.schema_version,
            vector_version=self.vector_version,
            hard_predicate_keys=spec.hard_predicate_keys,
            predicate_parameters=spec.predicate_parameters,
            query_vector=tuple(_l2_normalize(query)),
            active_blocks=tuple(weights),
            block_weights=weights,
            score_threshold=calibration.score_threshold,
            calibration_version=calibration.version,
            manifest_hash=HOTEL_BEHAVIOR_MANIFEST_HASH,
            calibration_hash=calibration.artifact_hash,
            audience_resolution_contract=SEGMENT_AUDIENCE_CONTRACT,
            segment_audience_spec_hash=spec.spec_hash,
            query_compiler_version=SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION,
            query_compiler_hash=SEGMENT_AUDIENCE_QUERY_COMPILER_HASH,
            template_id=spec.template_id,
            template_version=spec.template_version,
            template_semantic_hash=spec.template_semantic_hash,
            semantic_selection_policy_id=spec.semantic_selection_policy_id,
            semantic_anchor_policy_id=spec.semantic_anchor_policy_id,
            semantic_anchor_hash=calibration.semantic_anchor_hash,
            semantic_margin=calibration.semantic_margin,
            semantic_selection_status=calibration.semantic_selection_status,
            business_lift_status=calibration.business_lift_status,
            user_vectorizer_version=calibration.user_vectorizer_version,
            user_vectorizer_semantic_hash=(
                calibration.user_vectorizer_semantic_hash
            ),
        )

    def compile_custom_segment_audience(
        self,
        *,
        spec: SegmentAudienceSpec,
    ) -> CandidateBehaviorSpec:
        """Compile an allowlisted custom predicate without semantic-anchor inference.

        Exact structured predicates decide membership. The behavior vector only
        orders already-matching users and resolves overlap during allocation.
        """
        if not spec.is_custom_structured:
            raise ValueError("custom audience compiler requires a custom template")
        if spec.template_version == CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION:
            expected_template_hash = custom_structured_template_hash(
                template_version=CUSTOM_SOURCE_REFINEMENT_TEMPLATE_VERSION,
                window_days=spec.observation_window_days,
            )
            expected_selection_policy = CUSTOM_SOURCE_REFINEMENT_SELECTION_POLICY_ID
            expected_anchor_policy = CUSTOM_SOURCE_REFINEMENT_ANCHOR_POLICY_ID
            query_compiler_version = CUSTOM_SOURCE_REFINEMENT_QUERY_COMPILER_VERSION
            query_compiler_hash = CUSTOM_SOURCE_REFINEMENT_QUERY_COMPILER_HASH
            calibration_version = "custom_source_refinement_exact.v1"
            semantic_selection_status = "exact_source_refinement"
        else:
            expected_template_hash = custom_structured_template_hash(
                template_version=spec.template_version,
                window_days=spec.observation_window_days,
            )
            expected_selection_policy = CUSTOM_STRUCTURED_SELECTION_POLICY_ID
            expected_anchor_policy = CUSTOM_STRUCTURED_ANCHOR_POLICY_ID
            query_compiler_version = CUSTOM_STRUCTURED_QUERY_COMPILER_VERSION
            query_compiler_hash = CUSTOM_STRUCTURED_QUERY_COMPILER_HASH
            calibration_version = "custom_structured_exact.v1"
            semantic_selection_status = "exact_structured_conditions"
        if spec.template_semantic_hash != expected_template_hash:
            raise ValueError("custom audience template hash does not match")
        if spec.semantic_selection_policy_id != expected_selection_policy:
            raise ValueError("custom audience selection policy does not match")
        if spec.semantic_anchor_policy_id != expected_anchor_policy:
            raise ValueError("custom audience anchor policy does not match")

        query = [0.0] * VECTOR_DIM
        for signal_key in spec.query_signal_keys:
            try:
                query[_QUERY_DIMENSION_INDICES[signal_key]] = 1.0
            except KeyError as exc:
                raise ValueError(
                    f"unsupported custom segment query signal: {signal_key}"
                ) from exc
        for destination_id in spec.destination_ids:
            _add_signed_hash(query, destination_id)
        for month in spec.season_months:
            query[_MONTH_QUERY_INDICES[month]] = 1.0

        active_blocks = {
            block
            for block, indices in _BLOCKS.items()
            if any(query[index] != 0 for index in indices)
        }
        weights = {
            block: 1.0 / len(active_blocks)
            for block in sorted(active_blocks)
        }
        _apply_block_weights(query, weights)
        return CandidateBehaviorSpec(
            candidate_type=spec.candidate_type,
            schema_version=self.schema_version,
            vector_version=self.vector_version,
            hard_predicate_keys=spec.hard_predicate_keys,
            predicate_parameters=spec.predicate_parameters,
            query_vector=tuple(_l2_normalize(query)),
            active_blocks=tuple(weights),
            block_weights=weights,
            score_threshold=-1.0,
            calibration_version=calibration_version,
            manifest_hash=HOTEL_BEHAVIOR_MANIFEST_HASH,
            calibration_hash=query_compiler_hash,
            audience_resolution_contract=SEGMENT_AUDIENCE_CONTRACT,
            segment_audience_spec_hash=spec.spec_hash,
            query_compiler_version=query_compiler_version,
            query_compiler_hash=query_compiler_hash,
            template_id=CUSTOM_STRUCTURED_TEMPLATE_ID,
            template_version=spec.template_version,
            template_semantic_hash=spec.template_semantic_hash,
            semantic_selection_policy_id=spec.semantic_selection_policy_id,
            semantic_anchor_policy_id=spec.semantic_anchor_policy_id,
            semantic_anchor_hash=expected_template_hash,
            semantic_margin=2.0,
            semantic_selection_status=semantic_selection_status,
            business_lift_status="not_applicable_to_direct_segment",
        )


def canonical_destination(value: str) -> str:
    return canonical_destination_id(value)


def signed_hash_coordinate(value: str) -> tuple[int, float]:
    canonical = canonical_destination(value)
    if not canonical:
        raise ValueError("destination must not be empty")
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % DESTINATION_BLOCK_SIZE
    sign = 1.0 if digest[8] & 1 == 0 else -1.0
    return DESTINATION_BLOCK_START + bucket, sign


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != VECTOR_DIM or len(right) != VECTOR_DIM:
        raise ValueError("behavior vectors must contain 64 values")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _bounded_intensity(count: int, total: int) -> float:
    return min(1.0, math.log1p(max(0, count)) / math.log1p(max(1, total)))


def _smoothed_rate(numerator: int, denominator: int) -> float:
    # Beta prior mean 0.1 with prior strength 10.
    return (max(0, numerator) + 1.0) / (max(0, denominator) + 10.0)


def _recency(age_days: int | None) -> float:
    if age_days is None:
        return 0.0
    return math.exp(-max(0, age_days) / 30.0)


def _checkin_month(value: str) -> int | None:
    try:
        return date.fromisoformat(value[:10]).month
    except (TypeError, ValueError):
        return None


def _canonical_destinations(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            canonical
            for value in values
            if (canonical := canonical_destination(value))
        )
    )


def _add_signed_hash(vector: list[float], destination: str) -> None:
    try:
        index, sign = signed_hash_coordinate(destination)
    except ValueError:
        return
    vector[index] += sign


def _renormalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("candidate must contain at least one active behavior block")
    return {key: float(value) / total for key, value in weights.items()}


def _apply_block_weights(vector: list[float], weights: Mapping[str, float]) -> None:
    for block, indices in _BLOCKS.items():
        block_weight = weights.get(block, 0.0)
        block_norm = math.sqrt(sum(vector[index] ** 2 for index in indices))
        for index in indices:
            vector[index] = (
                (vector[index] / block_norm) * block_weight
                if block_weight > 0 and block_norm > 0
                else 0.0
            )


def _apply_candidate_signals(vector: list[float], candidate_type: str) -> None:
    for index in _CANDIDATE_QUERY_INDICES[candidate_type]:
        vector[index] = 1.0


def _apply_season_signals(vector: list[float], seasons: Sequence[str]) -> None:
    for season in seasons:
        index = _SEASON_QUERY_INDICES.get(season.strip().casefold())
        if index is not None:
            vector[index] = 1.0


def _supported_intent_benefits(benefits: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            benefit.strip().casefold()
            for benefit in benefits
            if benefit.strip().casefold() in _INTENT_BENEFIT_QUERY_INDICES
        )
    )


def _apply_intent_benefit_signals(
    vector: list[float],
    benefit_keys: Sequence[str],
) -> None:
    intent_specific_indices = {
        index
        for indices in _INTENT_BENEFIT_QUERY_INDICES.values()
        for index in indices
    }
    for index in intent_specific_indices:
        vector[index] = 0.0
    for benefit in benefit_keys:
        for index in _INTENT_BENEFIT_QUERY_INDICES[benefit]:
            vector[index] = 1.0


def _season_months(seasons: Sequence[str]) -> list[int]:
    months_by_season = {
        "spring": (3, 4, 5),
        "summer": (6, 7, 8),
        "fall": (9, 10, 11),
        "autumn": (9, 10, 11),
        "winter": (12, 1, 2),
    }
    months: list[int] = []
    for season in seasons:
        months.extend(months_by_season.get(season.strip().casefold(), ()))
    return list(dict.fromkeys(months))


def _l2_normalize(values: Sequence[float]) -> list[float]:
    if len(values) != VECTOR_DIM:
        raise ValueError("behavior vector must contain 64 values")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("behavior vector must contain finite values")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm == 0:
        raise ValueError("behavior vector must not be zero")
    return [float(value) / norm for value in values]
