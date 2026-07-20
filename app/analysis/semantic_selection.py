from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.audience_contract import (
    SEGMENT_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_QUERY_COMPILER_HASH,
    SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION,
    SEGMENT_AUDIENCE_SCHEMA_VERSION,
    SegmentAudienceContractError,
    SegmentAudienceSpec,
    SegmentDefinitionAudienceAdapter,
)
from app.analysis.behavior_vector_schema import (
    CandidateBehaviorSpec,
    CandidateCalibration,
    HOTEL_BEHAVIOR_MANIFEST_HASH,
    HOTEL_BEHAVIOR_SCHEMA_VERSION,
    HOTEL_BEHAVIOR_VECTOR_VERSION,
    USER_BEHAVIOR_VECTORIZER_SEMANTIC_HASH,
    USER_BEHAVIOR_VECTORIZER_VERSION,
    HotelBookingBehaviorSchemaV2,
    cosine_similarity,
)
from app.analysis.repositories import RawEventUserSignalRecord
from app.analysis.segment_audience_templates import (
    REGISTERED_SEGMENT_AUDIENCE_TEMPLATES,
)


BUNDLED_SEMANTIC_SELECTION_PATH = (
    Path(__file__).parent
    / "calibrations"
    / "hotel_behavior_v2_segment_audience_templates_v1.json"
)
BUNDLED_SEMANTIC_SELECTION_SHA256 = (
    "65060ff8aff2565fd5c2942ad0f9a6fa92104ad4a6f4ac0bcba7ba9658578e7c"
)


@dataclass(frozen=True, slots=True)
class _TemplateAnchors:
    template_id: str
    template_version: int
    selection_artifact_version: str
    selection_artifact_hash: str
    template_semantic_hash: str
    semantic_selection_policy_id: str
    semantic_anchor_policy_id: str
    accepted: tuple[Mapping[str, Any], ...]
    boundary_accepted: tuple[Mapping[str, Any], ...]
    boundary_rejected: tuple[Mapping[str, Any], ...]
    negative: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SemanticSelectionArtifact:
    version: str
    artifact_hash: str
    manifest_hash: str
    minimum_margin: float
    semantic_selection_status: str
    business_lift_status: str
    templates: Mapping[str, _TemplateAnchors]

    def calibration_for(
        self,
        *,
        segment_id: str,
        spec: SegmentAudienceSpec,
        schema: HotelBookingBehaviorSchemaV2,
    ) -> CandidateCalibration:
        anchors = self.templates.get(spec.template_id)
        if anchors is None:
            raise _contract_error(
                "segment_audience_template_unregistered",
                segment_id,
                f"semantic anchors are not registered for {spec.template_id}",
            )
        if (
            anchors.template_version != spec.template_version
            or anchors.template_semantic_hash != spec.template_semantic_hash
            or anchors.semantic_selection_policy_id
            != spec.semantic_selection_policy_id
            or anchors.semantic_anchor_policy_id != spec.semantic_anchor_policy_id
        ):
            raise _contract_error(
                "segment_audience_template_hash_mismatch",
                segment_id,
                "semantic anchor binding does not match the segment template",
            )

        provisional = CandidateCalibration(
            score_threshold=0.0,
            version=anchors.selection_artifact_version,
            artifact_hash=anchors.selection_artifact_hash,
            manifest_hash=self.manifest_hash,
            schema_version=HOTEL_BEHAVIOR_SCHEMA_VERSION,
            vector_version=HOTEL_BEHAVIOR_VECTOR_VERSION,
            template_id=spec.template_id,
            template_semantic_hash=spec.template_semantic_hash,
            semantic_selection_policy_id=spec.semantic_selection_policy_id,
            semantic_anchor_policy_id=spec.semantic_anchor_policy_id,
            semantic_selection_status=self.semantic_selection_status,
            business_lift_status=self.business_lift_status,
            user_vectorizer_version=USER_BEHAVIOR_VECTORIZER_VERSION,
            user_vectorizer_semantic_hash=(
                USER_BEHAVIOR_VECTORIZER_SEMANTIC_HASH
            ),
        )
        query = schema.compile_segment_audience(
            spec=spec,
            calibration=provisional,
        ).query_vector
        accepted_profiles = tuple(
            _instantiate_anchor(recipe, spec=spec, index=index)
            for index, recipe in enumerate(
                anchors.accepted + anchors.boundary_accepted
            )
        )
        rejected_profiles = tuple(
            _instantiate_anchor(recipe, spec=spec, index=index + 100)
            for index, recipe in enumerate(anchors.boundary_rejected)
        )
        negative_profiles = tuple(
            _instantiate_anchor(recipe, spec=spec, index=index + 200)
            for index, recipe in enumerate(anchors.negative)
        )
        if not accepted_profiles or not rejected_profiles or not negative_profiles:
            raise _contract_error(
                "segment_audience_calibration_invalid",
                segment_id,
                "semantic selection requires accepted, rejected, and negative anchors",
            )
        if any(
            not _matches_hard_predicates(profile, spec=spec)
            for profile in accepted_profiles + rejected_profiles
        ):
            raise _contract_error(
                "segment_audience_calibration_invalid",
                segment_id,
                "accepted and boundary-rejected anchors must satisfy hard predicates",
            )
        if any(
            _matches_hard_predicates(profile, spec=spec)
            for profile in negative_profiles
        ):
            raise _contract_error(
                "segment_audience_calibration_invalid",
                segment_id,
                "negative anchors must not satisfy hard predicates",
            )

        accepted_scores = tuple(
            cosine_similarity(query, schema.vectorize_user(profile))
            for profile in accepted_profiles
        )
        rejected_scores = tuple(
            cosine_similarity(query, schema.vectorize_user(profile))
            for profile in rejected_profiles
        )
        min_accepted = min(accepted_scores)
        max_rejected = max(rejected_scores)
        margin = min_accepted - max_rejected
        if margin < self.minimum_margin:
            raise _contract_error(
                "segment_audience_semantic_separation_insufficient",
                segment_id,
                (
                    f"{spec.template_id} semantic margin {margin:.6f} is below "
                    f"{self.minimum_margin:.6f}"
                ),
            )
        threshold = (min_accepted + max_rejected) / 2.0
        anchor_hash = _instantiated_anchor_hash(
            spec=spec,
            accepted=accepted_profiles,
            rejected=rejected_profiles,
            negative=negative_profiles,
        )
        return CandidateCalibration(
            score_threshold=threshold,
            version=anchors.selection_artifact_version,
            artifact_hash=anchors.selection_artifact_hash,
            manifest_hash=self.manifest_hash,
            schema_version=HOTEL_BEHAVIOR_SCHEMA_VERSION,
            vector_version=HOTEL_BEHAVIOR_VECTOR_VERSION,
            labeled_user_count=len(
                accepted_profiles + rejected_profiles + negative_profiles
            ),
            positive_user_count=len(accepted_profiles),
            template_id=spec.template_id,
            template_semantic_hash=spec.template_semantic_hash,
            semantic_selection_policy_id=spec.semantic_selection_policy_id,
            semantic_anchor_policy_id=spec.semantic_anchor_policy_id,
            semantic_anchor_hash=anchor_hash,
            semantic_margin=margin,
            semantic_selection_status=self.semantic_selection_status,
            business_lift_status=self.business_lift_status,
            user_vectorizer_version=USER_BEHAVIOR_VECTORIZER_VERSION,
            user_vectorizer_semantic_hash=(
                USER_BEHAVIOR_VECTORIZER_SEMANTIC_HASH
            ),
        )


class BundledSemanticSelectionProvider:
    def __init__(self) -> None:
        self._artifact: SemanticSelectionArtifact | None = None

    def require(
        self,
        *,
        segment_id: str,
        spec: SegmentAudienceSpec,
        schema: HotelBookingBehaviorSchemaV2,
    ) -> CandidateCalibration:
        if self._artifact is None:
            self._artifact = load_bundled_semantic_selection(
                segment_id=segment_id,
            )
        return self._artifact.calibration_for(
            segment_id=segment_id,
            spec=spec,
            schema=schema,
        )


_DEFAULT_SEMANTIC_SELECTION_PROVIDER = BundledSemanticSelectionProvider()


def compile_registered_segment_audience(
    *,
    segment_id: str,
    rule_json: Mapping[str, Any],
    provider: BundledSemanticSelectionProvider | None = None,
    schema: HotelBookingBehaviorSchemaV2 | None = None,
) -> CandidateBehaviorSpec:
    resolution = SegmentDefinitionAudienceAdapter().resolve(
        segment_id=segment_id,
        rule_json=rule_json,
    )
    if not resolution.is_v2 or resolution.spec is None:
        raise _contract_error(
            "segment_audience_contract_unsupported",
            segment_id,
            "registered audience compiler requires segment_audience.v1",
        )
    behavior_schema = schema or HotelBookingBehaviorSchemaV2()
    if resolution.spec.is_custom_structured:
        try:
            return behavior_schema.compile_custom_segment_audience(
                spec=resolution.spec,
            )
        except ValueError as exc:
            raise _contract_error(
                "segment_audience_manifest_mismatch",
                segment_id,
                str(exc),
            ) from exc
    calibration = (provider or _DEFAULT_SEMANTIC_SELECTION_PROVIDER).require(
        segment_id=segment_id,
        spec=resolution.spec,
        schema=behavior_schema,
    )
    try:
        return behavior_schema.compile_segment_audience(
            spec=resolution.spec,
            calibration=calibration,
        )
    except ValueError as exc:
        raise _contract_error(
            "segment_audience_manifest_mismatch",
            segment_id,
            str(exc),
        ) from exc


def semantic_query_vector_hash(spec: CandidateBehaviorSpec) -> str:
    serialized = json.dumps(list(spec.query_vector), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_bundled_semantic_selection(
    *,
    segment_id: str,
) -> SemanticSelectionArtifact:
    return load_semantic_selection_artifact(
        path=BUNDLED_SEMANTIC_SELECTION_PATH,
        expected_sha256=BUNDLED_SEMANTIC_SELECTION_SHA256,
        segment_id=segment_id,
    )


def load_semantic_selection_artifact(
    *,
    path: Path,
    expected_sha256: str,
    segment_id: str,
) -> SemanticSelectionArtifact:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _contract_error(
            "segment_audience_calibration_missing",
            segment_id,
            "bundled semantic selection artifact is unavailable",
        ) from exc
    artifact_hash = hashlib.sha256(raw).hexdigest()
    if artifact_hash != expected_sha256:
        raise _contract_error(
            "segment_audience_calibration_hash_mismatch",
            segment_id,
            "bundled semantic selection artifact hash does not match code",
        )
    if not isinstance(payload, Mapping):
        raise _invalid_artifact(segment_id, "artifact must be an object")
    expected = {
        "status": "validated",
        "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
        "schema_version": SEGMENT_AUDIENCE_SCHEMA_VERSION,
        "behavior_schema_version": HOTEL_BEHAVIOR_SCHEMA_VERSION,
        "vector_version": HOTEL_BEHAVIOR_VECTOR_VERSION,
        "manifest_hash": HOTEL_BEHAVIOR_MANIFEST_HASH,
        "query_compiler_version": SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION,
        "query_compiler_hash": SEGMENT_AUDIENCE_QUERY_COMPILER_HASH,
        "user_vectorizer_version": USER_BEHAVIOR_VECTORIZER_VERSION,
        "user_vectorizer_semantic_hash": (
            USER_BEHAVIOR_VECTORIZER_SEMANTIC_HASH
        ),
        "semantic_selection_status": "validated",
        "business_lift_status": "pending",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise _invalid_artifact(
                segment_id,
                f"semantic selection artifact {key} is incompatible",
            )
    version = str(payload.get("artifact_version", "")).strip()
    minimum_margin = float(payload.get("minimum_margin", -1))
    raw_templates = payload.get("templates")
    if (
        not version
        or minimum_margin < 0.02
        or not isinstance(raw_templates, Mapping)
        or set(raw_templates) != set(REGISTERED_SEGMENT_AUDIENCE_TEMPLATES)
    ):
        raise _invalid_artifact(
            segment_id,
            "semantic selection artifact registry is incomplete",
        )
    templates: dict[str, _TemplateAnchors] = {}
    for template_id, template in REGISTERED_SEGMENT_AUDIENCE_TEMPLATES.items():
        item = raw_templates.get(template_id)
        if not isinstance(item, Mapping):
            raise _invalid_artifact(segment_id, f"missing anchors: {template_id}")
        try:
            anchors = _TemplateAnchors(
                template_id=template_id,
                template_version=int(item["template_version"]),
                selection_artifact_version=str(
                    item["selection_artifact_version"]
                ),
                selection_artifact_hash=_canonical_mapping_hash(item),
                template_semantic_hash=str(item["template_semantic_hash"]),
                semantic_selection_policy_id=str(
                    item["semantic_selection_policy_id"]
                ),
                semantic_anchor_policy_id=str(item["semantic_anchor_policy_id"]),
                accepted=_anchor_recipes(item, "accepted"),
                boundary_accepted=_anchor_recipes(item, "boundary_accepted"),
                boundary_rejected=_anchor_recipes(item, "boundary_rejected"),
                negative=_anchor_recipes(item, "negative"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _invalid_artifact(
                segment_id,
                f"invalid anchors: {template_id}",
            ) from exc
        if (
            not anchors.selection_artifact_version
            or anchors.template_version != template.template_version
            or anchors.template_semantic_hash != template.semantic_hash
            or anchors.semantic_selection_policy_id
            != template.semantic_selection_policy_id
            or anchors.semantic_anchor_policy_id
            != template.semantic_anchor_policy_id
        ):
            raise _invalid_artifact(
                segment_id,
                f"template anchor binding is incompatible: {template_id}",
            )
        templates[template_id] = anchors
    return SemanticSelectionArtifact(
        version=version,
        artifact_hash=artifact_hash,
        manifest_hash=HOTEL_BEHAVIOR_MANIFEST_HASH,
        minimum_margin=minimum_margin,
        semantic_selection_status="validated",
        business_lift_status="pending",
        templates=templates,
    )


def _canonical_mapping_hash(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _anchor_recipes(
    item: Mapping[str, Any],
    key: str,
) -> tuple[Mapping[str, Any], ...]:
    values = item.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} anchors are required")
    if any(not isinstance(value, Mapping) for value in values):
        raise ValueError(f"{key} anchors must be objects")
    return tuple(values)


def _instantiate_anchor(
    recipe: Mapping[str, Any],
    *,
    spec: SegmentAudienceSpec,
    index: int,
) -> RawEventUserSignalRecord:
    counts = recipe.get("counts", {})
    if not isinstance(counts, Mapping):
        raise ValueError("anchor counts must be an object")
    event_age_days = recipe.get("event_age_days")
    if (
        not isinstance(event_age_days, int)
        or isinstance(event_age_days, bool)
        or not 0 <= event_age_days <= 365
    ):
        raise ValueError("anchor event_age_days must be an integer from 0 to 365")
    destinations = _anchor_destinations(
        str(recipe.get("destination_mode", "none")),
        spec=spec,
    )
    checkin_dates = _anchor_checkin_dates(
        str(recipe.get("season_mode", "none")),
        spec=spec,
    )
    benefit_counts = _anchor_benefit_counts(
        str(recipe.get("benefit_mode", "none")),
        spec=spec,
        count=int(recipe.get("benefit_count", 0)),
    )
    values = {
        "event_count": 1,
        "hotel_search_count": 0,
        "hotel_click_count": 0,
        "hotel_detail_view_count": 0,
        "promotion_impression_count": 0,
        "promotion_click_count": 0,
        "campaign_redirect_click_count": 0,
        "campaign_landing_count": 0,
        "booking_start_count": 0,
        "booking_complete_count": 0,
        "booking_cancel_count": 0,
        "deal_event_count": 0,
        "free_cancellation_count": 0,
        "breakfast_included_count": 0,
        "price_event_count": 0,
    }
    unknown = set(counts) - set(values)
    if unknown:
        raise ValueError("unknown anchor count fields: " + ", ".join(sorted(unknown)))
    values.update({key: int(value) for key, value in counts.items()})
    for key, value in benefit_counts.items():
        values[key] = max(values[key], value)
    if values["event_count"] <= 0:
        raise ValueError("anchor event_count must be positive")
    lead_counts = _anchor_lead_counts(recipe, checkin_dates=checkin_dates)
    return RawEventUserSignalRecord(
        project_id="semantic-anchor",
        user_id=f"{spec.template_id}:{index}",
        avg_price=0.0,
        destination_values=destinations,
        checkin_dates=checkin_dates,
        hotel_market_values=(),
        hotel_cluster_values=(),
        age_group_values=(),
        gender_values=(),
        preferred_category_values=(),
        destination_match_count=0,
        season_match_count=0,
        hotel_search_recency_days=(
            event_age_days if values["hotel_search_count"] > 0 else None
        ),
        hotel_detail_recency_days=(
            event_age_days if values["hotel_detail_view_count"] > 0 else None
        ),
        booking_start_recency_days=(
            event_age_days if values["booking_start_count"] > 0 else None
        ),
        deal_recency_days=(
            event_age_days if values["deal_event_count"] > 0 else None
        ),
        promotion_response_recency_days=(
            event_age_days
            if values["promotion_click_count"]
            + values["campaign_landing_count"]
            > 0
            else None
        ),
        **lead_counts,
        **values,
    )


def _anchor_lead_counts(
    recipe: Mapping[str, Any],
    *,
    checkin_dates: Sequence[str],
) -> Mapping[str, int]:
    values = {
        "lead_time_0_7_count": 0,
        "lead_time_8_30_count": 0,
        "lead_time_gt_30_count": 0,
        "weekend_checkin_count": 0,
    }
    if not checkin_dates:
        if "checkin_lead_days" in recipe:
            raise ValueError("checkin_lead_days requires a season anchor")
        return values
    lead_days = recipe.get("checkin_lead_days")
    if (
        not isinstance(lead_days, int)
        or isinstance(lead_days, bool)
        or lead_days < 0
    ):
        raise ValueError("season anchor requires non-negative checkin_lead_days")
    count = len(checkin_dates)
    if lead_days <= 7:
        values["lead_time_0_7_count"] = count
    elif lead_days <= 30:
        values["lead_time_8_30_count"] = count
    else:
        values["lead_time_gt_30_count"] = count
    return values


def _anchor_destinations(
    mode: str,
    *,
    spec: SegmentAudienceSpec,
) -> tuple[str, ...]:
    matched = spec.destination_ids[0] if spec.destination_ids else "jeju"
    others = tuple(
        value
        for value in ("seoul", "busan", "gangneung", "gyeongju")
        if value not in spec.destination_ids and value != matched
    )
    if mode == "none":
        return ()
    if mode == "matched_once":
        return (matched,)
    if mode == "matched_twice":
        return (matched, matched)
    if mode == "matched_dense":
        return (matched,) * 6
    if mode == "broad_matched":
        return (matched, others[0], others[1])
    if mode == "broad":
        return (others[0], others[1], others[2])
    if mode == "single_other":
        return (others[0],)
    raise ValueError(f"unsupported anchor destination_mode: {mode}")


def _anchor_checkin_dates(
    mode: str,
    *,
    spec: SegmentAudienceSpec,
) -> tuple[str, ...]:
    matched_month = spec.season_months[0] if spec.season_months else 7
    other_month = next(month for month in range(1, 13) if month != matched_month)
    if mode == "none":
        return ()
    if mode == "matched":
        return (_date_for_month(matched_month),)
    if mode == "matched_dense":
        return (_date_for_month(matched_month),) * 3
    if mode == "other":
        return (_date_for_month(other_month),)
    raise ValueError(f"unsupported anchor season_mode: {mode}")


def _anchor_benefit_counts(
    mode: str,
    *,
    spec: SegmentAudienceSpec,
    count: int,
) -> Mapping[str, int]:
    if mode == "none":
        return {}
    keys = spec.benefit_keys or ("discount",)
    if mode == "selected":
        selected = keys
    elif mode == "other":
        selected = ("breakfast_included",)
    else:
        raise ValueError(f"unsupported anchor benefit_mode: {mode}")
    result: dict[str, int] = {}
    for key in selected:
        if key in {"discount", "early_booking"}:
            result["deal_event_count"] = max(
                result.get("deal_event_count", 0),
                count,
            )
            result["price_event_count"] = max(
                result.get("price_event_count", 0),
                count,
            )
        elif key == "free_cancellation":
            result["free_cancellation_count"] = count
        elif key == "breakfast_included":
            result["breakfast_included_count"] = count
    return result


def _matches_hard_predicates(
    profile: RawEventUserSignalRecord,
    *,
    spec: SegmentAudienceSpec,
) -> bool:
    destinations = tuple(profile.destination_values)
    destination_set = set(spec.destination_ids)
    months = tuple(
        parsed.month
        for value in profile.checkin_dates
        if (parsed := _parse_date(value)) is not None
    )
    for key in spec.hard_predicate_keys:
        if key == "hotel_product_interest":
            matched = (
                profile.hotel_search_count
                + profile.hotel_click_count
                + profile.hotel_detail_view_count
                > 0
            )
        elif key == "target_destination_affinity":
            matched = sum(value in destination_set for value in destinations) >= 2
        elif key == "recent_destination_search":
            matched = any(value in destination_set for value in destinations)
        elif key == "booking_start_without_complete":
            matched = (
                profile.booking_start_count > profile.booking_complete_count
                and profile.booking_start_count > 0
            )
        elif key == "benefit_interest":
            selected = set(spec.benefit_keys)
            matched = _matches_benefit(profile, selected)
        elif key == "promotion_response":
            matched = (
                profile.promotion_click_count + profile.campaign_landing_count > 0
            )
        elif key == "general_destination_exploration":
            matched = len(set(destinations)) >= 2
        elif key == "season_match":
            matched = any(month in spec.season_months for month in months)
        else:
            raise ValueError(f"unsupported semantic anchor predicate: {key}")
        if not matched:
            return False
    return True


def _matches_benefit(
    profile: RawEventUserSignalRecord,
    selected: set[str],
) -> bool:
    if not selected:
        return (
            profile.deal_event_count
            + profile.price_event_count
            + profile.free_cancellation_count
            + profile.breakfast_included_count
            > 0
        )
    return (
        bool(selected & {"discount", "early_booking"})
        and profile.deal_event_count + profile.price_event_count > 0
    ) or (
        "free_cancellation" in selected
        and profile.free_cancellation_count > 0
    ) or (
        "breakfast_included" in selected
        and profile.breakfast_included_count > 0
    )


def _instantiated_anchor_hash(
    *,
    spec: SegmentAudienceSpec,
    accepted: Sequence[RawEventUserSignalRecord],
    rejected: Sequence[RawEventUserSignalRecord],
    negative: Sequence[RawEventUserSignalRecord],
) -> str:
    payload = {
        "spec_hash": spec.spec_hash,
        "accepted": [_profile_payload(value) for value in accepted],
        "boundary_rejected": [_profile_payload(value) for value in rejected],
        "negative": [_profile_payload(value) for value in negative],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _profile_payload(profile: RawEventUserSignalRecord) -> Mapping[str, Any]:
    return {
        name: getattr(profile, name)
        for name in profile.__dataclass_fields__
        if name not in {"project_id", "user_id"}
    }


def _date_for_month(month: int) -> str:
    return date(2026, month, 15).isoformat()


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _contract_error(
    code: str,
    segment_id: str,
    reason: str,
) -> SegmentAudienceContractError:
    return SegmentAudienceContractError(
        code=code,
        segment_id=segment_id,
        reason=reason,
    )


def _invalid_artifact(
    segment_id: str,
    reason: str,
) -> SegmentAudienceContractError:
    return _contract_error(
        "segment_audience_calibration_invalid",
        segment_id,
        reason,
    )
