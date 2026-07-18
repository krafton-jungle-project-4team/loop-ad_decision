from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json

import pytest

from app.analysis.audience_search import (
    AudienceSearchResult,
    AudienceSearchMethod,
    CandidateAudienceSearchService,
    SearchCandidate,
    SearchPolicy,
    threshold_recall_audit,
)
from app.analysis.audience_snapshot_repository import (
    AudienceSnapshotRepository as AnalysisAudienceSnapshotRepository,
    AudienceSnapshotWrite,
    _input_fingerprint,
    _spec_fingerprint,
)
from app.analysis.behavior_vector_schema import (
    CandidateCalibration,
    HotelBookingBehaviorSchemaV2,
    canonical_destination,
    cosine_similarity,
    signed_hash_coordinate,
    HOTEL_BEHAVIOR_MANIFEST_HASH,
)
from app.analysis.behavior_manifest import (
    BehaviorManifestError,
    behavior_manifest_hash,
    clickhouse_canonical_destination_sql,
    load_behavior_manifest,
    order_vector_terms_by_manifest,
)
from app.analysis.audience_v2 import (
    BUNDLED_AUDIENCE_CALIBRATION_PATH,
    BUNDLED_AUDIENCE_CALIBRATION_SHA256,
    AudienceV2Coordinator,
    BundledCandidateCalibrationProvider,
    _load_candidate_calibrations,
    load_bundled_candidate_calibrations,
)
from app.analysis.semantic_selection import (
    _instantiate_anchor,
    _matches_hard_predicates,
    compile_registered_segment_audience,
    load_bundled_semantic_selection,
    semantic_query_vector_hash,
)
from app.analysis.segment_audience_templates import (
    REGISTERED_SEGMENT_AUDIENCE_TEMPLATES,
    RegisteredSegmentAudienceBinder,
)
from app.analysis.audience_search_repository import (
    AudienceSearchContext,
    HardMatchAggregateRequest,
    PgClickHouseAudienceVectorSearchRepository,
    _hard_predicate_batch_query,
    _hard_predicate_query,
)
from app.analysis.raw_event_segments import PromotionIntent
from app.analysis.repositories import (
    PromotionRecord as AnalysisPromotionRecord,
    RawEventUserSignalRecord,
    SegmentDefinitionRecord,
)
from app.audience_contract import (
    SEGMENT_AUDIENCE_CONTRACT,
    SEGMENT_AUDIENCE_QUERY_COMPILER_HASH,
    SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION,
    SEGMENT_AUDIENCE_SCHEMA_VERSION,
    SegmentAudienceContractError,
    SegmentDefinitionAudienceAdapter,
)
from app.audience_exclusions import PromotionAudienceExclusionContext
from app.analysis.vector_service import SegmentVectorBuildResult
from app.decision.assignment_service import SegmentAssignmentService
from app.decision.audience_snapshots import (
    AudienceSnapshotContractError,
    AudienceSnapshotMember,
    AudienceSnapshotRepository as DecisionAudienceSnapshotRepository,
    AudienceSnapshotSet,
    TargetAudienceResolution,
)
from app.decision.matcher import SegmentCandidateReranker
from app.decision.repositories import (
    AdExperimentRecord,
    PromotionRunRecord,
    UserSegmentAssignmentInsertRecord,
)
from app.decision.schemas import AssignmentSource, SegmentAssignmentBuildRequest
from app.internal.user_behavior_vector_search_sync import (
    SearchVectorRevision,
    UserBehaviorVectorSearchSyncRepository,
    UserBehaviorVectorSearchSyncService,
    VectorSearchGeneration,
    VectorSyncCursor,
    _validate_revision,
)


def test_query_and_user_destination_use_same_signed_hash_coordinate() -> None:
    assert canonical_destination("  JEJU  ") == "jeju"
    assert signed_hash_coordinate("JEJU") == signed_hash_coordinate(" jeju ")

    schema = HotelBookingBehaviorSchemaV2()
    spec = schema.compile_candidate(
        candidate_type="target_destination_affinity",
        intent=_intent(destinations=("JEJU",)),
        calibration=CandidateCalibration(score_threshold=0.55, version="test.v1"),
    )
    destination_index, destination_sign = signed_hash_coordinate("jeju")
    assert spec.query_vector[destination_index] * destination_sign > 0
    assert spec.vector_version == "hotel_behavior.v2"
    assert spec.hard_predicate_keys == (
        "target_destination_affinity",
        "hotel_product_interest",
        "recent_destination_search",
    )
    assert abs(cosine_similarity(spec.query_vector, spec.query_vector) - 1.0) < 1e-9


def test_manifest_fixes_all_dimensions_and_destination_alias_sql() -> None:
    manifest = load_behavior_manifest()
    assert manifest["vector_dim"] == 64
    assert manifest["query_block_normalization"] == (
        "l2_per_active_block_then_weight_then_global_l2"
    )
    assert [item["index"] for item in manifest["dimensions"]] == list(range(64))
    assert behavior_manifest_hash() == HOTEL_BEHAVIOR_MANIFEST_HASH
    assert canonical_destination(" 제주도 ") == "jeju"
    assert signed_hash_coordinate("제주도") == signed_hash_coordinate("jeju")
    sql = clickhouse_canonical_destination_sql("destination_value")
    assert "multiIf(" in sql
    assert "'제주도'" in sql
    assert "'jeju'" in sql

    names = [str(item["name"]) for item in manifest["dimensions"]]
    ordered = order_vector_terms_by_manifest({name: name for name in names})
    assert ordered == tuple(names)
    with pytest.raises(BehaviorManifestError, match="missing="):
        order_vector_terms_by_manifest(
            {name: name for name in names if name != names[-1]}
        )


def test_hard_match_population_is_frozen_by_vector_revision_cutoff() -> None:
    sql = _hard_predicate_query(
        ("hotel_product_interest",),
        filter_user_ids=False,
        restrict_to_vector_population=True,
    )
    assert "FROM user_behavior_vector_revisions" in sql
    assert "ingested_at <=" in sql
    assert "vector_version = {vector_version:String}" in sql
    assert "received_at <=" in sql
    assert "raw_event_received_cutoff" in sql


def test_hard_match_rate_sample_uses_stable_salted_order() -> None:
    sql = _hard_predicate_query(
        ("hotel_product_interest",),
        filter_user_ids=False,
        restrict_to_vector_population=True,
        deterministic_sample=True,
    )
    assert "cityHash64(concat(user_id, {sample_seed:String}))" in sql
    assert "user_id ASC" in sql


def test_three_same_scope_predicates_share_one_batch_aggregate() -> None:
    query, parameters = _hard_predicate_batch_query(
        tuple(
            HardMatchAggregateRequest(
                segment_id=f"segment_{index}",
                hard_predicate_keys=("booking_start_without_complete",),
                predicate_parameters={},
            )
            for index in range(3)
        )
    )
    assert query.count("FROM raw_events") == 1
    assert query.count("FROM per_user") == 1
    assert all(f"AS match_{index}" in query for index in range(3))
    assert len(parameters) == 9


def test_batch_and_individual_destination_exploration_share_canonical_values() -> None:
    batch_query, _parameters = _hard_predicate_batch_query(
        (
            HardMatchAggregateRequest(
                segment_id="segment_explorer",
                hard_predicate_keys=("general_destination_exploration",),
                predicate_parameters={},
            ),
        )
    )
    individual_query = _hard_predicate_query(
        ("general_destination_exploration",),
        filter_user_ids=False,
        restrict_to_vector_population=True,
    )

    assert "multiIf(" in batch_query
    assert "multiIf(" in individual_query
    assert "hotel_market" not in individual_query
    assert "!= ''" in batch_query
    assert "!= ''" in individual_query


def test_segment_spec_query_is_deterministic_and_does_not_read_promotion() -> None:
    segment = _v2_segment("segment_a")
    resolution = SegmentDefinitionAudienceAdapter().resolve(
        segment_id=segment.segment_id,
        rule_json=segment.rule_json,
    )
    assert resolution.spec is not None
    schema = HotelBookingBehaviorSchemaV2()
    calibration = BundledCandidateCalibrationProvider().require(
        segment_id=segment.segment_id,
        spec=resolution.spec,
        schema=schema,
    )
    first = schema.compile_segment_audience(
        spec=resolution.spec,
        calibration=calibration,
    )
    changed_promotion = AnalysisPromotionRecord(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        channel="sms",
        goal_metric="inflow_rate",
        goal_target_value=Decimal("0.2"),
        goal_basis="all_segments",
        min_sample_size=50,
        landing_url="https://changed.example",
        message_brief="completely different promotion text",
    )
    assert changed_promotion.message_brief
    second = schema.compile_segment_audience(
        spec=resolution.spec,
        calibration=calibration,
    )
    assert first.query_vector == second.query_vector
    assert first.segment_audience_spec_hash == second.segment_audience_spec_hash


def test_registered_segment_query_never_uses_legacy_promotion_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("registered segment query used promotion compiler")

    monkeypatch.setattr(
        HotelBookingBehaviorSchemaV2,
        "compile_candidate",
        fail_if_called,
    )
    segment = _v2_segment(
        "segment_registered_funnel",
        candidate_type="funnel_recovery",
    )

    compiled = compile_registered_segment_audience(
        segment_id=segment.segment_id,
        rule_json=segment.rule_json,
    )

    assert compiled.audience_resolution_contract == SEGMENT_AUDIENCE_CONTRACT
    assert compiled.template_id == "hotel.funnel_recovery.v1"
    assert (
        compiled.query_compiler_version
        == SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION
    )
    assert cosine_similarity(
        compiled.query_vector,
        compiled.query_vector,
    ) == pytest.approx(
        1.0,
    )


def test_v2_contract_never_infers_a_missing_spec_from_promotion() -> None:
    with pytest.raises(SegmentAudienceContractError) as error:
        SegmentDefinitionAudienceAdapter().resolve(
            segment_id="segment_missing",
            rule_json={
                "audience_resolution_contract": "segment_audience.v1",
                "promotion_intent": {"destinations": ["jeju"]},
            },
        )

    assert error.value.code == "segment_audience_spec_missing"


def test_v2_contract_rejects_query_signals_outside_candidate_calibration() -> None:
    segment = _v2_segment("segment_incompatible")
    raw_spec = segment.rule_json["segment_audience_spec"]
    assert isinstance(raw_spec, dict)
    incompatible_spec = {
        **raw_spec,
        "query_signal_keys": ["promotion_click_intensity"],
    }

    with pytest.raises(SegmentAudienceContractError) as error:
        SegmentDefinitionAudienceAdapter().resolve(
            segment_id=segment.segment_id,
            rule_json={
                "audience_resolution_contract": "segment_audience.v1",
                "segment_audience_spec": incompatible_spec,
            },
        )

    assert error.value.code == "segment_audience_template_binding_mismatch"


def test_six_registered_templates_require_exact_signals_predicates_and_policies(
) -> None:
    binder = RegisteredSegmentAudienceBinder()
    for template in REGISTERED_SEGMENT_AUDIENCE_TEMPLATES.values():
        destinations = ("jeju",) if template.destination_min else ()
        raw_spec = binder.bind(
            candidate_type=template.candidate_type,
            destination_ids=destinations,
        )
        resolution = SegmentDefinitionAudienceAdapter().resolve(
            segment_id=f"segment_{template.candidate_type}",
            rule_json={
                "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
                "segment_audience_spec": raw_spec,
            },
        )
        assert resolution.spec is not None
        assert resolution.spec.query_signal_keys == template.query_signal_keys
        assert resolution.spec.hard_predicate_keys == template.hard_predicate_keys(
            destination_ids=destinations,
            season_months=(),
        )
        assert resolution.spec.parameter_policy_id == template.parameter_policy_id
        assert (
            resolution.spec.semantic_selection_policy_id
            == template.semantic_selection_policy_id
        )
        assert (
            resolution.spec.semantic_anchor_policy_id
            == template.semantic_anchor_policy_id
        )


def test_registered_template_parameters_canonicalize_before_hash_and_query() -> None:
    binder = RegisteredSegmentAudienceBinder()
    first = binder.bind(
        candidate_type="intent_matched",
        destination_ids=("제주", "busan", "jeju"),
        season_months=(8, 6, 7, 6),
    )
    second = binder.bind(
        candidate_type="intent_matched",
        destination_ids=("busan", "jeju"),
        season_months=(6, 7, 8),
    )
    first_segment = {
        "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
        "segment_audience_spec": first,
    }
    second_segment = {
        "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
        "segment_audience_spec": second,
    }

    first_compiled = compile_registered_segment_audience(
        segment_id="segment_same",
        rule_json=first_segment,
    )
    second_compiled = compile_registered_segment_audience(
        segment_id="segment_same",
        rule_json=second_segment,
    )
    assert first == second
    assert first_compiled.segment_audience_spec_hash == (
        second_compiled.segment_audience_spec_hash
    )
    assert first_compiled.query_vector == second_compiled.query_vector


def test_query_compiler_hash_is_not_coupled_to_unrelated_template_registry(
) -> None:
    compiler_semantics = {
        "version": SEGMENT_AUDIENCE_QUERY_COMPILER_VERSION,
        "schema_version": SEGMENT_AUDIENCE_SCHEMA_VERSION,
        "behavior_manifest_hash": behavior_manifest_hash(),
        "input": "registered_segment_audience_template_only",
        "destination_encoding": "canonical_id_sha256_signed_bucket",
        "normalization": "manifest_block_weight_then_global_l2",
        "predicate_policy": "registered_template_exact_binding",
    }
    expected = hashlib.sha256(
        json.dumps(
            compiler_semantics,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert SEGMENT_AUDIENCE_QUERY_COMPILER_HASH == expected


def test_unregistered_template_never_reaches_signature_execution() -> None:
    raw_spec = dict(
        RegisteredSegmentAudienceBinder().bind(
            candidate_type="promotion_responsive"
        )
    )
    raw_spec["template_id"] = "hotel.custom_signature.v1"

    with pytest.raises(SegmentAudienceContractError) as error:
        compile_registered_segment_audience(
            segment_id="segment_custom",
            rule_json={
                "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
                "segment_audience_spec": raw_spec,
            },
        )

    assert error.value.code == "segment_audience_template_unregistered"


def test_coordinator_batches_three_segments_once() -> None:
    search = _BatchCoordinatorSearchRepository()
    snapshots = _BatchSnapshotWriter()
    coordinator = AudienceV2Coordinator(
        search_repository=search,
        snapshot_repository=snapshots,
        segment_vector_service=_BatchSegmentVectorPreparer(),
        calibration_provider=_TestCalibrationProvider(),
    )
    prepared = coordinator.prepare_many(
        analysis_id="analysis",
        promotion=_analysis_promotion(),
        segments=tuple(_v2_segment(f"segment_{index}") for index in range(3)),
    )
    assert set(prepared) == {"segment_0", "segment_1", "segment_2"}
    assert search.batch_call_count == 1
    assert search.individual_call_count == 0
    assert len(snapshots.writes) == 3


def test_coordinator_uses_final_members_as_behavior_match_lower_bound() -> None:
    search = _BatchCoordinatorSearchRepository(
        hard_match_count=0,
        materialized_member_count=2,
    )
    snapshots = _BatchSnapshotWriter()
    coordinator = AudienceV2Coordinator(
        search_repository=search,
        snapshot_repository=snapshots,
        segment_vector_service=_BatchSegmentVectorPreparer(),
        calibration_provider=_TestCalibrationProvider(),
    )

    prepared = coordinator.prepare_many(
        analysis_id="analysis",
        promotion=_analysis_promotion(),
        segments=(_v2_segment("segment"),),
    )["segment"]

    assert prepared.matching_user_count == 2
    assert prepared.selected_user_count == 2
    assert snapshots.writes[0].search_result.hard_match_user_count == 2


def test_six_registered_templates_build_expected_exact_snapshot_members() -> None:
    schema = HotelBookingBehaviorSchemaV2()
    artifact = load_bundled_semantic_selection(segment_id="demo_validation")
    segments = tuple(
        _v2_segment(
            f"segment_{candidate_type}",
            candidate_type=candidate_type,
        )
        for candidate_type in (
            "intent_matched",
            "target_destination_affinity",
            "funnel_recovery",
            "benefit_value_seeker",
            "promotion_responsive",
            "general_destination_explorer",
        )
    )
    specs_by_segment = {}
    queries_by_segment = {}
    profiles = []
    expected_positive_user_ids = {}
    expected_negative_user_ids = {}
    for segment in segments:
        resolution = SegmentDefinitionAudienceAdapter().resolve(
            segment_id=segment.segment_id,
            rule_json=segment.rule_json,
        )
        assert resolution.spec is not None
        calibration = artifact.calibration_for(
            segment_id=segment.segment_id,
            spec=resolution.spec,
            schema=schema,
        )
        query = schema.compile_segment_audience(
            spec=resolution.spec,
            calibration=calibration,
        )
        anchors = artifact.templates[resolution.spec.template_id]
        positive = _instantiate_anchor(
            anchors.accepted[0],
            spec=resolution.spec,
            index=0,
        )
        negative = _instantiate_anchor(
            anchors.negative[0],
            spec=resolution.spec,
            index=200,
        )
        specs_by_segment[segment.segment_id] = resolution.spec
        queries_by_segment[segment.segment_id] = query
        profiles.extend((positive, negative))
        expected_positive_user_ids[segment.segment_id] = positive.user_id
        expected_negative_user_ids[segment.segment_id] = negative.user_id

    search = _SemanticCorpusSearchRepository(
        schema=schema,
        profiles=tuple(profiles),
        specs_by_segment=specs_by_segment,
        queries_by_segment=queries_by_segment,
    )
    snapshots = _BatchSnapshotWriter()
    coordinator = AudienceV2Coordinator(
        search_repository=search,
        snapshot_repository=snapshots,
        segment_vector_service=_BatchSegmentVectorPreparer(),
        schema=schema,
    )

    prepared = coordinator.prepare_many(
        analysis_id="analysis_demo",
        promotion=_analysis_promotion(),
        segments=segments,
    )

    assert search.batch_call_count == 1
    assert search.individual_call_count == 0
    assert search.exact_call_count == 6
    assert len(prepared) == len(snapshots.writes) == 6
    writes_by_segment = {write.segment_id: write for write in snapshots.writes}
    for segment in segments:
        write = writes_by_segment[segment.segment_id]
        member_ids = {member.user_id for member in write.search_result.members}
        assert expected_positive_user_ids[segment.segment_id] in member_ids
        assert expected_negative_user_ids[segment.segment_id] not in member_ids
        assert prepared[segment.segment_id].selected_user_count == len(member_ids)
        assert prepared[segment.segment_id].selected_user_count > 0

    demo_segment_ids = {
        "segment_funnel_recovery",
        "segment_benefit_value_seeker",
        "segment_promotion_responsive",
    }
    for segment_id in demo_segment_ids:
        positive_user_id = expected_positive_user_ids[segment_id]
        overlap_scores = {
            candidate_segment_id: member.behavior_fit_score
            for candidate_segment_id, write in writes_by_segment.items()
            if candidate_segment_id in demo_segment_ids
            for member in write.search_result.members
            if member.user_id == positive_user_id
        }
        assert max(overlap_scores, key=overlap_scores.get) == segment_id


def test_semantic_selection_is_manifest_bound_and_business_lift_pending(
    tmp_path,
) -> None:
    registry = load_bundled_candidate_calibrations(segment_id="segment")
    assert registry.artifact_hash == BUNDLED_AUDIENCE_CALIBRATION_SHA256
    assert registry.semantic_selection_status == "validated"
    assert registry.business_lift_status == "pending"
    template_artifact_hashes: set[str] = set()
    for candidate_type in (
        "intent_matched",
        "target_destination_affinity",
        "funnel_recovery",
        "benefit_value_seeker",
        "promotion_responsive",
        "general_destination_explorer",
    ):
        segment = _v2_segment(
            f"segment_{candidate_type}",
            candidate_type=candidate_type,
        )
        resolution = SegmentDefinitionAudienceAdapter().resolve(
            segment_id=segment.segment_id,
            rule_json=segment.rule_json,
        )
        assert resolution.spec is not None
        calibration = registry.calibration_for(
            segment_id=segment.segment_id,
            spec=resolution.spec,
            schema=HotelBookingBehaviorSchemaV2(),
        )
        assert calibration.score_threshold > 0
        assert calibration.semantic_margin >= 0.02
        assert calibration.business_lift_status == "pending"
        assert calibration.version.endswith("semantic-selection.v1")
        template_artifact_hashes.add(calibration.artifact_hash)

    assert len(template_artifact_hashes) == 6

    artifact = tmp_path / "calibration.json"
    raw = BUNDLED_AUDIENCE_CALIBRATION_PATH.read_bytes()
    artifact.write_bytes(raw)
    with pytest.raises(SegmentAudienceContractError) as mismatch:
        _load_candidate_calibrations(
            path=artifact,
            expected_sha256=hashlib.sha256(raw + b"changed").hexdigest(),
            segment_id="segment",
        )
    assert mismatch.value.code == "segment_audience_calibration_hash_mismatch"


def test_candidate_compiler_renormalizes_when_optional_blocks_are_missing() -> None:
    schema = HotelBookingBehaviorSchemaV2()
    spec = schema.compile_candidate(
        candidate_type="intent_matched",
        intent=_intent(),
        calibration=CandidateCalibration(score_threshold=0.5, version="test.v1"),
    )
    assert "destination" not in spec.active_blocks
    assert "timing" not in spec.active_blocks
    assert sum(spec.block_weights.values()) == 1.0


def test_candidate_query_block_norms_follow_declared_weights() -> None:
    spec = HotelBookingBehaviorSchemaV2().compile_candidate(
        candidate_type="target_destination_affinity",
        intent=_intent(destinations=("jeju",)),
        calibration=CandidateCalibration(score_threshold=0.5, version="test.v1"),
    )
    manifest = load_behavior_manifest()
    norms = {}
    for block, bounds in manifest["blocks"].items():
        start, end = bounds
        norms[block] = sum(
            spec.query_vector[index] ** 2 for index in range(start, end + 1)
        ) ** 0.5

    assert norms["destination"] / norms["funnel"] == pytest.approx(3.0)
    assert norms["destination"] / norms["derived"] == pytest.approx(3.0)


def test_behavior_query_scores_semantically_matching_user_higher() -> None:
    schema = HotelBookingBehaviorSchemaV2()
    spec = schema.compile_candidate(
        candidate_type="target_destination_affinity",
        intent=_intent(destinations=("jeju",)),
        calibration=CandidateCalibration(0.5, "test.v1"),
    )
    matching = schema.vectorize_user(
        _profile(
            user_id="matching",
            hotel_search_count=5,
            hotel_detail_view_count=4,
            destination_values=("jeju", "jeju"),
            destination_match_count=2,
        )
    )
    unrelated = schema.vectorize_user(
        _profile(
            user_id="unrelated",
            promotion_impression_count=5,
            promotion_click_count=3,
            campaign_landing_count=2,
            destination_values=("busan",),
        )
    )
    assert cosine_similarity(spec.query_vector, matching) > cosine_similarity(
        spec.query_vector,
        unrelated,
    )


def test_benefit_query_and_exact_predicate_follow_promotion_benefit() -> None:
    manifest = load_behavior_manifest()
    dimensions = {
        str(item["name"]): int(item["index"])
        for item in manifest["dimensions"]
    }
    spec = HotelBookingBehaviorSchemaV2().compile_candidate(
        candidate_type="benefit_value_seeker",
        intent=_intent(benefits=("free_cancellation",)),
        calibration=CandidateCalibration(0.5, "test.v1"),
    )
    assert spec.query_vector[dimensions["free_cancellation_interest"]] > 0
    assert spec.query_vector[dimensions["breakfast_interest"]] == 0
    assert spec.query_vector[dimensions["deal_interest_intensity"]] == 0
    assert spec.predicate_parameters["benefit_keys"] == (
        "free_cancellation",
    )
    predicate_sql = _hard_predicate_query(("benefit_interest",))
    assert "{benefit_keys:Array(String)}" in predicate_sql
    assert "free_cancellation" in predicate_sql


def test_recall_gate_uses_confidence_lower_bound() -> None:
    passed = threshold_recall_audit(
        retrieved_positive_count=100_000,
        audited_nonretrieved_count=10_000,
        audited_missed_positive_count=0,
        nonretrieved_population_count=900_000,
    )
    failed = threshold_recall_audit(
        retrieved_positive_count=10_000,
        audited_nonretrieved_count=10_000,
        audited_missed_positive_count=100,
        nonretrieved_population_count=900_000,
    )
    assert passed.passed is True
    assert failed.passed is False


def test_ann_sql_is_pinned_to_requested_generation() -> None:
    postgres = _SearchSqlPostgres()
    repository = PgClickHouseAudienceVectorSearchRepository(
        postgres=postgres,
        clickhouse=_UnusedRepository(),
    )

    repository.ann_search(
        project_id="project",
        vector_generation_id="uvgen_frozen",
        vector_version="hotel_behavior.v2",
        source_cutoff=datetime(2026, 7, 16, tzinfo=UTC),
        query_vector=tuple([1.0, *([0.0] * 63)]),
        limit=100,
    )

    query, params = postgres.fetchall_calls[0]
    assert "vector_generation_id = %s" in query
    assert "status = 'activated'" not in query
    assert "uvgen_frozen" in params
    assert postgres.executed == [
        ("SET LOCAL hnsw.ef_search = 100", ()),
        ("SET LOCAL hnsw.iterative_scan = 'strict_order'", ()),
        ("SET LOCAL hnsw.max_scan_tuples = 20000", ()),
    ]


def test_materialized_ann_uses_server_side_relation_and_returns_only_count() -> None:
    postgres = _SearchSqlPostgres()
    repository = PgClickHouseAudienceVectorSearchRepository(
        postgres=postgres,
        clickhouse=_UnusedRepository(),
    )
    count = repository.materialize_ann_candidates(
        project_id="project",
        vector_generation_id="uvgen_frozen",
        vector_version="hotel_behavior.v2",
        source_cutoff=datetime(2026, 7, 16, tzinfo=UTC),
        query_vector=tuple([1.0, *([0.0] * 63)]),
        limit=100_000,
    )
    assert count == 100_000
    assert postgres.fetchall_calls == []
    assert any(
        "CREATE TEMP TABLE audience_ann_retrieval" in query
        and "LIMIT %s" in query
        for query, _params in postgres.executed
    )


def test_materialized_exact_members_use_postgres_array_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres = _SearchSqlPostgres(
        fetchall_results=[
            [
                {
                    "user_id": "user_1",
                    "behavior_fit_score": 0.9,
                    "retrieval_rank": 1,
                },
                {
                    "user_id": "user_2",
                    "behavior_fit_score": 0.8,
                    "retrieval_rank": 2,
                },
            ]
        ]
    )
    repository = PgClickHouseAudienceVectorSearchRepository(
        postgres=postgres,
        clickhouse=_UnusedRepository(),
    )
    monkeypatch.setattr(
        repository,
        "_filter_hard_predicates",
        lambda **kwargs: list(kwargs["candidates"]),
    )

    repository._materialize_hard_filtered_relation(
        project_id="project",
        vector_generation_id="uvgen_frozen",
        vector_version="hotel_behavior.v2",
        source_cutoff=datetime(2026, 7, 16, tzinfo=UTC),
        hard_predicate_keys=("promotion_response",),
        predicate_parameters={},
        candidate_relation="audience_exact_candidates",
        member_relation="audience_exact_members",
        score_threshold=None,
    )

    _query, params = next(
        (query, params)
        for query, params in postgres.executed
        if "FROM unnest(" in query
    )
    assert params == (["user_1", "user_2"], [0.9, 0.8], [1, 2])


def test_temp_user_ids_use_postgres_array_parameters() -> None:
    postgres = _SearchSqlPostgres()
    repository = PgClickHouseAudienceVectorSearchRepository(
        postgres=postgres,
        clickhouse=_UnusedRepository(),
    )

    repository._replace_temp_user_ids(
        table_name="audience_ann_candidates",
        user_ids=("user_1", "user_2"),
    )

    _query, params = next(
        (query, params)
        for query, params in postgres.executed
        if "FROM unnest(%s::text[])" in query
    )
    assert params == (["user_1", "user_2"],)


def test_large_search_falls_back_to_exact_when_recall_gate_fails() -> None:
    repository = _SearchRepository(always_miss=True)
    service = CandidateAudienceSearchService(
        repository,
        policy=SearchPolicy(
            exact_user_limit=10,
            transition_user_limit=20,
            min_ann_candidates=5,
            max_ann_corpus_fraction=0.25,
            min_audit_sample=10,
            max_audit_sample=10,
        ),
    )
    result = service.search(
        project_id="project",
        vector_generation_id="uvgen_test",
        source_cutoff="cutoff",
        spec=HotelBookingBehaviorSchemaV2().compile_candidate(
            candidate_type="promotion_responsive",
            intent=_intent(),
            calibration=CandidateCalibration(0.5, "test.v1"),
        ),
        corpus_user_count=100,
        hard_match_user_count=20,
        estimated_score_pass_rate=0.5,
    )
    assert result.method == AudienceSearchMethod.EXACT_FALLBACK
    assert repository.ann_limits == [15]
    assert [member.user_id for member in result.members] == ["exact_user"]


def test_transition_keeps_exact_members_and_records_ann_shadow_recall() -> None:
    repository = _SearchRepository(always_miss=False)
    service = CandidateAudienceSearchService(
        repository,
        policy=SearchPolicy(
            exact_user_limit=10,
            transition_user_limit=100,
            min_ann_candidates=5,
        ),
    )
    result = service.search(
        project_id="project",
        vector_generation_id="uvgen_test",
        source_cutoff="cutoff",
        spec=HotelBookingBehaviorSchemaV2().compile_candidate(
            candidate_type="promotion_responsive",
            intent=_intent(),
            calibration=CandidateCalibration(0.5, "test.v1"),
        ),
        corpus_user_count=50,
        hard_match_user_count=10,
        estimated_score_pass_rate=0.5,
    )
    assert result.method == AudienceSearchMethod.TRANSITION
    assert [member.user_id for member in result.members] == ["exact_user"]
    assert result.recall_audit is not None
    assert repository.ann_limits == [8]


def test_large_production_search_keeps_members_in_temp_relation() -> None:
    repository = _MaterializedSearchRepository()
    service = CandidateAudienceSearchService(
        repository,
        policy=SearchPolicy(
            exact_user_limit=10,
            transition_user_limit=20,
            min_ann_candidates=900,
            max_ann_corpus_fraction=1.0,
            min_audit_sample=10_000,
            max_audit_sample=10_000,
        ),
    )
    result = service.search(
        project_id="project",
        vector_generation_id="uvgen_test",
        source_cutoff="cutoff",
        spec=HotelBookingBehaviorSchemaV2().compile_candidate(
            candidate_type="promotion_responsive",
            intent=_intent(),
            calibration=CandidateCalibration(0.5, "test.v1"),
        ),
        corpus_user_count=1000,
        hard_match_user_count=900,
        estimated_score_pass_rate=0.5,
    )
    assert result.method == AudienceSearchMethod.ANN
    assert result.members == ()
    assert result.members_relation == "audience_ann_members"
    assert result.final_user_count == 800
    assert repository.legacy_call_count == 0


def test_snapshot_bulk_copies_materialized_members_without_python_list() -> None:
    db = _SnapshotWriteDb()
    repository = AnalysisAudienceSnapshotRepository(db)
    now = datetime(2026, 7, 16, tzinfo=UTC)
    spec = HotelBookingBehaviorSchemaV2().compile_candidate(
        candidate_type="promotion_responsive",
        intent=_intent(),
        calibration=CandidateCalibration(0.5, "test.v1"),
    )
    snapshot_id = repository.save_completed(
        AudienceSnapshotWrite(
            analysis_id="analysis",
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
            segment_id="segment",
            segment_vector_id="segment_vector",
            vector_generation_id="uvgen_test",
            source_cutoff=now,
            window_start=now - timedelta(days=90),
            window_end=now,
            spec=spec,
            search_result=AudienceSearchResult(
                method=AudienceSearchMethod.ANN,
                members=(),
                corpus_user_count=1000,
                hard_match_user_count=900,
                requested_k=900,
                recall_audit=None,
                policy_version="audience_search.v2",
                materialized_member_count=800,
                members_relation="audience_ann_members",
            ),
            min_sample_size=100,
        )
    )
    assert snapshot_id
    member_inserts = [
        (query, params)
        for query, params in db.executed
        if "INSERT INTO segment_audience_members" in query
    ]
    assert len(member_inserts) == 1
    assert "FROM audience_ann_members" in member_inserts[0][0]
    assert member_inserts[0][1][1] == "ann"
    snapshot_insert = next(
        (query, params)
        for query, params in db.executed
        if "INSERT INTO segment_audience_snapshots" in query
    )
    assert snapshot_insert[0].count("%s") == len(snapshot_insert[1]) == 38
    assert "query_compiler_hash" in snapshot_insert[0]
    assert "calibration_hash" in snapshot_insert[0]


def test_snapshot_explicit_members_use_postgres_array_parameters() -> None:
    db = _SnapshotWriteDb(actual_member_count=2)
    repository = AnalysisAudienceSnapshotRepository(db)
    now = datetime(2026, 7, 16, tzinfo=UTC)
    spec = HotelBookingBehaviorSchemaV2().compile_candidate(
        candidate_type="promotion_responsive",
        intent=_intent(),
        calibration=CandidateCalibration(0.5, "test.v1"),
    )

    snapshot_id = repository.save_completed(
        AudienceSnapshotWrite(
            analysis_id="analysis",
            project_id="project",
            campaign_id="campaign",
            promotion_id="promotion",
            segment_id="segment",
            segment_vector_id="segment_vector",
            vector_generation_id="uvgen_test",
            source_cutoff=now,
            window_start=now - timedelta(days=90),
            window_end=now,
            spec=spec,
            search_result=AudienceSearchResult(
                method=AudienceSearchMethod.EXACT,
                members=(
                    SearchCandidate("user_1", 0.9, 1),
                    SearchCandidate("user_2", 0.8, 2),
                ),
                corpus_user_count=2,
                hard_match_user_count=2,
                requested_k=2,
                recall_audit=None,
                policy_version="audience_search.v2",
            ),
            min_sample_size=1,
        )
    )

    _query, params = next(
        (query, params)
        for query, params in db.executed
        if "FROM unnest(" in query
    )
    assert params == (
        [snapshot_id, snapshot_id],
        ["user_1", "user_2"],
        [Decimal("0.9"), Decimal("0.8")],
        ["exact", "exact"],
        [1, 2],
    )
    assert all(isinstance(value, list) for value in params)


def test_snapshot_binding_uses_contract_score_threshold_precision() -> None:
    segment = _v2_segment("segment", candidate_type="benefit_value_seeker")
    compiled = compile_registered_segment_audience(
        segment_id=segment.segment_id,
        rule_json=segment.rule_json,
    )
    stored_score_threshold = Decimal(str(compiled.score_threshold)).quantize(
        Decimal("0.000001")
    )
    assert stored_score_threshold != Decimal(str(compiled.score_threshold))
    now = datetime(2026, 7, 16, tzinfo=UTC)
    repository = AnalysisAudienceSnapshotRepository(
        _SnapshotBindingDb(
            {
                "snapshot_id": "snapshot",
                "segment_vector_id": "segment-vector",
                "vector_generation_id": "generation",
                "source_cutoff": now,
                "window_start": now - timedelta(days=90),
                "window_end": now,
                "eligible_user_count": 100,
                "behavior_match_count": 20,
                "final_user_count": 10,
                "selection_method": "exact",
                "estimated_recall": Decimal("1"),
                "recall_lower_bound": Decimal("1"),
                "recall_target": Decimal("1"),
                "meets_min_sample_size": True,
                "snapshot_status": "completed",
                "project_id": "project",
                "campaign_id": "campaign",
                "promotion_id": "promotion",
                "segment_id": segment.segment_id,
                "schema_version": compiled.schema_version,
                "vector_version": compiled.vector_version,
                "manifest_hash": compiled.manifest_hash,
                "calibration_version": compiled.calibration_version,
                "calibration_hash": compiled.calibration_hash,
                "audience_resolution_contract": (
                    compiled.audience_resolution_contract
                ),
                "segment_audience_spec_hash": compiled.segment_audience_spec_hash,
                "query_vector_hash": semantic_query_vector_hash(compiled),
                "query_compiler_version": compiled.query_compiler_version,
                "query_compiler_hash": compiled.query_compiler_hash,
                "score_threshold": stored_score_threshold,
                "metadata_json": {
                    "spec_fingerprint": _spec_fingerprint(compiled),
                },
                "generation_status": "activated",
                "generation_is_active": True,
                "actual_member_count": 10,
            }
        )
    )

    bound = repository.require_binding(
        snapshot_id="snapshot",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        segment_id=segment.segment_id,
        spec=compiled,
    )

    assert bound.snapshot_id == "snapshot"


def test_source_snapshot_fingerprint_changes_with_promotion_exclusion_revision() -> None:
    now = datetime(2026, 7, 17, tzinfo=UTC)
    spec = HotelBookingBehaviorSchemaV2().compile_candidate(
        candidate_type="promotion_responsive",
        intent=_intent(),
        calibration=CandidateCalibration(0.5, "test.v1"),
    )
    base = AudienceSnapshotWrite(
        analysis_id="analysis",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        segment_id="segment",
        segment_vector_id="segment_vector",
        vector_generation_id="generation",
        source_cutoff=now,
        window_start=now - timedelta(days=30),
        window_end=now,
        spec=spec,
        search_result=AudienceSearchResult(
            method=AudienceSearchMethod.EXACT,
            members=(),
            corpus_user_count=100,
            hard_match_user_count=20,
            requested_k=0,
            recall_audit=None,
            policy_version="audience_search.v2",
        ),
        min_sample_size=10,
    )
    revision_one = PromotionAudienceExclusionContext(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        revision=1,
        excluded_user_count=2,
        projection_revision=1,
    )
    revision_two = replace(
        revision_one,
        revision=2,
        excluded_user_count=3,
        projection_revision=2,
    )

    first = _input_fingerprint(replace(base, exclusion_context=revision_one))
    second = _input_fingerprint(replace(base, exclusion_context=revision_two))

    assert first != second


def test_assignment_reuses_preallocated_run_target_snapshots_without_winner_search() -> None:
    run = PromotionRunRecord(
        promotion_run_id="run_1",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        analysis_id="analysis",
        generation_id="generation",
        loop_count=1,
        status="planned",
        goal_snapshot_json={"audience_scope": {"base": "analysis_snapshot"}},
        segment_scope_json=("seg_a", "seg_b"),
        segment_scope_fingerprint="a" * 64,
    )
    assignments = _AssignmentWriter()
    snapshots = _SnapshotReader()
    service = SegmentAssignmentService(
        promotion_run_repository=_RunReader(run),
        ad_experiment_repository=_ExperimentReader(
            [_experiment("seg_a"), _experiment("seg_b")]
        ),
        segment_vector_repository=_UnusedRepository(),
        user_behavior_vector_repository=_UnusedRepository(),
        user_segment_assignment_repository=assignments,
        reranker=SegmentCandidateReranker(),
        audience_snapshot_repository=snapshots,
    )
    response = service.build_assignments(
        promotion_run_id="run_1",
        request=SegmentAssignmentBuildRequest(),
    )
    assert response.matching_mode == "analysis_snapshot_reuse"
    assert response.assignment_mode == "analysis_snapshot"
    assert response.input_stability == "snapshotted"
    assert response.assignment_count == 3
    assert {row.user_id: row.segment_id for row in assignments.rows} == {
        "user_1": "seg_b",
        "user_2": "seg_a",
        "user_3": "seg_b",
    }
    assert {
        row.user_id: row.similarity_score for row in assignments.rows
    } == {
        "user_1": Decimal("0.900000"),
        "user_2": Decimal("0.000000"),
        "user_3": None,
    }
    assert all(
        row.assignment_source == AssignmentSource.ANALYSIS_SNAPSHOT.value
        for row in assignments.rows
    )
    assert snapshots.consume_calls == 1
def test_vector_search_sync_is_incremental_and_advances_complete_cutoff() -> None:
    repository = _SyncRepository()
    first = UserBehaviorVectorSearchSyncService(repository).sync(
        project_id="project",
        vector_version="hotel_behavior.v2",
        vector_generation_id="uvgen_test",
        batch_size=1,
        max_batches=1,
    )
    assert first.status == "in_progress"
    assert first.synced_user_count == 1

    result = UserBehaviorVectorSearchSyncService(repository).sync(
        project_id="project",
        vector_version="hotel_behavior.v2",
        vector_generation_id="uvgen_test",
        batch_size=1,
        max_batches=2,
    )
    assert result.status == "activated"
    assert result.synced_vector_count == 2
    assert result.source_cutoff == repository.generation.window_end
    assert [row.user_id for row in repository.upserted] == ["user_1", "user_2"]
    assert result.active_generation_id == "uvgen_test"


def test_failed_vector_generation_preserves_previous_active_generation() -> None:
    repository = _SyncRepository()
    repository.source_user_count = 1
    repository.active_generation_id = "uvgen_previous"

    result = UserBehaviorVectorSearchSyncService(repository).sync(
        project_id="project",
        vector_version="hotel_behavior.v2",
        vector_generation_id="uvgen_test",
        batch_size=10,
        max_batches=10,
    )

    assert result.status == "failed"
    assert result.active_generation_id == "uvgen_previous"


def test_vector_revision_window_compares_at_clickhouse_utc_millisecond_precision(
) -> None:
    repository = _SyncRepository()
    generation = replace(
        repository.generation,
        window_start=repository.generation.window_start.replace(microsecond=123_456),
        window_end=repository.generation.window_end.replace(microsecond=654_321),
    )
    revision = replace(
        repository.revisions[0],
        window_start=generation.window_start.replace(
            tzinfo=None,
            microsecond=123_000,
        ),
        window_end=generation.window_end.replace(
            tzinfo=None,
            microsecond=654_000,
        ),
    )

    _validate_revision(revision, generation)

    with pytest.raises(
        ValueError,
        match="search vector revision does not belong to generation",
    ):
        _validate_revision(
            replace(
                revision,
                window_end=revision.window_end + timedelta(milliseconds=1),
            ),
            generation,
        )


def test_vector_revision_bulk_upsert_uses_postgres_array_parameters() -> None:
    class RecordingPostgres:
        params: tuple[object, ...] | None = None

        def execute(self, _query: str, params: tuple[object, ...] = ()) -> None:
            self.params = params

    source = _SyncRepository()
    postgres = RecordingPostgres()
    repository = UserBehaviorVectorSearchSyncRepository(
        clickhouse=object(),
        postgres=postgres,
    )

    repository.bulk_upsert_revisions(
        generation=source.generation,
        revisions=source.revisions,
    )

    assert postgres.params is not None
    assert all(isinstance(value, list) for value in postgres.params[5:])


def test_assignment_snapshot_contract_accepts_superseded_consistent_generation() -> None:
    repository = DecisionAudienceSnapshotRepository(
        _SnapshotContractDb(
            [
                _snapshot_contract_row("seg_a", "snapshot_a"),
                _snapshot_contract_row(
                    "seg_b",
                    "snapshot_b",
                    candidate_type="promotion_responsive",
                ),
            ]
        )
    )

    result = repository.require_complete_set(
        analysis_id="analysis",
        segment_ids=("seg_b", "seg_a"),
    )

    assert result.segment_ids == ("seg_a", "seg_b")
    assert result.snapshot_ids == ("snapshot_a", "snapshot_b")


def test_assignment_snapshot_contract_rejects_semantic_or_generation_drift() -> None:
    rows = [
        _snapshot_contract_row("seg_a", "snapshot_a"),
        {
            **_snapshot_contract_row("seg_b", "snapshot_b"),
            "calibration_version": "different-calibration",
        },
    ]
    repository = DecisionAudienceSnapshotRepository(_SnapshotContractDb(rows))

    with pytest.raises(AudienceSnapshotContractError, match="semantic_mismatch"):
        repository.require_complete_set(
            analysis_id="analysis",
            segment_ids=("seg_a", "seg_b"),
        )


def test_assignment_snapshot_contract_rejects_generation_drift() -> None:
    rows = [
        _snapshot_contract_row("seg_a", "snapshot_a"),
        {
            **_snapshot_contract_row("seg_b", "snapshot_b"),
            "generation_matches": False,
        },
    ]
    repository = DecisionAudienceSnapshotRepository(_SnapshotContractDb(rows))

    with pytest.raises(AudienceSnapshotContractError, match="validation failed"):
        repository.require_complete_set(
            analysis_id="analysis",
            segment_ids=("seg_a", "seg_b"),
        )


def _intent(
    *,
    destinations: tuple[str, ...] = (),
    benefits: tuple[str, ...] = (),
) -> PromotionIntent:
    return PromotionIntent(
        summary="hotel promotion",
        product="hotel",
        season=(),
        destinations=destinations,
        benefits=benefits,
        audience_hints=(),
        channel="onsite_banner",
        goal_metric="booking_conversion_rate",
        funnel_goal="booking_complete",
        desired_behaviors=(),
        explicit_conditions=(),
    )


def _profile(user_id: str, **overrides: object) -> RawEventUserSignalRecord:
    values = {
        "project_id": "project",
        "user_id": user_id,
        "event_count": 10,
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
        "avg_price": 0.0,
        "destination_values": (),
        "checkin_dates": (),
        "hotel_market_values": (),
        "hotel_cluster_values": (),
        "age_group_values": (),
        "gender_values": (),
        "preferred_category_values": (),
        "destination_match_count": 0,
        "season_match_count": 0,
    }
    values.update(overrides)
    return RawEventUserSignalRecord(**values)


def _analysis_promotion() -> AnalysisPromotionRecord:
    return AnalysisPromotionRecord(
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        channel="onsite_banner",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.03"),
        goal_basis="all_segments",
        min_sample_size=5,
        landing_url=None,
        message_brief=None,
    )


def _v2_segment(
    segment_id: str,
    *,
    candidate_type: str = "intent_matched",
) -> SegmentDefinitionRecord:
    destinations = ("jeju",) if candidate_type == "target_destination_affinity" else ()
    audience_spec = RegisteredSegmentAudienceBinder().bind(
        candidate_type=candidate_type,
        destination_ids=destinations,
    )
    return SegmentDefinitionRecord(
        segment_id=segment_id,
        project_id="project",
        segment_name=segment_id,
        source="ai_suggested",
        query_preview_id=None,
        natural_language_query=None,
        generated_sql=None,
        rule_json={
            "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
            "segment_audience_spec": dict(audience_spec),
        },
        profile_json={},
        sample_size=10,
        total_eligible_user_count=100,
        sample_ratio=Decimal("0.1"),
        status="active",
        campaign_id="campaign",
        promotion_id="promotion",
    )


class _TestCalibrationProvider:
    def __init__(self) -> None:
        self.delegate = BundledCandidateCalibrationProvider()

    def require(self, **kwargs: object) -> CandidateCalibration:
        return self.delegate.require(**kwargs)


class _BatchCoordinatorSearchRepository:
    def __init__(
        self,
        *,
        hard_match_count: int = 10,
        materialized_member_count: int = 2,
    ) -> None:
        self.batch_call_count = 0
        self.individual_call_count = 0
        self.hard_match_count = hard_match_count
        self.materialized_member_count = materialized_member_count
        now = datetime(2026, 7, 16, tzinfo=UTC)
        self.context = AudienceSearchContext(
            vector_generation_id="uvgen_active",
            manifest_hash=HOTEL_BEHAVIOR_MANIFEST_HASH,
            source_cutoff=now,
            source_revision_cutoff=now,
            window_start=now - timedelta(days=30),
            corpus_user_count=100,
        )

    def get_context(self, **_kwargs: object) -> AudienceSearchContext:
        return self.context

    def count_hard_matches_batch(self, *, requests, **_kwargs: object):
        self.batch_call_count += 1
        return {
            request.segment_id: self.hard_match_count for request in requests
        }

    def count_hard_matches(self, **_kwargs: object) -> int:
        self.individual_call_count += 1
        return self.hard_match_count

    def estimate_score_pass_rate(self, **_kwargs: object) -> float:
        return 0.5

    def materialize_exact_members(self, **_kwargs: object) -> int:
        return self.materialized_member_count


class _SemanticCorpusSearchRepository:
    def __init__(
        self,
        *,
        schema: HotelBookingBehaviorSchemaV2,
        profiles,
        specs_by_segment,
        queries_by_segment,
    ) -> None:
        self.schema = schema
        self.profiles = profiles
        self.specs_by_segment = specs_by_segment
        self.query_by_vector = {
            tuple(query.query_vector): (specs_by_segment[segment_id], query)
            for segment_id, query in queries_by_segment.items()
        }
        now = datetime(2026, 7, 16, tzinfo=UTC)
        self.context = AudienceSearchContext(
            vector_generation_id="uvgen_demo",
            manifest_hash=HOTEL_BEHAVIOR_MANIFEST_HASH,
            source_cutoff=now,
            source_revision_cutoff=now,
            window_start=now - timedelta(days=30),
            corpus_user_count=len(profiles),
        )
        self.batch_call_count = 0
        self.individual_call_count = 0
        self.exact_call_count = 0

    def get_context(self, **_kwargs: object) -> AudienceSearchContext:
        return self.context

    def count_hard_matches_batch(self, *, requests, **_kwargs: object):
        self.batch_call_count += 1
        return {
            request.segment_id: sum(
                _matches_hard_predicates(
                    profile,
                    spec=self.specs_by_segment[request.segment_id],
                )
                for profile in self.profiles
            )
            for request in requests
        }

    def count_hard_matches(self, **_kwargs: object) -> int:
        self.individual_call_count += 1
        raise AssertionError("all registered demo predicates must batch")

    def estimate_score_pass_rate(self, **kwargs: object) -> float:
        spec, query = self.query_by_vector[tuple(kwargs["query_vector"])]
        hard_matches = [
            profile
            for profile in self.profiles
            if _matches_hard_predicates(profile, spec=spec)
        ]
        if not hard_matches:
            return 0.0
        return sum(
            cosine_similarity(
                query.query_vector,
                self.schema.vectorize_user(profile),
            )
            >= query.score_threshold
            for profile in hard_matches
        ) / len(hard_matches)

    def exact_search(self, **kwargs: object) -> list[SearchCandidate]:
        self.exact_call_count += 1
        spec, query = self.query_by_vector[tuple(kwargs["query_vector"])]
        members = []
        for profile in self.profiles:
            score = cosine_similarity(
                query.query_vector,
                self.schema.vectorize_user(profile),
            )
            if (
                _matches_hard_predicates(profile, spec=spec)
                and score >= query.score_threshold
            ):
                members.append((profile.user_id, score))
        members.sort(key=lambda value: (-value[1], value[0]))
        return [
            SearchCandidate(
                user_id=user_id,
                behavior_fit_score=score,
                retrieval_rank=index,
            )
            for index, (user_id, score) in enumerate(members, start=1)
        ]


class _BatchSegmentVectorPreparer:
    def prepare_segment_vector(self, request) -> SegmentVectorBuildResult:
        return SegmentVectorBuildResult(
            segment_id=request.segment_id,
            segment_vector_id=f"vector_{request.segment_id}",
            vector_values=list(request.query_vector),
            source="behavior_query",
        )


class _BatchSnapshotWriter:
    def __init__(self) -> None:
        self.writes = []

    def save_completed(self, write) -> str:
        self.writes.append(write)
        return f"snapshot_{write.segment_id}"


class _SearchRepository:
    def __init__(self, *, always_miss: bool) -> None:
        self.always_miss = always_miss
        self.ann_limits: list[int] = []

    def exact_search(self, **_kwargs: object) -> list[SearchCandidate]:
        return [SearchCandidate("exact_user", 0.9, 1)]

    def ann_search(self, **kwargs: object) -> list[SearchCandidate]:
        self.ann_limits.append(int(kwargs["limit"]))
        return [SearchCandidate(f"ann_{index}", 0.8, index) for index in range(5)]

    def exact_filter_candidates(self, **_kwargs: object) -> list[SearchCandidate]:
        return [SearchCandidate("ann_0", 0.8, 1)]

    def audit_nonretrieved(self, **_kwargs: object) -> tuple[int, int]:
        return (10, 10 if self.always_miss else 0)


class _MaterializedSearchRepository:
    def __init__(self) -> None:
        self.legacy_call_count = 0

    def materialize_exact_members(self, **_kwargs: object) -> int:
        return 800

    def materialize_ann_candidates(self, **_kwargs: object) -> int:
        return 900

    def materialize_ann_members(self, **_kwargs: object) -> int:
        return 800

    def audit_materialized_nonretrieved(
        self,
        **_kwargs: object,
    ) -> tuple[int, int]:
        return 10_000, 0

    def compare_materialized_members(self, **_kwargs: object) -> tuple[int, int]:
        return 800, 0

    def exact_search(self, **_kwargs: object):
        self.legacy_call_count += 1
        raise AssertionError("materialized production path must not fetch all members")

    def ann_search(self, **_kwargs: object):
        self.legacy_call_count += 1
        raise AssertionError("materialized production path must not fetch all ANN rows")

    def exact_filter_candidates(self, **_kwargs: object):
        self.legacy_call_count += 1
        raise AssertionError("materialized production path must not pass large id arrays")

    def audit_nonretrieved(self, **_kwargs: object):
        self.legacy_call_count += 1
        raise AssertionError("materialized production path must use relation anti-join")


class _SearchSqlPostgres:
    def __init__(
        self,
        *,
        fetchall_results: list[list[dict[str, object]]] | None = None,
    ) -> None:
        self.executed: list[tuple[str, object]] = []
        self.fetchall_calls: list[tuple[str, tuple[object, ...]]] = []
        self._fetchall_results = list(fetchall_results or [])

    def execute(self, query: str, params: object = ()) -> None:
        self.executed.append((query, params))

    def fetchall(self, query: str, params: tuple[object, ...]):
        self.fetchall_calls.append((query, params))
        return self._fetchall_results.pop(0) if self._fetchall_results else []

    def fetchone(self, query: str, _params: object = ()):
        if "SELECT count(*) AS row_count" in query:
            return {"row_count": 100_000}
        return None


class _SnapshotWriteDb:
    def __init__(self, *, actual_member_count: int = 800) -> None:
        self.actual_member_count = actual_member_count
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def fetchone(self, query: str, _params: object = ()):
        if "SELECT count(*) AS actual_member_count" in query:
            return {"actual_member_count": self.actual_member_count}
        return None

    def execute(self, query: str, params: object = ()) -> None:
        self.executed.append((query, tuple(params)))


class _SnapshotBindingDb:
    def __init__(self, row: dict[str, object]) -> None:
        self.row = row

    def fetchone(self, _query: str, _params: object = ()) -> dict[str, object]:
        return self.row

    def execute(self, _query: str, _params: object = ()) -> None:
        raise AssertionError("snapshot binding validation must be read-only")


class _RunReader:
    def __init__(self, run: PromotionRunRecord) -> None:
        self.run = run

    def get_by_id(self, _promotion_run_id: str) -> PromotionRunRecord:
        return self.run


class _ExperimentReader:
    def __init__(self, rows: list[AdExperimentRecord]) -> None:
        self.rows = rows

    def list_by_run(self, _promotion_run_id: str) -> list[AdExperimentRecord]:
        return self.rows


class _SnapshotReader:
    def __init__(self) -> None:
        self.consume_calls = 0

    def resolve_run_contract(self, **_kwargs: object) -> TargetAudienceResolution:
        return TargetAudienceResolution(
            "analysis",
            ("seg_a", "seg_b"),
            "segment_audience.v1",
        )

    def require_run_binding_set(self, **_kwargs: object) -> AudienceSnapshotSet:
        return AudienceSnapshotSet("analysis", ("seg_a", "seg_b"), "hotel_behavior.v2", 3)

    def consume_run_members(self, **_kwargs: object) -> None:
        self.consume_calls += 1

    def list_run_members(self, **kwargs: object) -> list[AudienceSnapshotMember]:
        if kwargs["after_user_id"] is not None:
            return []
        # Final allocation snapshots are already mutually exclusive.
        return [
            AudienceSnapshotMember("user_1", "seg_b", Decimal("0.9")),
            AudienceSnapshotMember("user_2", "seg_a", Decimal("-0.25")),
            AudienceSnapshotMember("user_3", "seg_b", None),
        ]


class _AssignmentWriter:
    def __init__(self) -> None:
        self.rows = []

    def list_existing_user_ids(self, **_kwargs: object) -> set[str]:
        return set()

    def insert_many(self, rows):
        self.rows.extend(rows)
        return [
            UserSegmentAssignmentInsertRecord(
                row.user_id,
                row.segment_id,
                row.fallback,
                row.fallback_reason,
                row.similarity_score,
            )
            for row in rows
        ]


class _UnusedRepository:
    def __getattr__(self, name: str):
        raise AssertionError(f"snapshot assignment must not call {name}")


class _SyncRepository:
    def __init__(self) -> None:
        now = datetime(2026, 7, 16, tzinfo=UTC)
        self.revisions = [
            SearchVectorRevision(
                project_id="project",
                user_id=f"user_{index}",
                vector_dim=64,
                vector_values=tuple([1.0, *([0.0] * 63)]),
                vector_version="hotel_behavior.v2",
                source="raw_events",
                window_start=now - timedelta(days=90),
                window_end=now,
                updated_at=now,
                vector_row_id=f"row_{index}",
                ingested_at=now + timedelta(seconds=index),
            )
            for index in (1, 2)
        ]
        self.upserted = []
        self.status = "in_progress"
        self.last_user_id = None
        self.active_generation_id = None
        self.source_user_count = 2
        self.generation = VectorSearchGeneration(
            vector_generation_id="uvgen_test",
            project_id="project",
            vector_version="hotel_behavior.v2",
            manifest_hash="a" * 64,
            window_start=now - timedelta(days=90),
            window_end=now,
            source_revision_cutoff=now + timedelta(minutes=1),
            expected_user_count=2,
            synced_user_count=0,
            invalid_user_count=0,
            cursor=VectorSyncCursor(),
            status="in_progress",
        )

    def get_generation(self, **_kwargs: object) -> VectorSearchGeneration:
        return VectorSearchGeneration(
            vector_generation_id=self.generation.vector_generation_id,
            project_id=self.generation.project_id,
            vector_version=self.generation.vector_version,
            manifest_hash=self.generation.manifest_hash,
            window_start=self.generation.window_start,
            window_end=self.generation.window_end,
            source_revision_cutoff=self.generation.source_revision_cutoff,
            expected_user_count=2,
            synced_user_count=len(self.upserted),
            invalid_user_count=0,
            cursor=VectorSyncCursor(self.last_user_id),
            status=self.status,
        )

    def list_revisions(self, *, generation: VectorSearchGeneration, **_kwargs: object):
        if generation.cursor.user_id is None:
            return [self.revisions[0]]
        if generation.cursor.user_id == "user_1":
            return [self.revisions[1]]
        return []

    def bulk_upsert_revisions(self, *, revisions, **_kwargs: object) -> None:
        self.upserted.extend(revisions)

    def save_progress(self, *, last_user_id: str, **_kwargs: object) -> None:
        self.last_user_id = last_user_id

    def count_source_users(self, **_kwargs: object) -> int:
        return self.source_user_count

    def count_synced_users(self, **_kwargs: object) -> int:
        return len(self.upserted)

    def activate_generation(self, **_kwargs: object) -> None:
        self.status = "activated"
        self.active_generation_id = "uvgen_test"

    def get_active_generation_id(self, **_kwargs: object) -> str | None:
        return self.active_generation_id

    def mark_failed(self, **_kwargs: object) -> None:
        self.status = "failed"


class _SnapshotContractDb:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def fetchall(self, _query: str, _params: object) -> list[dict[str, object]]:
        return self.rows

    def execute(self, _query: str, _params: object = ()) -> None:
        raise AssertionError("contract validation must be read-only")


def _snapshot_contract_row(
    segment_id: str,
    snapshot_id: str,
    *,
    candidate_type: str = "intent_matched",
) -> dict[str, object]:
    segment = _v2_segment(segment_id, candidate_type=candidate_type)
    compiled = compile_registered_segment_audience(
        segment_id=segment_id,
        rule_json=segment.rule_json,
    )
    return {
        "segment_id": segment_id,
        "rule_json": segment.rule_json,
        "audience_snapshot_id": snapshot_id,
        "vector_version": compiled.vector_version,
        "schema_version": compiled.schema_version,
        "manifest_hash": compiled.manifest_hash,
        "calibration_version": compiled.calibration_version,
        "calibration_hash": compiled.calibration_hash,
        "audience_resolution_contract": compiled.audience_resolution_contract,
        "segment_audience_spec_hash": compiled.segment_audience_spec_hash,
        "query_vector_hash": semantic_query_vector_hash(compiled),
        "query_compiler_version": compiled.query_compiler_version,
        "query_compiler_hash": compiled.query_compiler_hash,
        "score_threshold": Decimal(str(compiled.score_threshold)),
        "matcher_version": "exact_cosine_rerank.v2",
        "search_policy_version": "audience_search.v2",
        "metadata_json": {
            "calibration_hash": compiled.calibration_hash,
            "audience_resolution_contract": compiled.audience_resolution_contract,
            "query_compiler_version": compiled.query_compiler_version,
            "query_compiler_hash": compiled.query_compiler_hash,
            "template_id": compiled.template_id,
            "template_version": compiled.template_version,
            "template_semantic_hash": compiled.template_semantic_hash,
            "hard_predicate_keys": list(compiled.hard_predicate_keys),
            "predicate_parameters": {
                key: list(value)
                for key, value in compiled.predicate_parameters.items()
            },
            "semantic_selection_policy_id": (
                compiled.semantic_selection_policy_id
            ),
            "semantic_anchor_policy_id": compiled.semantic_anchor_policy_id,
            "semantic_anchor_hash": compiled.semantic_anchor_hash,
            "semantic_margin": compiled.semantic_margin,
            "semantic_selection_status": compiled.semantic_selection_status,
            "business_lift_status": compiled.business_lift_status,
            "user_vectorizer_version": compiled.user_vectorizer_version,
            "user_vectorizer_semantic_hash": (
                compiled.user_vectorizer_semantic_hash
            ),
        },
        "snapshot_status": "completed",
        "snapshot_kind": "final",
        "source_snapshot_id": f"source_{segment_id}",
        "snapshot_allocation_plan_id": "plan",
        "target_allocation_plan_id": "plan",
        "audience_reservation_state": "reserved",
        "audience_status": "targetable",
        "final_user_count": 2,
        "identity_matches": True,
        "generation_status": "superseded",
        "generation_is_active": False,
        "generation_matches": True,
        "actual_member_count": 2,
    }


def _experiment(segment_id: str) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=f"exp_{segment_id}",
        project_id="project",
        campaign_id="campaign",
        promotion_id="promotion",
        promotion_run_id="run_1",
        analysis_id="analysis",
        generation_id="generation",
        segment_id=segment_id,
        segment_name=segment_id,
        content_id=f"content_{segment_id}",
        content_option_id=f"option_{segment_id}",
        channel="onsite_banner",
        loop_count=1,
        status="planned",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.03"),
        goal_basis="all_segments",
    )
