from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence

from app.decision.repositories import (
    AdExperimentRepository,
    AdExperimentWrite,
    ContentCandidateRepository,
    EvaluationMetricRepository,
    GenerationRunRepository,
    PromotionAnalysisRepository,
    PromotionEvaluationRepository,
    PromotionEvaluationRecord,
    PromotionEvaluationWrite,
    PromotionRepository,
    PromotionRunRepository,
    PromotionRunWrite,
    PromotionTargetSegmentRepository,
    SegmentVectorRepository,
    UserBehaviorVectorRepository,
    UserSegmentAssignmentRepository,
    UserSegmentAssignmentWrite,
)
from app.decision.schemas import (
    AdExperimentStatus,
    AssignmentSource,
    Channel,
    GoalBasis,
    GoalMetric,
    PromotionEvaluationStatus,
    PromotionRunStatus,
)


@dataclass(frozen=True)
class DbCall:
    operation: str
    query: str
    params: Sequence[Any] | Mapping[str, Any]


class FakePostgresExecutor:
    def __init__(
        self,
        *,
        fetchone_result: Mapping[str, Any] | None = None,
        fetchall_result: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.calls: list[DbCall] = []

    def fetchone(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Mapping[str, Any] | None:
        self.calls.append(DbCall("fetchone", query, params))
        return self.fetchone_result

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> list[Mapping[str, Any]]:
        self.calls.append(DbCall("fetchall", query, params))
        return self.fetchall_result

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> None:
        self.calls.append(DbCall("execute", query, params))


class FakeClickHouseResult:
    def __init__(self, rows: list[Any]) -> None:
        self.result_rows = rows


class FakeClickHouseClient:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.calls: list[DbCall] = []

    def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> FakeClickHouseResult:
        self.calls.append(DbCall("query", query, parameters or {}))
        return FakeClickHouseResult(self.rows)


def compact_sql(query: str) -> str:
    return " ".join(query.split()).lower()


def test_promotion_repository_get_by_id_includes_b2_goal_and_loop_fields() -> None:
    db = FakePostgresExecutor(
        fetchone_result={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "channel": Channel.ONSITE_BANNER.value,
            "goal_metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
            "goal_target_value": Decimal("0.030000"),
            "goal_basis": GoalBasis.ALL_SEGMENTS.value,
            "min_sample_size": 1000,
            "max_loop_count": 3,
        }
    )
    repo = PromotionRepository(db)

    promotion = repo.get_by_id("promo_banner_001")

    assert promotion is not None
    assert promotion.project_id == "hotel-client-a"
    assert promotion.campaign_id == "camp_summer_2026"
    assert promotion.goal_metric == GoalMetric.BOOKING_CONVERSION_RATE.value
    assert promotion.max_loop_count == 3
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from promotions" in sql
    assert "where promotion_id = %s" in sql
    assert "goal_metric" in sql
    assert "goal_target_value" in sql
    assert "goal_basis" in sql
    assert "min_sample_size" in sql
    assert "max_loop_count" in sql
    assert call.params == ("promo_banner_001",)


def test_analysis_repository_gets_latest_completed_for_promotion() -> None:
    db = FakePostgresExecutor(
        fetchone_result={
            "analysis_id": "analysis_banner_001",
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "focus_segment_ids_json": ["seg_family_trip"],
            "operator_instruction": None,
            "input_snapshot_json": {"promotion_id": "promo_banner_001"},
            "profile_summary_json": {"selected_segment_count": 1},
            "output_json": {"target_segment_count": 1},
            "status": "completed",
        }
    )
    repo = PromotionAnalysisRepository(db)

    analysis = repo.get_latest_completed_for_promotion("promo_banner_001")

    assert analysis is not None
    assert analysis.analysis_id == "analysis_banner_001"
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from promotion_analyses" in sql
    assert "where promotion_id = %s" in sql
    assert "and status = 'completed'" in sql
    assert "order by updated_at desc, created_at desc, analysis_id desc" in sql
    assert "limit 1" in sql
    assert call.params == ("promo_banner_001",)


def test_generation_repository_gets_latest_completed_for_promotion() -> None:
    db = FakePostgresExecutor(
        fetchone_result={
            "generation_id": "generation_banner_001",
            "analysis_id": "analysis_banner_001",
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "content_option_count": 3,
            "operator_instruction": "Keep hotel wording.",
            "input_json": {"analysis_id": "analysis_banner_001"},
            "output_json": {"content_count": 4},
            "generation_report_json": {"status": "ok"},
            "status": "completed",
        }
    )
    repo = GenerationRunRepository(db)

    generation = repo.get_latest_completed_for_promotion("promo_banner_001")

    assert generation is not None
    assert generation.generation_id == "generation_banner_001"
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from generation_runs" in sql
    assert "where promotion_id = %s" in sql
    assert "and status = 'completed'" in sql
    assert "order by updated_at desc, created_at desc, generation_id desc" in sql
    assert "input_json" in sql
    assert "generation_report_json" in sql
    assert call.params == ("promo_banner_001",)


def test_target_segment_repository_lists_segments_for_analysis() -> None:
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "analysis_id": "analysis_banner_001",
                "project_id": "hotel-client-a",
                "campaign_id": "camp_summer_2026",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_family_trip",
                "segment_name": "Family hotel trip",
                "segment_vector_id": "segvec_family_trip_v1",
                "rule_json": {"srch_children_cnt": {"gt": 0}},
                "profile_json": {"primary_segment": "seg_family_trip"},
                "content_brief_json": {"keywords": ["family room"]},
                "data_evidence_json": {"event_count": 120},
                "estimated_size": 1200,
                "priority": "high",
                "status": "planned",
            }
        ]
    )
    repo = PromotionTargetSegmentRepository(db)

    segments = repo.list_for_analysis("analysis_banner_001")

    assert [segment.segment_id for segment in segments] == ["seg_family_trip"]
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from promotion_target_segments" in sql
    assert "where analysis_id = %s" in sql
    assert "order by id asc" in sql
    assert "segment_vector_id" in sql
    assert call.params == ("analysis_banner_001",)


def test_content_candidate_repository_lists_approved_or_active_with_segment_keys() -> None:
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "content_id": "content_family_trip_001",
                "content_option_id": "option_a",
                "generation_id": "generation_banner_001",
                "analysis_id": "analysis_banner_001",
                "project_id": "hotel-client-a",
                "campaign_id": "camp_summer_2026",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_family_trip",
                "channel": Channel.ONSITE_BANNER.value,
                "status": "approved",
            }
        ]
    )
    repo = ContentCandidateRepository(db)

    candidates = repo.list_approved_or_active_for_generation("generation_banner_001")

    assert candidates[0].segment_id == "seg_family_trip"
    assert candidates[0].content_id == "content_family_trip_001"
    assert candidates[0].content_option_id == "option_a"
    assert candidates[0].channel == Channel.ONSITE_BANNER.value
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from content_candidates" in sql
    assert "where generation_id = %s" in sql
    assert "and status in ('approved', 'active')" in sql
    assert "segment_id" in sql
    assert "content_id" in sql
    assert "content_option_id" in sql
    assert "channel" in sql
    assert call.params == ("generation_banner_001",)


def test_promotion_run_repository_inserts_all_required_fields() -> None:
    db = FakePostgresExecutor()
    repo = PromotionRunRepository(db)
    run = PromotionRunWrite(
        promotion_run_id="prun_banner_001_loop_1",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        loop_count=1,
        status=PromotionRunStatus.PLANNED.value,
        goal_snapshot_json={"metric": "booking_conversion_rate"},
    )

    repo.insert(run)

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert call.operation == "execute"
    assert "insert into promotion_runs" in sql
    assert "promotion_run_id" in sql
    assert "project_id" in sql
    assert "campaign_id" in sql
    assert "analysis_id" in sql
    assert "generation_id" in sql
    assert "goal_snapshot_json" in sql
    assert call.params == (
        "prun_banner_001_loop_1",
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
        "analysis_banner_001",
        "generation_banner_001",
        1,
        PromotionRunStatus.PLANNED.value,
        {"metric": "booking_conversion_rate"},
    )


def test_promotion_run_uniqueness_check_uses_promotion_id_and_loop_count() -> None:
    db = FakePostgresExecutor(fetchone_result={"exists": 1})
    repo = PromotionRunRepository(db)

    exists = repo.exists_for_promotion_loop(
        promotion_id="promo_banner_001",
        loop_count=1,
    )

    assert exists is True
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from promotion_runs" in sql
    assert "where promotion_id = %s" in sql
    assert "and loop_count = %s" in sql
    assert "promotion_run_id" not in sql
    assert "segment_id" not in sql
    assert call.params == ("promo_banner_001", 1)


def test_promotion_run_repository_updates_status() -> None:
    db = FakePostgresExecutor()
    repo = PromotionRunRepository(db)

    repo.update_status(
        promotion_run_id="prun_banner_001_loop_1",
        status=PromotionRunStatus.PARTIAL_GOAL_MET.value,
    )

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert call.operation == "execute"
    assert "update promotion_runs" in sql
    assert "set status = %s" in sql
    assert "updated_at = now()" in sql
    assert "where promotion_run_id = %s" in sql
    assert call.params == (
        PromotionRunStatus.PARTIAL_GOAL_MET.value,
        "prun_banner_001_loop_1",
    )


def test_ad_experiment_repository_inserts_segment_content_and_goal_fields() -> None:
    db = FakePostgresExecutor()
    repo = AdExperimentRepository(db)
    experiment = AdExperimentWrite(
        ad_experiment_id="adexp_family_trip_001",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        segment_id="seg_family_trip",
        segment_name="Family hotel trip",
        content_id="content_family_trip_001",
        content_option_id="option_a",
        channel=Channel.ONSITE_BANNER.value,
        loop_count=1,
        status=AdExperimentStatus.PLANNED.value,
        goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        goal_target_value=Decimal("0.030000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
    )

    repo.insert_many([experiment])

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert call.operation == "execute"
    assert "insert into ad_experiments" in sql
    assert "segment_id" in sql
    assert "segment_name" in sql
    assert "content_id" in sql
    assert "content_option_id" in sql
    assert "goal_metric" in sql
    assert "goal_target_value" in sql
    assert "goal_basis" in sql
    assert call.params == (
        "adexp_family_trip_001",
        "hotel-client-a",
        "camp_summer_2026",
        "promo_banner_001",
        "prun_banner_001_loop_1",
        "analysis_banner_001",
        "generation_banner_001",
        "seg_family_trip",
        "Family hotel trip",
        "content_family_trip_001",
        "option_a",
        Channel.ONSITE_BANNER.value,
        1,
        AdExperimentStatus.PLANNED.value,
        GoalMetric.BOOKING_CONVERSION_RATE.value,
        Decimal("0.030000"),
        GoalBasis.ALL_SEGMENTS.value,
    )


def test_ad_experiment_uniqueness_check_uses_run_and_segment() -> None:
    db = FakePostgresExecutor(fetchone_result=None)
    repo = AdExperimentRepository(db)

    exists = repo.exists_for_run_segment(
        promotion_run_id="prun_banner_001_loop_1",
        segment_id="seg_family_trip",
    )

    assert exists is False
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from ad_experiments" in sql
    assert "where promotion_run_id = %s" in sql
    assert "and segment_id = %s" in sql
    assert "promotion_id" not in sql
    assert "loop_count" not in sql
    assert call.params == ("prun_banner_001_loop_1", "seg_family_trip")


def test_ad_experiment_repository_get_by_id_reads_evaluation_context() -> None:
    db = FakePostgresExecutor(
        fetchone_result={
            "ad_experiment_id": "adexp_family_trip_001",
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "promotion_run_id": "prun_banner_001_loop_1",
            "analysis_id": "analysis_banner_001",
            "generation_id": "generation_banner_001",
            "segment_id": "seg_family_trip",
            "segment_name": "Family hotel trip",
            "content_id": "content_family_trip_001",
            "content_option_id": "option_a",
            "channel": Channel.ONSITE_BANNER.value,
            "loop_count": 1,
            "status": AdExperimentStatus.RUNNING.value,
            "goal_metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
            "goal_target_value": Decimal("0.030000"),
            "goal_basis": GoalBasis.ALL_SEGMENTS.value,
        }
    )
    repo = AdExperimentRepository(db)

    experiment = repo.get_by_id("adexp_family_trip_001")

    assert experiment is not None
    assert experiment.project_id == "hotel-client-a"
    assert experiment.content_option_id == "option_a"
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from ad_experiments" in sql
    assert "where ad_experiment_id = %s" in sql
    assert "project_id" in sql
    assert "campaign_id" in sql
    assert "promotion_id" in sql
    assert "segment_id" in sql
    assert "content_id" in sql
    assert "content_option_id" in sql
    assert call.params == ("adexp_family_trip_001",)


def test_user_segment_assignment_write_carries_assignment_source() -> None:
    assigned_at = datetime(2026, 7, 3, tzinfo=UTC)

    assignment = UserSegmentAssignmentWrite(
        project_id="hotel-client-a",
        promotion_run_id="prun_banner_001_loop_1",
        user_id="user_001",
        segment_id="seg_existing_all",
        ad_experiment_id="adexp_existing_all_001",
        content_id="content_existing_all_001",
        content_option_id="option_a",
        similarity_score=Decimal("0.410000"),
        fallback=True,
        assignment_source=AssignmentSource.FALLBACK.value,
        assigned_at=assigned_at,
        expires_at=None,
    )

    assert assignment.fallback is True
    assert assignment.assignment_source == AssignmentSource.FALLBACK.value
    assert assignment.assigned_at == assigned_at


def test_segment_vector_repository_filters_run_context_and_version() -> None:
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "segment_vector_id": "segvec_family_v1",
                "project_id": "hotel-client-a",
                "promotion_id": "promo_banner_001",
                "promotion_run_id": None,
                "analysis_id": "analysis_banner_001",
                "segment_id": "seg_family_trip",
                "vector_dim": 64,
                "vector_values": [1.0] + [0.0] * 63,
                "vector_version": "v1",
                "source": "decision_analysis",
            }
        ]
    )
    repo = SegmentVectorRepository(db)

    vectors = repo.list_for_run_segments(
        project_id="hotel-client-a",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        segment_ids=["seg_family_trip"],
        vector_version="v1",
    )

    assert vectors[0].segment_id == "seg_family_trip"
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from segment_vectors" in sql
    assert "project_id = %s" in sql
    assert "promotion_id = %s" in sql
    assert "analysis_id = %s" in sql
    assert "segment_id = any(%s)" in sql
    assert "vector_version = %s" in sql
    assert "vector_dim = %s" in sql
    assert call.params == (
        "hotel-client-a",
        "promo_banner_001",
        "analysis_banner_001",
        ["seg_family_trip"],
        "v1",
        64,
    )


def test_user_segment_assignment_repository_inserts_official_columns_only() -> None:
    assigned_at = datetime(2026, 7, 3, tzinfo=UTC)
    db = FakePostgresExecutor(fetchone_result={"id": 1})
    repo = UserSegmentAssignmentRepository(db)

    inserted_count = repo.insert_many(
        [
            UserSegmentAssignmentWrite(
                project_id="hotel-client-a",
                promotion_run_id="prun_banner_001_loop_1",
                user_id="user_001",
                segment_id="seg_existing_all",
                ad_experiment_id="adexp_existing_all_001",
                content_id="content_existing_all_001",
                content_option_id="option_a",
                similarity_score=Decimal("0.410000"),
                fallback=True,
                assignment_source=AssignmentSource.FALLBACK.value,
                assigned_at=assigned_at,
                expires_at=None,
            )
        ]
    )

    assert inserted_count == 1
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "insert into user_segment_assignments" in sql
    assert "promotion_run_id" in sql
    assert "user_id" in sql
    assert "segment_id" in sql
    assert "ad_experiment_id" in sql
    assert "content_id" in sql
    assert "content_option_id" in sql
    assert "similarity_score" in sql
    assert "fallback" in sql
    assert "assignment_source" in sql
    assert "assignment_status" not in sql
    assert "on conflict (promotion_run_id, user_id) do nothing" in sql
    assert "returning id" in sql


def test_user_segment_assignment_repository_counts_final_assignments() -> None:
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "segment_id": "seg_family_trip",
                "assigned_user_count": 42,
            }
        ]
    )
    repo = UserSegmentAssignmentRepository(db)

    counts = repo.count_by_run_segments(
        promotion_run_id="prun_banner_001_loop_1",
        segment_ids=["seg_family_trip", "seg_mobile_user"],
    )

    assert counts == {"seg_family_trip": 42}
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from user_segment_assignments" in sql
    assert "where promotion_run_id = %s" in sql
    assert "segment_id = any(%s)" in sql
    assert "group by segment_id" in sql
    assert call.params == (
        "prun_banner_001_loop_1",
        ["seg_family_trip", "seg_mobile_user"],
    )


def test_user_behavior_vector_repository_reads_latest_vectors_with_argmax() -> None:
    client = FakeClickHouseClient(
        rows=[
            (
                "hotel-client-a",
                "user_001",
                64,
                [1.0] + [0.0] * 63,
                "v1",
                "batch_profile",
            )
        ]
    )
    repo = UserBehaviorVectorRepository(client)

    vectors = repo.list_by_user_ids(
        project_id="hotel-client-a",
        user_ids=["user_001"],
        vector_version="v1",
    )

    assert vectors[0].user_id == "user_001"
    assert vectors[0].vector_values == [1.0] + [0.0] * 63
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "argmax(vector_values, updated_at)" in sql
    assert "argmax(vector_dim, updated_at)" in sql
    assert "from user_behavior_vectors" in sql
    assert "group by project_id, user_id, vector_version" in sql
    assert "user_id in {user_ids:array(string)}" in sql
    assert call.params == {
        "project_id": "hotel-client-a",
        "vector_version": "v1",
        "vector_dim": 64,
        "user_ids": ["user_001"],
    }


def test_user_behavior_vector_repository_limits_project_scope() -> None:
    client = FakeClickHouseClient(rows=[])
    repo = UserBehaviorVectorRepository(client)

    vectors = repo.list_for_project(
        project_id="hotel-client-a",
        vector_version="v1",
        limit=500,
    )

    assert vectors == []
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "argmax(vector_values, updated_at)" in sql
    assert "limit {limit:uint32}" in sql
    assert "user_id in" not in sql
    assert call.params == {
        "project_id": "hotel-client-a",
        "vector_version": "v1",
        "vector_dim": 64,
        "limit": 500,
    }


def test_promotion_evaluation_repository_inserts_required_contract_columns() -> None:
    db = FakePostgresExecutor()
    repo = PromotionEvaluationRepository(db)

    repo.insert(
        PromotionEvaluationWrite(
            evaluation_id="eval_adexp_family_trip_001",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            promotion_run_id="prun_banner_001_loop_1",
            ad_experiment_id="adexp_family_trip_001",
            segment_id="seg_family_trip",
            content_id="content_family_trip_001",
            content_option_id="option_a",
            metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
            target_value=Decimal("0.030000"),
            actual_value=Decimal("0.025000"),
            numerator_count=25,
            denominator_count=1000,
            sample_size=1000,
            basis=GoalBasis.ALL_SEGMENTS.value,
            status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            feedback=None,
            next_loop_required=True,
            result_json={"status_reason": "target_not_met"},
        )
    )

    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "insert into promotion_evaluations" in sql
    assert "project_id" in sql
    assert "campaign_id" in sql
    assert "promotion_id" in sql
    assert "segment_id" in sql
    assert "content_id" in sql
    assert "content_option_id" in sql
    assert "basis" in sql
    assert "goal_near" not in sql
    assert call.params[15] == GoalBasis.ALL_SEGMENTS.value
    assert call.params[18] is True


def test_promotion_evaluation_repository_lists_latest_individual_evaluations() -> None:
    db = FakePostgresExecutor(
        fetchall_result=[
            {
                "evaluation_id": "eval_adexp_family_trip_latest",
                "project_id": "hotel-client-a",
                "campaign_id": "camp_summer_2026",
                "promotion_id": "promo_banner_001",
                "promotion_run_id": "prun_banner_001_loop_1",
                "ad_experiment_id": "adexp_family_trip_001",
                "segment_id": "seg_family_trip",
                "content_id": "content_family_trip_001",
                "content_option_id": "option_a",
                "metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
                "target_value": Decimal("0.300000"),
                "actual_value": Decimal("0.400000"),
                "numerator_count": 4,
                "denominator_count": 10,
                "sample_size": 10,
                "basis": GoalBasis.ALL_SEGMENTS.value,
                "status": PromotionEvaluationStatus.GOAL_MET.value,
                "feedback": None,
                "next_loop_required": False,
                "result_json": {"status_reason": "target_met"},
            }
        ]
    )
    repo = PromotionEvaluationRepository(db)

    evaluations = repo.list_latest_by_run_ad_experiments(
        "prun_banner_001_loop_1"
    )

    assert evaluations == [
        PromotionEvaluationRecord(
            evaluation_id="eval_adexp_family_trip_latest",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            promotion_run_id="prun_banner_001_loop_1",
            ad_experiment_id="adexp_family_trip_001",
            segment_id="seg_family_trip",
            content_id="content_family_trip_001",
            content_option_id="option_a",
            metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
            target_value=Decimal("0.300000"),
            actual_value=Decimal("0.400000"),
            numerator_count=4,
            denominator_count=10,
            sample_size=10,
            basis=GoalBasis.ALL_SEGMENTS.value,
            status=PromotionEvaluationStatus.GOAL_MET.value,
            feedback=None,
            next_loop_required=False,
            result_json={"status_reason": "target_met"},
        )
    ]
    call = db.calls[0]
    sql = compact_sql(call.query)
    assert "from promotion_evaluations" in sql
    assert "where promotion_run_id = %s" in sql
    assert "and ad_experiment_id is not null" in sql
    assert "select distinct on (ad_experiment_id)" in sql
    assert "order by ad_experiment_id asc, created_at desc, evaluation_id desc" in sql
    assert call.params == ("prun_banner_001_loop_1",)


def test_evaluation_metric_repository_counts_inflow_rate_events() -> None:
    client = FakeClickHouseClient(rows=[(4, 10)])
    repo = EvaluationMetricRepository(client)

    counts = repo.count_inflow_rate(
        ad_experiment_record(goal_metric=GoalMetric.INFLOW_RATE.value)
    )

    assert counts.numerator_count == 4
    assert counts.denominator_count == 10
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from promotion_touch_events" in sql
    assert "campaign_landing" in sql
    assert "campaign_redirect_click" in sql
    assert "project_id = {project_id:string}" in sql
    assert "promotion_run_id = {promotion_run_id:string}" in sql
    assert "ad_experiment_id = {ad_experiment_id:string}" in sql
    assert call.params == {
        "project_id": "hotel-client-a",
        "promotion_run_id": "prun_banner_001_loop_1",
        "ad_experiment_id": "adexp_family_trip_001",
    }


def test_evaluation_metric_repository_counts_booking_conversion_with_nullable_keys() -> None:
    client = FakeClickHouseClient(rows=[(0, 10)])
    repo = EvaluationMetricRepository(client)

    counts = repo.count_booking_conversion_rate(
        ad_experiment_record(goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value)
    )

    assert counts.numerator_count == 0
    assert counts.denominator_count == 10
    call = client.calls[0]
    sql = compact_sql(call.query)
    assert "from booking_outcome_events" in sql
    assert "from promotion_touch_events" in sql
    assert "booking_complete" in sql
    assert "promotion_click" in sql
    assert "promotion_run_id is not null" in sql
    assert "ad_experiment_id is not null" in sql
    assert call.params == {
        "project_id": "hotel-client-a",
        "promotion_run_id": "prun_banner_001_loop_1",
        "ad_experiment_id": "adexp_family_trip_001",
    }


def test_promotion_evaluation_status_enum_does_not_emit_goal_near() -> None:
    statuses = {status.value for status in PromotionEvaluationStatus}
    evaluation = PromotionEvaluationWrite(
        evaluation_id="eval_adexp_family_trip_001",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        ad_experiment_id="adexp_family_trip_001",
        segment_id="seg_family_trip",
        content_id="content_family_trip_001",
        content_option_id="option_a",
        metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        target_value=Decimal("0.030000"),
        actual_value=Decimal("0.025000"),
        numerator_count=25,
        denominator_count=1000,
        sample_size=1000,
        basis=GoalBasis.ALL_SEGMENTS.value,
        status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
        feedback=None,
        next_loop_required=True,
        result_json={"failed_segment_ids": ["seg_family_trip"]},
    )

    assert "goal_near" not in statuses
    assert evaluation.status == PromotionEvaluationStatus.GOAL_NOT_MET.value


def ad_experiment_record(
    *,
    goal_metric: str = GoalMetric.BOOKING_CONVERSION_RATE.value,
) -> Any:
    return type(
        "AdExperimentLike",
        (),
        {
            "ad_experiment_id": "adexp_family_trip_001",
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "promotion_run_id": "prun_banner_001_loop_1",
            "analysis_id": "analysis_banner_001",
            "generation_id": "generation_banner_001",
            "segment_id": "seg_family_trip",
            "segment_name": "Family hotel trip",
            "content_id": "content_family_trip_001",
            "content_option_id": "option_a",
            "channel": Channel.ONSITE_BANNER.value,
            "loop_count": 1,
            "status": AdExperimentStatus.RUNNING.value,
            "goal_metric": goal_metric,
            "goal_target_value": Decimal("0.030000"),
            "goal_basis": GoalBasis.ALL_SEGMENTS.value,
        },
    )()
