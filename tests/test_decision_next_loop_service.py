from __future__ import annotations

from decimal import Decimal
from typing import Sequence

import pytest

from app.decision.next_loop_service import (
    NextLoopAnalysisResult,
    NextLoopConflictError,
    NextLoopGenerationResult,
    NextLoopNotFoundError,
    NextLoopService,
    NextLoopValidationError,
)
from app.decision.repositories import (
    AdExperimentRecord,
    PromotionEvaluationRecord,
    PromotionRecord,
    PromotionRunRecord,
)
from app.decision.schemas import (
    AdExperimentStatus,
    Channel,
    NextLoopRequest,
    PromotionEvaluationStatus,
    PromotionRunStatus,
)


def test_next_loop_prepares_focus_generation_for_goal_not_met_segments_only() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_luxury"],
            failed_ad_experiment_ids=["adexp_luxury_001"],
            operator_instruction="Emphasize breakfast benefits.",
        ),
    )

    assert "status" not in response.model_dump()
    assert response.previous_promotion_run_id == "prun_banner_001_loop_1"
    assert response.next_promotion_run_id is None
    assert response.loop_count == 2
    assert response.next_analysis_id == "analysis_next_001"
    assert response.next_generation_id == "generation_next_001"
    assert response.next_ad_experiments == []
    assert repos.analysis_gateway.calls == [
        (
            "hotel-client-a",
            "camp_summer_2026",
            "promo_banner_001",
            ("seg_luxury",),
            2,
            "prun_banner_001_loop_1",
            ("adexp_luxury_001",),
            "Emphasize breakfast benefits.",
        )
    ]
    assert repos.generation_gateway.calls == [
        (
            "hotel-client-a",
            "camp_summer_2026",
            "promo_banner_001",
            "analysis_next_001",
            ("seg_luxury",),
            2,
            "prun_banner_001_loop_1",
            "generation_banner_001",
            "Emphasize breakfast benefits.",
        )
    ]


def test_next_loop_noops_when_failed_ids_are_empty() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(),
    )

    assert "status" not in response.model_dump()
    assert response.next_promotion_run_id is None
    assert response.next_ad_experiments == []
    assert repos.analysis_gateway.calls == []
    assert repos.generation_gateway.calls == []


@pytest.mark.parametrize(
    "status",
    [
        PromotionEvaluationStatus.GOAL_MET.value,
        PromotionEvaluationStatus.INSUFFICIENT_DATA.value,
        PromotionEvaluationStatus.PARTIAL_GOAL_MET.value,
    ],
)
def test_next_loop_allows_only_goal_not_met_evaluations(status: str) -> None:
    repos = FakeNextLoopRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=status,
            )
        ]
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="only goal_not_met"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_rejects_failed_ids_outside_previous_run() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="failed_segment_ids"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_missing"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_rejects_failed_segment_experiment_mismatch() -> None:
    repos = FakeNextLoopRepos()
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="must match"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_family_trip"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )


def test_next_loop_rejects_max_loop_count_exceeded() -> None:
    repos = FakeNextLoopRepos(
        run=promotion_run_record(loop_count=2),
        promotion=promotion_record(max_loop_count=2),
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="max_loop_count"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_2",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )


def test_next_loop_rejects_existing_next_loop() -> None:
    repos = FakeNextLoopRepos(existing_next_loop=True)
    service = make_service(repos)

    with pytest.raises(NextLoopConflictError, match="already exists"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )


def test_next_loop_rejects_gateway_segments_outside_failed_set() -> None:
    repos = FakeNextLoopRepos(
        analysis_gateway=FakeAnalysisGateway(target_segment_ids=["seg_luxury", "seg_spa"])
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="analysis result"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )

    assert repos.generation_gateway.calls == []


def test_next_loop_rejects_generation_segments_outside_failed_set() -> None:
    repos = FakeNextLoopRepos(
        generation_gateway=FakeGenerationGateway(
            generated_segment_ids=["seg_luxury", "seg_spa"],
        )
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="generation result"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_rejects_generation_that_is_not_completed() -> None:
    repos = FakeNextLoopRepos(
        generation_gateway=FakeGenerationGateway(status="failed"),
    )
    service = make_service(repos)

    with pytest.raises(NextLoopValidationError, match="must be completed"):
        service.create_next_loop(
            promotion_run_id="prun_banner_001_loop_1",
            request=NextLoopRequest(
                failed_segment_ids=["seg_luxury"],
                failed_ad_experiment_ids=["adexp_luxury_001"],
            ),
        )



def test_next_loop_prepares_focus_generation_for_multiple_failed_segments() -> None:
    repos = FakeNextLoopRepos(
        experiments=default_experiments(),
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
            ),
        ],
        analysis_gateway=FakeAnalysisGateway(
            target_segment_ids=["seg_family_trip", "seg_luxury"],
        ),
        generation_gateway=FakeGenerationGateway(
            generated_segment_ids=["seg_family_trip", "seg_luxury"],
        ),
    )
    service = make_service(repos)

    response = service.create_next_loop(
        promotion_run_id="prun_banner_001_loop_1",
        request=NextLoopRequest(
            failed_segment_ids=["seg_family_trip", "seg_luxury"],
            failed_ad_experiment_ids=[
                "adexp_family_trip_001",
                "adexp_luxury_001",
            ],
        ),
    )

    assert "status" not in response.model_dump()
    assert response.next_promotion_run_id is None
    assert response.next_ad_experiments == []
    assert response.next_analysis_id == "analysis_next_001"
    assert response.next_generation_id == "generation_next_001"
    assert response.loop_count == 2


def test_next_loop_rejects_missing_previous_run() -> None:
    repos = FakeNextLoopRepos(run=None)
    service = make_service(repos)

    with pytest.raises(NextLoopNotFoundError):
        service.create_next_loop(
            promotion_run_id="missing",
            request=NextLoopRequest(),
        )


def make_service(repos: "FakeNextLoopRepos") -> NextLoopService:
    return NextLoopService(
        promotion_repository=repos.promotions,
        promotion_run_repository=repos.runs,
        ad_experiment_repository=repos.experiments,
        promotion_evaluation_repository=repos.evaluations,
        analysis_gateway=repos.analysis_gateway,
        generation_gateway=repos.generation_gateway,
    )


class FakeNextLoopRepos:
    def __init__(
        self,
        *,
        run: PromotionRunRecord | None = None,
        promotion: PromotionRecord | None = None,
        experiments: list[AdExperimentRecord] | None = None,
        evaluations: list[PromotionEvaluationRecord] | None = None,
        existing_next_loop: bool = False,
        analysis_gateway: "FakeAnalysisGateway" | None = None,
        generation_gateway: "FakeGenerationGateway" | None = None,
    ) -> None:
        self.promotions = FakePromotionRepository(promotion or promotion_record())
        self.runs = FakePromotionRunRepository(
            run if run is not None else promotion_run_record(),
            existing_next_loop=existing_next_loop,
        )
        self.experiments = FakeAdExperimentRepository(
            experiments if experiments is not None else default_experiments()
        )
        self.evaluations = FakePromotionEvaluationRepository(
            evaluations if evaluations is not None else default_evaluations()
        )
        self.analysis_gateway = analysis_gateway or FakeAnalysisGateway()
        self.generation_gateway = generation_gateway or FakeGenerationGateway()


class FakePromotionRepository:
    def __init__(self, promotion: PromotionRecord | None) -> None:
        self.promotion = promotion

    def get_by_id(self, promotion_id: str) -> PromotionRecord | None:
        if self.promotion is None or self.promotion.promotion_id != promotion_id:
            return None
        return self.promotion


class FakePromotionRunRepository:
    def __init__(
        self,
        run: PromotionRunRecord | None,
        *,
        existing_next_loop: bool = False,
    ) -> None:
        self.run = run
        self.existing_next_loop = existing_next_loop

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        if self.run is None or self.run.promotion_run_id != promotion_run_id:
            return None
        return self.run

    def exists_for_promotion_loop(self, *, promotion_id: str, loop_count: int) -> bool:
        _ = promotion_id, loop_count
        return self.existing_next_loop

    def insert(self, _run: object) -> None:
        raise AssertionError("insert should be called through run creator")

    def update_status(self, *, promotion_run_id: str, status: str) -> None:
        _ = promotion_run_id, status


class FakeAdExperimentRepository:
    def __init__(self, experiments: list[AdExperimentRecord]) -> None:
        self.experiments = experiments

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        return [
            experiment
            for experiment in self.experiments
            if experiment.promotion_run_id == promotion_run_id
        ]

    def get_by_id(self, ad_experiment_id: str) -> AdExperimentRecord | None:
        return next(
            (
                experiment
                for experiment in self.experiments
                if experiment.ad_experiment_id == ad_experiment_id
            ),
            None,
        )

    def insert_many(self, _experiments: Sequence[object]) -> None:
        raise AssertionError("insert_many should be called through run creator")

    def exists_for_run_segment(
        self,
        *,
        promotion_run_id: str,
        segment_id: str,
    ) -> bool:
        _ = promotion_run_id, segment_id
        return False

    def update_status(self, *, ad_experiment_id: str, status: str) -> None:
        _ = ad_experiment_id, status


class FakePromotionEvaluationRepository:
    def __init__(self, evaluations: list[PromotionEvaluationRecord]) -> None:
        self.evaluations = evaluations

    def list_latest_by_run_ad_experiments(
        self,
        promotion_run_id: str,
    ) -> list[PromotionEvaluationRecord]:
        return [
            evaluation
            for evaluation in self.evaluations
            if evaluation.promotion_run_id == promotion_run_id
        ]

    def insert(self, _evaluation: object) -> None:
        raise AssertionError("B6 should not insert evaluations")


class FakeAnalysisGateway:
    def __init__(self, target_segment_ids: list[str] | None = None) -> None:
        self.target_segment_ids = target_segment_ids or ["seg_luxury"]
        self.calls: list[
            tuple[str, str, str, tuple[str, ...], int, str, tuple[str, ...], str | None]
        ] = []

    def start_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        focus_segment_ids: Sequence[str],
        loop_count: int,
        source_promotion_run_id: str,
        source_failed_ad_experiment_ids: Sequence[str],
        operator_instruction: str | None,
    ) -> NextLoopAnalysisResult:
        self.calls.append(
            (
                project_id,
                campaign_id,
                promotion_id,
                tuple(focus_segment_ids),
                loop_count,
                source_promotion_run_id,
                tuple(source_failed_ad_experiment_ids),
                operator_instruction,
            )
        )
        return NextLoopAnalysisResult(
            analysis_id="analysis_next_001",
            target_segment_ids=self.target_segment_ids,
        )


class FakeGenerationGateway:
    def __init__(
        self,
        generated_segment_ids: list[str] | None = None,
        *,
        status: str = "completed",
    ) -> None:
        self.generated_segment_ids = generated_segment_ids or ["seg_luxury"]
        self.status = status
        self.calls: list[
            tuple[str, str, str, str, tuple[str, ...], int, str, str, str | None]
        ] = []

    def start_generation(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
        analysis_id: str,
        focus_segment_ids: Sequence[str],
        loop_count: int,
        source_promotion_run_id: str,
        source_generation_id: str,
        operator_instruction: str | None,
    ) -> NextLoopGenerationResult:
        self.calls.append(
            (
                project_id,
                campaign_id,
                promotion_id,
                analysis_id,
                tuple(focus_segment_ids),
                loop_count,
                source_promotion_run_id,
                source_generation_id,
                operator_instruction,
            )
        )
        return NextLoopGenerationResult(
            generation_id="generation_next_001",
            generated_segment_ids=self.generated_segment_ids,
            status=self.status,
        )


def promotion_record(*, max_loop_count: int = 3) -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel=Channel.ONSITE_BANNER.value,
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.300000"),
        goal_basis="all_segments",
        min_sample_size=10,
        max_loop_count=max_loop_count,
    )


def promotion_run_record(*, loop_count: int = 1) -> PromotionRunRecord:
    return PromotionRunRecord(
        promotion_run_id=f"prun_banner_001_loop_{loop_count}",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id=f"analysis_banner_00{loop_count}",
        generation_id=f"generation_banner_00{loop_count}",
        loop_count=loop_count,
        status=PromotionRunStatus.PARTIAL_GOAL_MET.value,
        goal_snapshot_json={
            "goal_metric": "booking_conversion_rate",
            "goal_target_value": "0.300000",
            "goal_basis": "all_segments",
            "min_sample_size": 10,
        },
    )


def default_experiments() -> list[AdExperimentRecord]:
    return [
        ad_experiment_record(
            ad_experiment_id="adexp_family_trip_001",
            segment_id="seg_family_trip",
        ),
        ad_experiment_record(
            ad_experiment_id="adexp_luxury_001",
            segment_id="seg_luxury",
        ),
    ]


def ad_experiment_record(
    *,
    ad_experiment_id: str,
    segment_id: str,
) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=ad_experiment_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        segment_id=segment_id,
        segment_name=f"Segment {segment_id}",
        content_id=f"content_{segment_id}_001",
        content_option_id=f"option_{segment_id}_001",
        channel=Channel.ONSITE_BANNER.value,
        loop_count=1,
        status=AdExperimentStatus.GOAL_NOT_MET.value,
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.300000"),
        goal_basis="all_segments",
    )


def default_evaluations() -> list[PromotionEvaluationRecord]:
    return [
        evaluation_record(
            ad_experiment_id="adexp_family_trip_001",
            segment_id="seg_family_trip",
            status=PromotionEvaluationStatus.GOAL_MET.value,
        ),
        evaluation_record(
            ad_experiment_id="adexp_luxury_001",
            segment_id="seg_luxury",
            status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
        ),
    ]


def evaluation_record(
    *,
    ad_experiment_id: str,
    segment_id: str,
    status: str,
) -> PromotionEvaluationRecord:
    return PromotionEvaluationRecord(
        evaluation_id=f"eval_{ad_experiment_id}",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        ad_experiment_id=ad_experiment_id,
        segment_id=segment_id,
        content_id=f"content_{segment_id}_001",
        content_option_id=f"option_{segment_id}_001",
        metric="booking_conversion_rate",
        target_value=Decimal("0.300000"),
        actual_value=Decimal("0.100000"),
        numerator_count=1,
        denominator_count=10,
        sample_size=10,
        basis="all_segments",
        status=status,
        feedback=None,
        next_loop_required=status == PromotionEvaluationStatus.GOAL_NOT_MET.value,
        result_json={"status_reason": status},
    )
