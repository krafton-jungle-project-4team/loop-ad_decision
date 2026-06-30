from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterable

from app.contents.assets import ContentAssetService, InMemoryAssetStorage
from app.contents.generators import MockContentGenerator, PartialContentGenerationError
from app.contents.prompt_builder import ContentPromptBuilder
from app.contents.repository import GenerationLockUnavailable
from app.contents.service import ContentGenerationService
from app.contents.types import (
    ACTION_STATUS_CONTENT_GENERATED,
    ACTION_STATUS_FAILED,
    ACTION_STATUS_RECOMMENDED,
    ERROR_TYPE_CONTENT_GENERATION_FAILED,
    GENERATION_MODEL_MOCK,
    GENERATION_STATUS_GENERATED,
    VARIANT_KEYS,
    GeneratedContentDraft,
    GeneratedContentRecord,
    RecommendationActionTarget,
    SegmentContext,
)


class FakeContentRepository:
    def __init__(
        self,
        targets: list[RecommendationActionTarget],
        *,
        lock_available: bool = True,
    ) -> None:
        self.targets = targets
        self.lock_available = lock_available
        self.records: dict[tuple[int | str, int, str], GeneratedContentRecord] = {}
        self.drafts: dict[tuple[int | str, int, str], GeneratedContentDraft] = {}
        self.action_statuses: dict[int, str] = {}
        self.action_errors: dict[int, dict[str, str]] = {}
        self.locked: list[tuple[int | str, int]] = []
        self.next_id = 1

    def list_generation_targets(
        self,
        *,
        project_id: int | str,
        analysis_date: str,
        eligible_statuses: tuple[str, ...],
    ) -> Iterable[RecommendationActionTarget]:
        return [
            target
            for target in self.targets
            if target.project_id == project_id
            and str(target.analysis_date) == analysis_date
            and target.status in eligible_statuses
        ]

    @contextmanager
    def generation_lock(self, *, project_id: int | str, recommendation_action_id: int):
        if not self.lock_available:
            raise GenerationLockUnavailable("generation already in progress")
        self.locked.append((project_id, recommendation_action_id))
        yield

    def get_generated_content(
        self,
        *,
        project_id: int | str,
        recommendation_action_id: int,
        variant_key: str,
    ) -> GeneratedContentRecord | None:
        return self.records.get((project_id, recommendation_action_id, variant_key))

    def upsert_generated_content(
        self,
        *,
        draft: GeneratedContentDraft,
        force: bool,
    ) -> GeneratedContentRecord:
        key = (draft.project_id, draft.recommendation_action_id, draft.variant_key)
        existing = self.records.get(key)
        if existing is None:
            record = GeneratedContentRecord(
                id=self.next_id,
                project_id=draft.project_id,
                recommendation_action_id=draft.recommendation_action_id,
                segment_id=draft.segment_id,
                variant_key=draft.variant_key,
                generation_status=draft.generation_status,
                created_run_id=draft.created_run_id,
                metadata=draft.metadata,
            )
            self.next_id += 1
            self.records[key] = record
        else:
            assert force is True
            record = GeneratedContentRecord(
                id=existing.id,
                project_id=existing.project_id,
                recommendation_action_id=existing.recommendation_action_id,
                segment_id=existing.segment_id,
                variant_key=existing.variant_key,
                generation_status=draft.generation_status,
                created_run_id=draft.created_run_id,
                metadata=draft.metadata,
            )
            self.records[key] = record
        self.drafts[key] = draft
        return record

    def mark_action_content_generated(self, *, recommendation_action_id: int) -> None:
        self.action_statuses[recommendation_action_id] = ACTION_STATUS_CONTENT_GENERATED

    def mark_action_failed(
        self,
        *,
        recommendation_action_id: int,
        error_type: str,
        error_message: str,
    ) -> None:
        self.action_statuses[recommendation_action_id] = ACTION_STATUS_FAILED
        self.action_errors[recommendation_action_id] = {
            "error_type": error_type,
            "error_message": error_message,
        }


class FailingGenerator:
    def generate(self, *, target: RecommendationActionTarget, variant_key: str):
        raise RuntimeError(f"boom {variant_key}")


class PartialFailingGenerator:
    def generate(self, *, target: RecommendationActionTarget, variant_key: str):
        draft = MockContentGenerator().generate(target=target, variant_key=variant_key)
        raise PartialContentGenerationError("partial failure", draft)


class StaticModelGenerator(MockContentGenerator):
    def __init__(self, generation_model: str) -> None:
        super().__init__(generation_model=generation_model)


def make_target(
    *,
    action_id: int = 10,
    status: str = ACTION_STATUS_RECOMMENDED,
    is_default: bool = False,
    root_cause: dict[str, object] | None = None,
) -> RecommendationActionTarget:
    segment = SegmentContext(
        id=1 if not is_default else 99,
        segment_key="age_30s__gender_male__channel_kakao__category_fresh"
        if not is_default
        else "default",
        name="카카오 유입 30대 남성 신선식품",
        is_default=is_default,
        attributes={"category": "신선식품", "landing_url": "/collections/fresh"},
    )
    return RecommendationActionTarget(
        id=action_id,
        project_id="demo-shop",
        recommendation_result_id=100,
        action_key="highlight_benefit_banner",
        action_type="banner",
        status=status,
        segment=segment,
        analysis_date="2021-01-04",
        action_title="혜택 강조 배너",
        action_description="상품 조회 후 장바구니 전환율을 높이기 위한 혜택 노출",
        metrics={"view_to_purchase_rate": 0.03},
        root_cause=root_cause or {"cause_key": "view_to_cart", "summary": "혜택 노출 부족"},
    )


def make_service(
    repository: FakeContentRepository,
    generator=None,
    asset_service=None,
) -> ContentGenerationService:
    return ContentGenerationService(
        repository=repository,
        generator=generator or MockContentGenerator(),
        asset_service=asset_service,
    )


def test_mock_generator_creates_control_and_treatment_contents() -> None:
    repository = FakeContentRepository([make_target()])
    service = make_service(repository)

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
    )

    assert summary.actions_seen == 1
    assert summary.actions_created == 1
    assert summary.created_actions == 1
    assert summary.variants_created == 2
    assert summary.created_contents == 2
    assert summary.mock_calls == 2
    assert summary.llm_calls == 0
    assert summary.elapsed_ms >= 0
    assert set(repository.drafts) == {
        ("demo-shop", 10, "control"),
        ("demo-shop", 10, "treatment_a"),
    }
    assert all(draft.generation_status == GENERATION_STATUS_GENERATED for draft in repository.drafts.values())
    assert all(draft.generation_model == GENERATION_MODEL_MOCK for draft in repository.drafts.values())
    assert repository.action_statuses[10] == ACTION_STATUS_CONTENT_GENERATED
    assert repository.locked == [("demo-shop", 10)]


def test_mock_generator_changes_banner_copy_by_root_cause_with_single_action() -> None:
    generator = MockContentGenerator()

    view_to_cart = generator.generate(
        target=make_target(root_cause={"cause_key": "view_to_cart"}),
        variant_key="treatment_a",
    )
    cart_to_checkout = generator.generate(
        target=make_target(root_cause={"cause_key": "cart_to_checkout"}),
        variant_key="treatment_a",
    )
    checkout_to_purchase = generator.generate(
        target=make_target(root_cause={"cause_key": "checkout_to_purchase"}),
        variant_key="treatment_a",
    )
    stockout = generator.generate(
        target=make_target(root_cause={"cause_key": "stockout"}),
        variant_key="treatment_a",
    )

    assert {view_to_cart.content_type, cart_to_checkout.content_type} == {"banner"}
    assert view_to_cart.cta_label == "혜택 보기"
    assert cart_to_checkout.cta_label == "쿠폰 받기"
    assert checkout_to_purchase.cta_label == "지금 구매하기"
    assert stockout.cta_label == "대체 상품 보기"
    assert len(
        {
            view_to_cart.title,
            cart_to_checkout.title,
            checkout_to_purchase.title,
            stockout.title,
        }
    ) == 4
    assert "cart_to_checkout" in cart_to_checkout.image_prompt


def test_run_id_is_stored_on_generated_content_drafts() -> None:
    repository = FakeContentRepository([make_target()])
    service = make_service(repository)

    service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        run_id=123,
    )

    assert all(draft.created_run_id == 123 for draft in repository.drafts.values())
    assert all(record.created_run_id == 123 for record in repository.records.values())


def test_default_segment_is_never_processed() -> None:
    repository = FakeContentRepository([make_target(is_default=True)])
    service = make_service(repository)

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        force=True,
    )

    assert summary.actions_seen == 1
    assert summary.actions_skipped == 1
    assert repository.drafts == {}
    assert repository.action_statuses == {}


def test_force_false_skips_existing_content_without_mapping_side_effects() -> None:
    repository = FakeContentRepository([make_target()])
    repository.records[("demo-shop", 10, "control")] = GeneratedContentRecord(
        id=1,
        project_id="demo-shop",
        recommendation_action_id=10,
        segment_id=1,
        variant_key="control",
        generation_status=GENERATION_STATUS_GENERATED,
    )
    repository.records[("demo-shop", 10, "treatment_a")] = GeneratedContentRecord(
        id=2,
        project_id="demo-shop",
        recommendation_action_id=10,
        segment_id=1,
        variant_key="treatment_a",
        generation_status=GENERATION_STATUS_GENERATED,
    )
    service = make_service(repository)

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        force=False,
    )

    assert summary.actions_skipped == 1
    assert summary.variants_skipped == 2
    assert repository.drafts == {}
    assert repository.action_statuses[10] == ACTION_STATUS_CONTENT_GENERATED


def test_force_true_updates_existing_ai_generated_content_row() -> None:
    repository = FakeContentRepository([make_target()])
    repository.records[("demo-shop", 10, "control")] = GeneratedContentRecord(
        id=1,
        project_id="demo-shop",
        recommendation_action_id=10,
        segment_id=1,
        variant_key="control",
        generation_status=GENERATION_STATUS_GENERATED,
    )
    service = make_service(repository)

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        force=True,
    )

    assert summary.actions_created == 1
    assert summary.created_actions == 1
    assert summary.variants_created == 1
    assert summary.variants_updated == 1
    assert summary.created_contents == 1
    assert summary.updated_contents == 1
    assert repository.records[("demo-shop", 10, "control")].id == 1
    assert ("demo-shop", 10, "control") in repository.drafts


def test_force_true_with_only_existing_content_is_counted_as_update() -> None:
    repository = FakeContentRepository([make_target()])
    repository.records[("demo-shop", 10, "control")] = GeneratedContentRecord(
        id=1,
        project_id="demo-shop",
        recommendation_action_id=10,
        segment_id=1,
        variant_key="control",
        generation_status=GENERATION_STATUS_GENERATED,
    )
    repository.records[("demo-shop", 10, "treatment_a")] = GeneratedContentRecord(
        id=2,
        project_id="demo-shop",
        recommendation_action_id=10,
        segment_id=1,
        variant_key="treatment_a",
        generation_status=GENERATION_STATUS_GENERATED,
    )
    service = make_service(repository)

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        force=True,
    )

    assert summary.actions_updated == 1
    assert summary.updated_actions == 1
    assert summary.updated_contents == 2
    assert repository.records[("demo-shop", 10, "control")].id == 1
    assert repository.records[("demo-shop", 10, "treatment_a")].id == 2


def test_generation_lock_unavailable_skips_without_generator_call() -> None:
    repository = FakeContentRepository([make_target()], lock_available=False)
    service = make_service(repository, FailingGenerator())

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
    )

    assert summary.actions_skipped == 1
    assert summary.skipped_actions == 1
    assert summary.mock_calls == 0
    assert summary.llm_calls == 0
    assert repository.drafts == {}
    assert repository.action_statuses == {}
    assert summary.results[0].error_message == "generation already in progress"


def test_non_mock_generation_model_is_counted_as_llm_call() -> None:
    repository = FakeContentRepository([make_target()])
    service = make_service(repository, StaticModelGenerator("gpt-test"))

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
    )

    assert summary.mock_calls == 0
    assert summary.llm_calls == 2
    assert all(draft.generation_model == "gpt-test" for draft in repository.drafts.values())


def test_failed_actions_are_retried_only_with_force_true() -> None:
    target = make_target(status=ACTION_STATUS_FAILED)
    repository = FakeContentRepository([target])
    service = make_service(repository)

    no_force = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        force=False,
    )
    force = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        force=True,
    )

    assert no_force.actions_seen == 0
    assert force.actions_seen == 1
    assert force.actions_created == 1


def test_generation_failure_marks_action_failed_without_required_failed_content_row() -> None:
    repository = FakeContentRepository([make_target()])
    service = make_service(repository, FailingGenerator())

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
    )

    assert summary.actions_failed == 1
    assert repository.drafts == {}
    assert repository.action_statuses[10] == ACTION_STATUS_FAILED
    assert repository.action_errors[10]["error_type"] == ERROR_TYPE_CONTENT_GENERATION_FAILED
    assert "boom control" in repository.action_errors[10]["error_message"]


def test_partial_failure_can_store_failed_content_when_required_fields_exist() -> None:
    repository = FakeContentRepository([make_target()])
    service = make_service(repository, PartialFailingGenerator())

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
    )

    assert summary.actions_failed == 1
    assert repository.drafts[("demo-shop", 10, "control")].generation_status == "failed"
    assert repository.action_statuses[10] == ACTION_STATUS_FAILED


def test_partial_failure_failed_content_keeps_run_id() -> None:
    repository = FakeContentRepository([make_target()])
    service = make_service(repository, PartialFailingGenerator())

    service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        run_id=456,
    )

    assert repository.drafts[("demo-shop", 10, "control")].created_run_id == 456


def test_prompt_builder_removes_raw_event_and_pii_fields() -> None:
    target = replace(
        make_target(),
        metrics={
            "view_to_purchase_rate": 0.03,
            "external_user_id": "user-1",
            "nested": {"session_id": "s1", "safe": 1},
        },
        root_cause={"raw_events": [{"event_id": "evt-1"}], "cause_key": "view_to_cart"},
    )

    prompt = ContentPromptBuilder().build(target, "treatment_a")

    assert "external_user_id" not in prompt["metrics"]
    assert "session_id" not in prompt["metrics"]["nested"]
    assert prompt["metrics"]["nested"]["safe"] == 1
    assert "raw_events" not in prompt["root_cause"]
    assert prompt["root_cause"]["cause_key"] == "view_to_cart"


def test_content_generated_actions_are_reprocessed_only_with_force_true() -> None:
    target = make_target(status=ACTION_STATUS_CONTENT_GENERATED)
    repository = FakeContentRepository([target])
    service = make_service(repository)

    no_force = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
    )
    force = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
        force=True,
    )

    assert no_force.actions_seen == 0
    assert force.actions_seen == 1
    assert set(force.results[0].created_variant_keys) == set(VARIANT_KEYS)


def test_asset_service_populates_image_url_and_media_key_before_upsert() -> None:
    repository = FakeContentRepository([make_target()])
    storage = InMemoryAssetStorage(public_base_url="https://cdn.example.com")
    asset_service = ContentAssetService(storage=storage)
    service = make_service(repository, asset_service=asset_service)

    summary = service.generate_for_actions(
        project_id="demo-shop",
        analysis_date="2021-01-04",
    )

    control_draft = repository.drafts[("demo-shop", 10, "control")]
    assert summary.variants_created == 2
    assert control_draft.image_url == (
        "https://cdn.example.com/generated-contents/projects/demo-shop/"
        "actions/10/variants/control/banner.svg"
    )
    assert control_draft.media_s3_key == (
        "generated-contents/projects/demo-shop/actions/10/variants/control/banner.svg"
    )
    assert storage.objects[control_draft.media_s3_key].content_type == "image/svg+xml"
