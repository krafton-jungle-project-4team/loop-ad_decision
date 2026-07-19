from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import pytest

from app.generation.brand_context import BrandContextSnapshot
from app.generation.prompt_builder import (
    GenerationPromptInput,
    PromotionOfferLink,
    PromotionPromptInput,
    TargetSegmentPromptInput,
)
from app.generation.repositories import GenerationRunRecord
from app.generation.schemas import (
    ContentChannel,
    GenerationRequest,
    GenerationStatus,
)
from app.generation.submission import (
    GenerationIdempotencyConflict,
    GenerationInputUnavailable,
    GenerationSnapshotError,
    GenerationSubmissionService,
    GenerationSubmissionUnavailable,
    build_generation_input_snapshot,
    generation_id_for_request,
    generation_request_fingerprint,
    prompt_inputs_from_snapshot,
)


def generation_request(**overrides: Any) -> GenerationRequest:
    values: dict[str, Any] = {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "content_option_count": 2,
        "operator_instruction": "Keep the CTA concise.",
    }
    values.update(overrides)
    return GenerationRequest(**values)


def promotion_input() -> PromotionPromptInput:
    return PromotionPromptInput(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel=ContentChannel.ONSITE_BANNER,
        goal_metric="booking_conversion_rate",
        goal_target_value="0.030000",
        goal_basis="all_segments",
        message_brief="Drive summer hotel bookings.",
        landing_url="https://demo-stay.example.com/summer",
    )


def offer_catalog(*offer_ids: str) -> dict[str, Any]:
    return {
        "schema_version": "stayloop.promotion-price-catalog.v1",
        "catalog_id": "black-friday-hotels",
        "catalog_version": "v2",
        "hotels": [
            {
                "offer_id": offer_id,
                "hotel_name": offer_id.replace("-", " ").title(),
                "destination_id": "jeju",
                "currency": "KRW",
                "sale_price_per_night": 278000,
                "original_price_per_night": 342000,
                "discount_rate_percent": 19,
                "image_path": f"/stayloop/promotions/{offer_id}.png",
                "asset_id": f"hotel-{offer_id}-hero",
            }
            for offer_id in offer_ids
        ],
    }


def target_segment_input(
    segment_id: str = "seg_repeat_hotel_no_booking",
    *,
    priority: str = "high",
) -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id="analysis_banner_001",
        promotion_id="promo_banner_001",
        segment_id=segment_id,
        segment_name=f"Audience {segment_id}",
        content_brief_json={
            "keywords": ["refundable rooms", "summer stay"],
            "message_direction": "Emphasize verified hotel benefits.",
        },
        segment_vector_id=f"segvec_{segment_id}",
        estimated_size=1342,
        priority=priority,
        content_slug=segment_id.removeprefix("seg_"),
        natural_language_query="repeat hotel viewers who did not book",
        generated_sql="SELECT user_id FROM hotel_detail_events",
        sample_ratio="0.018000",
        source="analysis",
        query_preview_id="seg_query_preview_001",
        status="approved",
    )


class FakeConnection:
    def __init__(
        self,
        events: list[str],
        *,
        commit_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.commit_error = commit_error
        self.commit_count = 0
        self.rollback_count = 0

    def commit(self) -> None:
        self.events.append("commit")
        self.commit_count += 1
        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self) -> None:
        self.events.append("rollback")
        self.rollback_count += 1


class InMemorySubmissionRepository:
    def __init__(
        self,
        events: list[str],
        *,
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        self.submitted_records: list[GenerationRunRecord] = []

    def create_or_get_idempotent(
        self,
        record: GenerationRunRecord,
    ) -> tuple[dict[str, Any], bool]:
        self.events.append(f"persist:{record.status}")
        self.submitted_records.append(record)
        if self.error is not None:
            raise self.error
        assert record.idempotency_key is not None
        key = (record.project_id, record.idempotency_key)
        existing = self.records.get(key)
        if existing is not None:
            return existing, False
        persisted = asdict(record)
        self.records[key] = persisted
        return persisted, True


class FakeInputReader:
    def __init__(
        self,
        events: list[str],
        *,
        promotion: PromotionPromptInput | None,
        target_segments: list[TargetSegmentPromptInput],
        promotion_error: ValueError | None = None,
    ) -> None:
        self.events = events
        self.promotion = promotion
        self.target_segments = target_segments
        self.promotion_error = promotion_error

    def get_promotion_input(
        self,
        _request: GenerationRequest,
    ) -> PromotionPromptInput | None:
        self.events.append("read:promotion")
        if self.promotion_error is not None:
            raise self.promotion_error
        return self.promotion

    def list_target_segment_inputs(
        self,
        _request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        self.events.append("read:targets")
        return list(self.target_segments)


class FakeCoordinator:
    def __init__(self, events: list[str], *, accepting: bool = True) -> None:
        self.events = events
        self._accepting = accepting
        self.wake_count = 0

    @property
    def accepting(self) -> bool:
        return self._accepting

    def wake(self) -> None:
        self.events.append("wake")
        self.wake_count += 1


class FakeBrandContextRepository:
    def __init__(
        self,
        events: list[str],
        *,
        catalog: dict[str, Any],
    ) -> None:
        self.events = events
        self.catalog = catalog
        self.snapshot = BrandContextSnapshot(
            context_version="v2",
            manifest_key="brand-context/hotel-client-a/manifests/v2/manifest.json",
            manifest_sha256="a" * 64,
            guide_version="v2",
            asset_manifest_version="v2",
            catalog_version="v2",
        )

    def resolve_snapshot(self, *, project_id: str) -> BrandContextSnapshot:
        self.events.append(f"read:brand-context:{project_id}")
        return self.snapshot

    def load_offer_catalog(
        self,
        *,
        project_id: str,
        snapshot: BrandContextSnapshot,
    ) -> dict[str, Any]:
        assert snapshot is self.snapshot
        self.events.append(f"read:offer-catalog:{project_id}")
        return self.catalog


def build_service(
    *,
    events: list[str],
    connection: FakeConnection | None = None,
    repository: InMemorySubmissionRepository | None = None,
    promotion: PromotionPromptInput | None = None,
    target_segments: list[TargetSegmentPromptInput] | None = None,
    coordinator: FakeCoordinator | None = None,
    brand_context_repository: FakeBrandContextRepository | None = None,
    promotion_error: ValueError | None = None,
) -> tuple[
    GenerationSubmissionService,
    FakeConnection,
    InMemorySubmissionRepository,
    FakeCoordinator,
]:
    connection = connection or FakeConnection(events)
    repository = repository or InMemorySubmissionRepository(events)
    coordinator = coordinator or FakeCoordinator(events)
    input_reader = FakeInputReader(
        events,
        promotion=promotion if promotion is not None else promotion_input(),
        target_segments=(
            target_segments
            if target_segments is not None
            else [target_segment_input()]
        ),
        promotion_error=promotion_error,
    )
    return (
        GenerationSubmissionService(
            connection=connection,
            generation_run_repository=repository,
            generation_input_reader=input_reader,
            brand_context_repository=brand_context_repository,
            coordinator=coordinator,
        ),
        connection,
        repository,
        coordinator,
    )


def test_submit_persists_requested_row_before_commit_and_wake() -> None:
    events: list[str] = []
    service, connection, repository, coordinator = build_service(events=events)

    response = service.submit(
        generation_request(),
        idempotency_key=" generation:banner:001 ",
    )

    assert events == [
        "read:promotion",
        "read:targets",
        "persist:requested",
        "commit",
        "wake",
    ]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert coordinator.wake_count == 1
    assert response.status is GenerationStatus.REQUESTED
    assert response.promotion_id == "promo_banner_001"

    record = repository.submitted_records[0]
    assert record.status == GenerationStatus.REQUESTED.value
    assert record.idempotency_key == "generation:banner:001"
    assert record.output_json is None
    assert record.started_at is None
    assert record.finished_at is None
    assert record.worker_id is None
    assert record.lease_token is None
    assert record.input_json["schema_version"] == "generation.request.v1"
    assert record.input_json["target_segment_ids"] == [
        "seg_repeat_hotel_no_booking"
    ]
    assert len(record.request_fingerprint or "") == 64


def test_generation_snapshot_preserves_offer_links_and_verified_catalog() -> None:
    promotion = replace(
        promotion_input(),
        channel=ContentChannel.EMAIL,
        offer_links=(
            PromotionOfferLink(
                offer_id="jeju-ocean-breeze-006",
                destination_url=(
                    "https://demo-shoppingmall.dev.loop-ad.org/"
                    "hotel/jeju-ocean-breeze-006"
                ),
            ),
        ),
    )
    offer_catalog = {
        "schema_version": "stayloop.promotion-price-catalog.v1",
        "catalog_id": "black-friday-hotels",
        "catalog_version": "v2",
        "hotels": [
            {
                "offer_id": "jeju-ocean-breeze-006",
                "hotel_name": "Jeju Ocean Breeze Resort",
                "destination_id": "jeju",
                "currency": "KRW",
                "sale_price_per_night": 278000,
                "original_price_per_night": 342000,
                "discount_rate_percent": 19,
                "image_path": "/stayloop/promotions/jeju-resort-exterior.png",
                "asset_id": "hotel-jeju-ocean-breeze-006-hero",
            }
        ],
    }

    snapshot = build_generation_input_snapshot(
        request=generation_request(content_option_count=3),
        promotion=promotion,
        target_segments=[target_segment_input()],
        offer_catalog=offer_catalog,
    )
    prompt_inputs = prompt_inputs_from_snapshot(snapshot)

    assert snapshot["promotion"]["offer_links"] == (
        {
            "offer_id": "jeju-ocean-breeze-006",
            "destination_url": (
                "https://demo-shoppingmall.dev.loop-ad.org/"
                "hotel/jeju-ocean-breeze-006"
            ),
        },
    )
    assert prompt_inputs[0].promotion.offer_links == promotion.offer_links
    assert prompt_inputs[0].offer_catalog == offer_catalog


def test_submit_accepts_email_offer_link_from_current_catalog() -> None:
    events: list[str] = []
    offer_id = "jeju-ocean-breeze-006"
    promotion = replace(
        promotion_input(),
        channel=ContentChannel.EMAIL,
        offer_links=(
            PromotionOfferLink(
                offer_id=offer_id,
                destination_url=(
                    "https://demo-shoppingmall.dev.loop-ad.org/hotel/"
                    f"{offer_id}"
                ),
            ),
        ),
    )
    brand_context_repository = FakeBrandContextRepository(
        events,
        catalog=offer_catalog(offer_id),
    )
    service, connection, repository, coordinator = build_service(
        events=events,
        promotion=promotion,
        brand_context_repository=brand_context_repository,
    )

    response = service.submit(
        generation_request(),
        idempotency_key="email-offer-link",
    )

    assert response.status is GenerationStatus.REQUESTED
    assert events == [
        "read:brand-context:hotel-client-a",
        "read:promotion",
        "read:offer-catalog:hotel-client-a",
        "read:targets",
        "persist:requested",
        "commit",
        "wake",
    ]
    assert repository.submitted_records[0].input_json["offer_catalog"] == (
        brand_context_repository.catalog
    )
    assert connection.commit_count == 1
    assert coordinator.wake_count == 1


def test_submit_rejects_email_offer_id_missing_from_current_catalog() -> None:
    events: list[str] = []
    missing_offer_id = "jeju-missing-hotel-999"
    promotion = replace(
        promotion_input(),
        channel=ContentChannel.EMAIL,
        offer_links=(
            PromotionOfferLink(
                offer_id=missing_offer_id,
                destination_url=(
                    "https://demo-shoppingmall.dev.loop-ad.org/hotel/"
                    f"{missing_offer_id}"
                ),
            ),
        ),
    )
    service, connection, repository, coordinator = build_service(
        events=events,
        promotion=promotion,
        brand_context_repository=FakeBrandContextRepository(
            events,
            catalog=offer_catalog("jeju-ocean-breeze-006"),
        ),
    )

    with pytest.raises(GenerationInputUnavailable, match=missing_offer_id):
        service.submit(generation_request(), idempotency_key="missing-offer")

    assert events == [
        "read:brand-context:hotel-client-a",
        "read:promotion",
        "read:offer-catalog:hotel-client-a",
    ]
    assert repository.submitted_records == []
    assert connection.commit_count == 0
    assert coordinator.wake_count == 0


def test_submit_rejects_noncanonical_email_offer_destination_url() -> None:
    events: list[str] = []
    offer_id = "jeju-ocean-breeze-006"
    promotion = replace(
        promotion_input(),
        channel=ContentChannel.EMAIL,
        offer_links=(
            PromotionOfferLink(
                offer_id=offer_id,
                destination_url=(
                    "https://demo-shoppingmall.dev.loop-ad.org/promotions/summer"
                ),
            ),
        ),
    )
    service, connection, repository, coordinator = build_service(
        events=events,
        promotion=promotion,
        brand_context_repository=FakeBrandContextRepository(
            events,
            catalog=offer_catalog(offer_id),
        ),
    )

    with pytest.raises(GenerationInputUnavailable, match="canonical"):
        service.submit(generation_request(), idempotency_key="wrong-destination")

    assert events == [
        "read:brand-context:hotel-client-a",
        "read:promotion",
        "read:offer-catalog:hotel-client-a",
    ]
    assert repository.submitted_records == []
    assert connection.commit_count == 0
    assert coordinator.wake_count == 0


def test_promotion_input_rejects_duplicate_offer_destination_urls() -> None:
    duplicate_url = (
        "https://demo-shoppingmall.dev.loop-ad.org/hotel/jeju-ocean-breeze-006"
    )

    with pytest.raises(ValueError, match="duplicate destination_url"):
        replace(
            promotion_input(),
            channel=ContentChannel.EMAIL,
            offer_links=(
                PromotionOfferLink(
                    offer_id="jeju-ocean-breeze-006",
                    destination_url=duplicate_url,
                ),
                PromotionOfferLink(
                    offer_id="jeju-aewol-sunset-007",
                    destination_url=duplicate_url,
                ),
            ),
        )


def test_submit_preserves_email_without_offer_links_for_backward_compatibility() -> None:
    events: list[str] = []
    service, connection, repository, coordinator = build_service(
        events=events,
        promotion=replace(promotion_input(), channel=ContentChannel.EMAIL),
    )

    response = service.submit(
        generation_request(),
        idempotency_key="legacy-email-without-offers",
    )

    assert response.status is GenerationStatus.REQUESTED
    assert events == [
        "read:promotion",
        "read:targets",
        "persist:requested",
        "commit",
        "wake",
    ]
    assert repository.submitted_records[0].input_json["promotion"][
        "offer_links"
    ] == ()
    assert connection.commit_count == 1
    assert coordinator.wake_count == 1


def test_submit_rejects_reserved_internal_idempotency_key_prefix() -> None:
    events: list[str] = []
    service, connection, repository, coordinator = build_service(events=events)

    with pytest.raises(ValueError, match="reserved internal prefix"):
        service.submit(
            generation_request(),
            idempotency_key="loopad-internal:next-loop:attacker-controlled",
        )

    assert events == []
    assert repository.submitted_records == []
    assert connection.commit_count == 0
    assert coordinator.wake_count == 0


def test_submit_stops_before_reads_when_coordinator_is_shutting_down() -> None:
    events: list[str] = []
    coordinator = FakeCoordinator(events, accepting=False)
    service, connection, repository, _ = build_service(
        events=events,
        coordinator=coordinator,
    )

    with pytest.raises(GenerationSubmissionUnavailable, match="shutting down"):
        service.submit(generation_request(), idempotency_key="stable-key")

    assert events == []
    assert repository.submitted_records == []
    assert connection.commit_count == 0


def test_submit_rejects_duplicate_segment_snapshot_before_insert() -> None:
    events: list[str] = []
    duplicate = target_segment_input()
    service, connection, repository, coordinator = build_service(
        events=events,
        target_segments=[duplicate, duplicate],
    )

    with pytest.raises(GenerationSnapshotError, match="duplicates"):
        service.submit(generation_request(), idempotency_key="stable-key")

    assert repository.submitted_records == []
    assert connection.commit_count == 0
    assert coordinator.wake_count == 0


def test_same_idempotency_key_and_fingerprint_returns_existing_run() -> None:
    events: list[str] = []
    service, connection, repository, coordinator = build_service(events=events)

    first = service.submit(generation_request(), idempotency_key="stable-key")
    second = service.submit(generation_request(), idempotency_key="stable-key")

    assert second == first
    assert len(repository.records) == 1
    assert repository.submitted_records[0].request_fingerprint == (
        repository.submitted_records[1].request_fingerprint
    )
    assert connection.commit_count == 2
    assert connection.rollback_count == 0
    assert coordinator.wake_count == 2


def test_same_idempotency_key_with_different_fingerprint_rolls_back() -> None:
    events: list[str] = []
    service, connection, repository, coordinator = build_service(events=events)
    service.submit(generation_request(), idempotency_key="stable-key")

    with pytest.raises(GenerationIdempotencyConflict, match="different"):
        service.submit(
            generation_request(content_option_count=3),
            idempotency_key="stable-key",
        )

    assert len(repository.records) == 1
    assert connection.commit_count == 1
    assert connection.rollback_count == 1
    assert coordinator.wake_count == 1
    assert events[-2:] == ["persist:requested", "rollback"]


@pytest.mark.parametrize("failure_point", ["persist", "commit"])
def test_submit_rolls_back_and_does_not_wake_on_transaction_failure(
    failure_point: str,
) -> None:
    events: list[str] = []
    error = RuntimeError(f"{failure_point} failed")
    connection = FakeConnection(
        events,
        commit_error=error if failure_point == "commit" else None,
    )
    repository = InMemorySubmissionRepository(
        events,
        error=error if failure_point == "persist" else None,
    )
    service, connection, _, coordinator = build_service(
        events=events,
        connection=connection,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match=failure_point):
        service.submit(generation_request(), idempotency_key="stable-key")

    assert connection.rollback_count == 1
    assert coordinator.wake_count == 0
    assert events[-1] == "rollback"


def test_submit_rejects_missing_promotion_before_persisting() -> None:
    events: list[str] = []
    connection = FakeConnection(events)
    repository = InMemorySubmissionRepository(events)
    coordinator = FakeCoordinator(events)
    service = GenerationSubmissionService(
        connection=connection,
        generation_run_repository=repository,
        generation_input_reader=FakeInputReader(
            events,
            promotion=None,
            target_segments=[target_segment_input()],
        ),
        coordinator=coordinator,
    )

    with pytest.raises(GenerationInputUnavailable, match="promotion input"):
        service.submit(generation_request(), idempotency_key="stable-key")

    assert events == ["read:promotion"]
    assert repository.submitted_records == []
    assert connection.commit_count == 0
    assert coordinator.wake_count == 0


def test_submit_maps_invalid_stored_offer_links_to_input_unavailable() -> None:
    events: list[str] = []
    service, connection, repository, coordinator = build_service(
        events=events,
        promotion_error=ValueError(
            "promotion offer_links must not contain duplicate destination_url"
        ),
    )

    with pytest.raises(
        GenerationInputUnavailable,
        match="duplicate destination_url",
    ):
        service.submit(generation_request(), idempotency_key="invalid-offer-links")

    assert events == ["read:promotion"]
    assert repository.submitted_records == []
    assert connection.commit_count == 0
    assert coordinator.wake_count == 0


def test_submit_rejects_missing_confirmed_targets_before_persisting() -> None:
    events: list[str] = []
    service, connection, repository, coordinator = build_service(
        events=events,
        target_segments=[],
    )

    with pytest.raises(GenerationInputUnavailable, match="confirmed"):
        service.submit(generation_request(), idempotency_key="stable-key")

    assert events == ["read:promotion", "read:targets"]
    assert repository.submitted_records == []
    assert connection.commit_count == 0
    assert coordinator.wake_count == 0


def test_snapshot_roundtrip_is_sorted_and_fingerprint_is_deterministic() -> None:
    request = generation_request()
    promotion = promotion_input()
    target_a = target_segment_input("seg_a", priority="medium")
    target_b = target_segment_input("seg_b", priority="high")

    reversed_snapshot = build_generation_input_snapshot(
        request=request,
        promotion=promotion,
        target_segments=[target_b, target_a],
    )
    ordered_snapshot = build_generation_input_snapshot(
        request=request,
        promotion=promotion,
        target_segments=[
            replace(
                target_a,
                content_brief_json={
                    "message_direction": "Emphasize verified hotel benefits.",
                    "keywords": ["refundable rooms", "summer stay"],
                },
            ),
            target_b,
        ],
    )

    assert [
        item["segment_id"] for item in reversed_snapshot["target_segments"]
    ] == ["seg_a", "seg_b"]
    assert reversed_snapshot["target_segment_ids"] == ["seg_a", "seg_b"]
    assert generation_request_fingerprint(reversed_snapshot) == (
        generation_request_fingerprint(ordered_snapshot)
    )
    assert prompt_inputs_from_snapshot(reversed_snapshot) == [
        GenerationPromptInput(
            request=request,
            promotion=promotion,
            target_segment=target_a,
        ),
        GenerationPromptInput(
            request=request,
            promotion=promotion,
            target_segment=target_b,
        ),
    ]

    first_id = generation_id_for_request(
        promotion_id=request.promotion_id,
        project_id=request.project_id,
        idempotency_key="stable-key",
    )
    second_id = generation_id_for_request(
        promotion_id=request.promotion_id,
        project_id=request.project_id,
        idempotency_key="stable-key",
    )
    assert first_id == second_id
    assert len(first_id) <= 100
