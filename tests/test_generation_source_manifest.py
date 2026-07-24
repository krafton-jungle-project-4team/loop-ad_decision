from dataclasses import replace

import pytest

from app.generation.artifacts import ArtifactIdentity, StoredAsset
from app.generation.generator import GeneratedContent
from app.generation.prompt_builder import (
    GenerationPromptInput,
    PromotionPromptInput,
    TargetSegmentPromptInput,
)
from app.generation.schemas import ContentChannel, GenerationRequest
from app.generation.source_manifest import (
    SourceManifest,
    SourceManifestError,
    SourceManifestIdentityError,
    source_manifest_key,
    source_request_fingerprint,
)


IDENTITY = ArtifactIdentity(
    project_id="hotel-client-a",
    promotion_id="promo_banner_001",
    generation_id="generation_banner_001",
    content_id="content_banner_repeat_hotel_001",
)
FINGERPRINT = "a" * 64


def test_source_manifest_round_trip_preserves_exact_generated_source() -> None:
    content = generated_content()
    manifest = SourceManifest(
        identity=IDENTITY,
        channel=ContentChannel.ONSITE_BANNER,
        request_fingerprint=FINGERPRINT,
        content=content,
    )

    restored = SourceManifest.from_bytes(
        manifest.to_bytes(),
        expected_identity=IDENTITY,
        expected_channel=ContentChannel.ONSITE_BANNER,
        expected_request_fingerprint=FINGERPRINT,
    )

    assert restored == manifest
    assert restored.content.image_prompt == content.image_prompt
    assert restored.content.image_url is None


def test_source_manifest_rejects_different_generation_input() -> None:
    manifest = SourceManifest(
        identity=IDENTITY,
        channel=ContentChannel.ONSITE_BANNER,
        request_fingerprint=FINGERPRINT,
        content=generated_content(),
    )

    with pytest.raises(SourceManifestIdentityError):
        SourceManifest.from_bytes(
            manifest.to_bytes(),
            expected_identity=replace(IDENTITY, generation_id="generation_other"),
            expected_channel=ContentChannel.ONSITE_BANNER,
            expected_request_fingerprint=FINGERPRINT,
        )


def test_source_manifest_rejects_different_channel() -> None:
    manifest = SourceManifest(
        identity=IDENTITY,
        channel=ContentChannel.ONSITE_BANNER,
        request_fingerprint=FINGERPRINT,
        content=generated_content(),
    )

    with pytest.raises(SourceManifestIdentityError):
        SourceManifest.from_bytes(
            manifest.to_bytes(),
            expected_identity=IDENTITY,
            expected_channel=ContentChannel.EMAIL,
            expected_request_fingerprint=FINGERPRINT,
        )


def test_source_manifest_never_serializes_generated_image_bytes() -> None:
    content = replace(
        generated_content(),
        image_artifact=StoredAsset(
            storage_key="genai/project/promotion/generation/content/image.png",
            public_url="https://assets.example.test/image.png",
            sha256="b" * 64,
            bytes=123,
            content_type="image/png",
        ),
    )

    with pytest.raises(SourceManifestError, match="image bytes"):
        SourceManifest(
            identity=IDENTITY,
            channel=ContentChannel.ONSITE_BANNER,
            request_fingerprint=FINGERPRINT,
            content=content,
        )


def test_source_request_fingerprint_is_stable_for_segment_id_order() -> None:
    first = prompt_input(segment_ids=["seg_b", "seg_a"])
    second = replace(
        first,
        request=first.request.model_copy(update={"segment_ids": ["seg_a", "seg_b"]}),
    )

    assert source_request_fingerprint(
        identity=IDENTITY,
        prompt_input=first,
        option_index=1,
    ) == source_request_fingerprint(
        identity=IDENTITY,
        prompt_input=second,
        option_index=1,
    )


def test_source_request_fingerprint_includes_selected_catalog_identity() -> None:
    base = prompt_input(segment_ids=["seg_a"])
    selected = replace(
        base,
        request=base.request.model_copy(
            update={"offer_set_id": "summer-lastcall"}
        ),
        offer_catalog={
            "catalog_id": "black-friday-hotels-lastcall",
            "catalog_version": "v3",
            "catalog_sha256": "b" * 64,
            "hotels": [],
        },
    )
    changed_catalog = replace(
        selected,
        offer_catalog={
            **selected.offer_catalog,
            "catalog_sha256": "c" * 64,
        },
    )

    assert source_request_fingerprint(
        identity=IDENTITY,
        prompt_input=selected,
        option_index=1,
    ) != source_request_fingerprint(
        identity=IDENTITY,
        prompt_input=changed_catalog,
        option_index=1,
    )


def test_source_manifest_key_uses_private_hierarchical_prefix() -> None:
    assert source_manifest_key(
        base_prefix="genai-source/",
        identity=IDENTITY,
    ) == (
        "genai-source/hotel-client-a/promo_banner_001/generation_banner_001/"
        "content_banner_repeat_hotel_001/source.json"
    )


def generated_content() -> GeneratedContent:
    return GeneratedContent(
        title="이번 주말 호텔 특가",
        body="예약 가능한 객실을 확인해보세요.",
        cta="호텔 보기",
        image_prompt="bright hotel room, no visible text",
        landing_url="https://demo-stay.example.com/summer",
    )


def prompt_input(*, segment_ids: list[str]) -> GenerationPromptInput:
    request = GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        segment_ids=segment_ids,
        content_option_count=1,
        operator_instruction=None,
    )
    return GenerationPromptInput(
        request=request,
        promotion=PromotionPromptInput(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            channel=ContentChannel.ONSITE_BANNER,
            goal_metric="booking_conversion_rate",
            goal_target_value="0.030000",
            goal_basis="all_segments",
            message_brief="Drive hotel booking conversion.",
            landing_url="https://demo-stay.example.com/summer",
        ),
        target_segment=TargetSegmentPromptInput(
            analysis_id=request.analysis_id,
            promotion_id=request.promotion_id,
            segment_id="seg_a",
            segment_name="Segment A",
            content_brief_json={"message_direction": "Highlight hotel stays."},
            segment_vector_id="segvec_a",
            estimated_size=100,
            priority="high",
        ),
    )
