from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from structlog.testing import capture_logs

from app.generation.artifacts import ArtifactIdentity
from app.generation.brand_context import (
    BRAND_CONTEXT_EMBEDDING_MODEL,
    BRAND_CONTEXT_EMBEDDING_VERSION,
    BRAND_CONTEXT_QUERY_VERSION,
    BRAND_CONTEXT_RETRIEVAL_POLICY_VERSION,
    BrandContextRepository,
    BrandContextRetrievalService,
    BrandContextSnapshot,
    BrandGuardrails,
    OpenAIEmbeddingClient,
    RetrievedBrandContext,
    RetrievedBrandDocument,
    validate_brand_guardrails,
)
from app.generation.errors import PermanentGenerationError
from app.generation.generator import GeneratedContent
from app.generation.prompt_builder import (
    GenerationContextBuilder,
    GenerationInputBuilder,
    GenerationPromptInput,
    PromptBuilder,
    PromotionPromptInput,
    TargetSegmentPromptInput,
)
from app.generation.schemas import ContentChannel, GenerationRequest
from app.generation.service import GenerationService
from app.generation.submission import (
    build_generation_input_snapshot,
    prompt_inputs_from_snapshot,
)


class FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.query = ""
        self.params: dict[str, Any] = {}

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, query: str, params: dict[str, Any]) -> None:
        self.query = query
        self.params = params

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.cursor_value = FakeCursor(rows)

    def cursor(self, *, row_factory: object = None) -> FakeCursor:
        del row_factory
        return self.cursor_value


class StaticBrandContextProvider:
    def __init__(self, context: RetrievedBrandContext) -> None:
        self.context = context

    def retrieve(self, *_: object) -> RetrievedBrandContext:
        return self.context


class ForbiddenSmsGenerator:
    version = "test-forbidden.v1"

    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: object,
        option_index: int,
        artifact_identity: ArtifactIdentity,
    ) -> GeneratedContent:
        del prompt_result, option_index, artifact_identity
        return GeneratedContent(
            message="과장광고 문구 {{redirect_url}}",
            landing_url=prompt_input.promotion.landing_url,
        )


class CountingEmbeddingClient:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        assert "promo_demo" not in text
        self.calls += 1
        return [0.001] * 1024


class CountingDocumentReader:
    def __init__(self, documents: list[RetrievedBrandDocument]) -> None:
        self.documents = documents
        self.calls = 0

    def retrieve(self, **_: object) -> list[RetrievedBrandDocument]:
        self.calls += 1
        return self.documents


class CountingDocumentLoader:
    def __init__(self, documents: tuple[RetrievedBrandDocument, ...]) -> None:
        self.documents = documents
        self.calls = 0

    def load_documents(self, **_: object) -> tuple[RetrievedBrandDocument, ...]:
        self.calls += 1
        return self.documents


def snapshot() -> BrandContextSnapshot:
    return BrandContextSnapshot(
        context_version="v1",
        manifest_key="brand-context/demo_project/manifests/v1/manifest.json",
        manifest_sha256="a" * 64,
        guide_version="v1",
        asset_manifest_version="v1",
        catalog_version="fixture-catalog-v1",
    )


def request() -> GenerationRequest:
    return GenerationRequest(
        project_id="demo_project",
        campaign_id="camp_demo",
        promotion_id="promo_demo",
        analysis_id="analysis_demo",
        content_option_count=1,
    )


def promotion(
    channel: ContentChannel = ContentChannel.ONSITE_BANNER,
) -> PromotionPromptInput:
    return PromotionPromptInput(
        project_id="demo_project",
        campaign_id="camp_demo",
        promotion_id="promo_demo",
        channel=channel,
        goal_metric="booking_conversion_rate",
        goal_target_value="0.03",
        goal_basis="all_segments",
        message_brief="예약 결정을 명확하게 돕습니다.",
        landing_url="https://demo.example.test/hotel",
        offer_type="hotel_deal",
        landing_type="hotel_detail_page",
    )


def target() -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id="analysis_demo",
        promotion_id="promo_demo",
        segment_id="seg_demo",
        segment_name="예약 전환 고객",
        content_brief_json={
            "message_direction": "예약 정보를 명확하게 안내합니다.",
            "keywords": ["호텔", "예약"],
        },
        segment_vector_id="segvec_demo",
        estimated_size=120,
        priority="high",
    )


def prompt_input(
    channel: ContentChannel = ContentChannel.ONSITE_BANNER,
) -> GenerationPromptInput:
    return GenerationInputBuilder().build(
        request=request(),
        promotion=promotion(channel),
        target_segments=[target()],
        brand_context=snapshot(),
    )[0]


def retrieved_context(
    channel: ContentChannel = ContentChannel.ONSITE_BANNER,
    *,
    forbidden_terms: tuple[str, ...] = (),
) -> RetrievedBrandContext:
    documents = [
        RetrievedBrandDocument(
            document_id="rag_demo_brand_voice_v1_0",
            source_kind="brand_guide",
            source_id="brand-voice",
            source_version="v1",
            document_text=(
                "StayLoop uses calm, clear, and trustworthy travel language."
            ),
            metadata={
                "required": True,
                "channels": ["email", "onsite_banner", "sms"],
                "forbidden_terms": list(forbidden_terms),
                "approved_colors": ["#123456"],
                "approved_image_styles": ["calm editorial hotel photography"],
            },
            distance=0.021,
        )
    ]
    selected_asset_id = None
    if channel != ContentChannel.SMS:
        documents.append(
            RetrievedBrandDocument(
                document_id="rag_demo_home_hero_v1_0",
                source_kind="brand_asset",
                source_id="home-hero",
                source_version="v1",
                document_text="StayLoop representative hotel hero asset.",
                metadata={"role": "hero", "active": True},
                distance=0.034,
            )
        )
        selected_asset_id = "home-hero"
    return RetrievedBrandContext(
        snapshot=snapshot(),
        documents=tuple(documents),
        query_sha256="b" * 64,
        guardrails=BrandGuardrails.from_documents(documents),
        selected_asset_id=selected_asset_id,
    )


def test_repository_resolves_contract_snapshot_with_project_filter() -> None:
    rows = [
        {
            "document_id": "rag_demo_brand_voice_v1_0",
            "context_version": "v1",
            "source_kind": "brand_guide",
            "source_id": "brand-voice",
            "source_version": "v1",
            "s3_key": "brand-context/demo_project/guidelines/brand-voice/v1/content.md",
            "metadata_json": {
                "manifest_key": (
                    "brand-context/demo_project/manifests/v1/manifest.json"
                ),
                "manifest_sha256": "c" * 64,
                "catalog_version": "fixture-catalog-v1",
            },
            "content_sha256": "d" * 64,
        },
        {
            "document_id": "rag_demo_home_hero_v1_0",
            "context_version": "v1",
            "source_kind": "brand_asset",
            "source_id": "home-hero",
            "source_version": "v1",
            "s3_key": "brand-context/demo_project/assets/home-hero/v1/original.jpg",
            "metadata_json": {
                "manifest_key": (
                    "brand-context/demo_project/manifests/v1/manifest.json"
                ),
                "manifest_sha256": "c" * 64,
                "catalog_version": "fixture-catalog-v1",
            },
            "content_sha256": "e" * 64,
        },
    ]
    connection = FakeConnection(rows)

    resolved = BrandContextRepository(connection).resolve_snapshot(
        project_id="demo_project"
    )

    assert resolved == replace(snapshot(), manifest_sha256="c" * 64)
    assert "WHERE project_id = %(project_id)s" in connection.cursor_value.query
    assert connection.cursor_value.params == {
        "project_id": "demo_project",
        "embedding_model": BRAND_CONTEXT_EMBEDDING_MODEL,
        "embedding_version": BRAND_CONTEXT_EMBEDDING_VERSION,
    }


def test_repository_retrieval_hard_filters_project_context_and_channel() -> None:
    connection = FakeConnection(
        [
            {
                "document_id": "rag_demo_brand_voice_v1_0",
                "source_kind": "brand_guide",
                "source_id": "brand-voice",
                "source_version": "v1",
                "s3_key": None,
                "document_text": "Calm and clear.",
                "metadata_json": {"required": True},
                "distance": 0.012,
            }
        ]
    )

    documents = BrandContextRepository(connection).retrieve(
        project_id="demo_project",
        context_version="v1",
        channel=ContentChannel.SMS,
        query_embedding=[0.001] * 1024,
    )

    assert [document.document_id for document in documents] == [
        "rag_demo_brand_voice_v1_0"
    ]
    query = connection.cursor_value.query
    assert "WHERE project_id = %(project_id)s" in query
    assert "context_version = %(context_version)s" in query
    assert "status = 'active'" in query
    assert "source_kind <> 'brand_asset'" in query
    assert connection.cursor_value.params["project_id"] == "demo_project"
    assert connection.cursor_value.params["channel"] == "sms"


def test_embedding_client_uses_contract_model_and_dimensions() -> None:
    captured: dict[str, Any] = {}

    def transport(
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        captured.update(
            {
                "endpoint": endpoint,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"data": [{"embedding": [0.001] * 1024}]}

    with capture_logs() as logs:
        result = OpenAIEmbeddingClient(
            api_key="test-key",
            timeout_seconds=7,
            transport=transport,
        ).embed("brand query")

    assert len(result) == 1024
    assert captured["payload"] == {
        "model": BRAND_CONTEXT_EMBEDDING_MODEL,
        "input": "brand query",
        "dimensions": 1024,
        "encoding_format": "float",
    }
    assert captured["timeout_seconds"] == 7
    assert [
        record["event"]
        for record in logs
        if str(record["event"]).startswith("provider_request_")
    ] == ["provider_request_prepared", "provider_request_completed"]
    assert "test-key" not in str(logs)


def test_brand_snapshot_roundtrip_is_immutable_generation_input() -> None:
    confirmed_target = replace(
        target(),
        source_content_brief_json={"schema_version": "content_brief.v2"},
        data_evidence_json={"source": "confirmed_fixture"},
    )
    generation_snapshot = build_generation_input_snapshot(
        request=request(),
        promotion=promotion(),
        target_segments=[confirmed_target],
        brand_context=snapshot(),
        model_version="gpt-test",
    )

    assert generation_snapshot["brand_context"] == snapshot().to_snapshot()
    assert generation_snapshot["placement"] == {
        "type": "onsite_banner",
        "slot_id": "C1_MAIN_TOP",
    }
    assert generation_snapshot["offer"]["type"] == "hotel_deal"
    assert generation_snapshot["landing"]["type"] == "hotel_detail_page"
    assert generation_snapshot["target_segments"][0]["content_brief"] == {
        "schema_version": "content_brief.v2"
    }
    assert generation_snapshot["target_segments"][0]["data_evidence"] == {
        "source": "confirmed_fixture"
    }
    assert generation_snapshot["versions"]["embedding"] == (
        BRAND_CONTEXT_EMBEDDING_VERSION
    )
    assert generation_snapshot["versions"]["retrieval_policy"] == (
        BRAND_CONTEXT_RETRIEVAL_POLICY_VERSION
    )
    assert prompt_inputs_from_snapshot(generation_snapshot) == [prompt_input()]


def test_prompt_injects_brand_context_without_exposing_storage_keys() -> None:
    value = prompt_input()
    context = replace(
        GenerationContextBuilder().build(value),
        brand_context=retrieved_context(),
    )

    result = PromptBuilder().build(value, generation_context=context)

    assert "StayLoop uses calm, clear" in result.generation_prompt
    assert "approved_colors" in result.generation_prompt
    assert "Never reveal" in result.generation_prompt
    assert "brand-context/demo_project" not in result.generation_prompt
    assert result.metadata_json["brand_context_fingerprint"] == "sha256:" + "a" * 64


def test_retrieval_is_cached_per_promotion_snapshot_for_consistent_lineage() -> None:
    first = prompt_input()
    second = replace(
        first,
        target_segment=replace(
            first.target_segment,
            segment_id="seg_other",
            segment_name="다른 고객군",
        ),
    )
    expected = retrieved_context()
    embedding_client = CountingEmbeddingClient()
    repository = CountingDocumentReader(list(expected.documents))
    provider = BrandContextRetrievalService(
        repository=repository,
        embedding_client=embedding_client,
    )

    first_result = provider.retrieve(
        first,
        GenerationContextBuilder().build(first),
    )
    second_result = provider.retrieve(
        second,
        GenerationContextBuilder().build(second),
    )

    assert first_result is second_result
    assert embedding_client.calls == 1
    assert repository.calls == 1


def test_retrieval_uses_verified_s3_context_when_rag_index_is_empty() -> None:
    value = prompt_input()
    expected = retrieved_context()
    embedding_client = CountingEmbeddingClient()
    repository = CountingDocumentReader([])
    source_loader = CountingDocumentLoader(expected.documents)
    provider = BrandContextRetrievalService(
        repository=repository,
        embedding_client=embedding_client,
        source_loader=source_loader,
    )

    result = provider.retrieve(
        value,
        GenerationContextBuilder().build(value),
    )

    assert result is not None
    assert result.documents == expected.documents
    assert result.selected_asset_id == "home-hero"
    assert result.guardrails.approved_colours == ("#123456",)
    assert embedding_client.calls == 1
    assert repository.calls == 1
    assert source_loader.calls == 1


def test_generation_persists_contract_lineage_and_private_text_stays_internal() -> None:
    value = prompt_input()
    result = GenerationService(
        brand_context_provider=StaticBrandContextProvider(retrieved_context()),
    ).execute_durable(
        generation_id="generation_demo_001",
        prompt_inputs=[value],
    )

    candidate = result.content_candidates[0]
    lineage = candidate.metadata_json["creative"]["lineage"]
    assert lineage["context_version"] == "v1"
    assert lineage["selected_asset_id"] == "home-hero"
    assert lineage["document_ids"] == [
        "rag_demo_brand_voice_v1_0",
        "rag_demo_home_hero_v1_0",
    ]
    assert lineage["provider_request_id"].startswith("loopad:generation_demo_001:")
    retrieval = result.output_json["retrieval_snapshot"]
    assert retrieval["query_version"] == BRAND_CONTEXT_QUERY_VERSION
    assert retrieval["documents"] == lineage["documents"]
    public_json = json.dumps(candidate.to_public_values(), default=str)
    assert "StayLoop uses calm" not in public_json
    assert "rag_demo_brand_voice" not in public_json


def test_brand_rule_violation_fails_closed_before_candidate_checkpoint() -> None:
    value = prompt_input(ContentChannel.SMS)
    context = retrieved_context(
        ContentChannel.SMS,
        forbidden_terms=("과장광고",),
    )

    with pytest.raises(
        PermanentGenerationError,
        match="snapshotted brand rules",
    ):
        GenerationService(
            brand_context_provider=StaticBrandContextProvider(context),
            content_generator=ForbiddenSmsGenerator(),
        ).execute_durable(
            generation_id="generation_sms_001",
            prompt_inputs=[value],
        )


def test_guardrail_rejects_unapproved_hex_colour() -> None:
    with pytest.raises(PermanentGenerationError) as exc_info:
        validate_brand_guardrails(
            retrieved_context(),
            content_values={"image_prompt": "Use a #ffffff background."},
        )

    assert exc_info.value.code == "brand_colour_guardrail_violation"
