from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

from psycopg.rows import dict_row

from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
)
from app.generation.schemas import ContentChannel
from app.logging import duration_ms, log, now_ms

if TYPE_CHECKING:
    from app.generation.prompt_builder import (
        GenerationContext,
        GenerationPromptInput,
    )


BRAND_CONTEXT_CONTRACT_VERSION = "generation-v1"
BRAND_CONTEXT_EMBEDDING_MODEL = "text-embedding-3-large"
BRAND_CONTEXT_EMBEDDING_VERSION = "text-embedding-3-large-1024-v1"
BRAND_CONTEXT_EMBEDDING_DIMENSIONS = 1024
BRAND_CONTEXT_RETRIEVAL_POLICY_VERSION = "exact-cosine-v1"
BRAND_CONTEXT_QUERY_VERSION = "decision-brand-query-v1"
BRAND_CONTEXT_PROMPT_VERSION = "generation-v1"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
MAX_RETRIEVAL_DOCUMENTS = 8
MAX_DOCUMENT_TEXT_LENGTH = 4_000

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HEX_COLOUR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}\b")


@dataclass(frozen=True, slots=True)
class BrandContextSnapshot:
    context_version: str
    manifest_key: str
    manifest_sha256: str
    guide_version: str
    asset_manifest_version: str
    catalog_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "context_version",
            "manifest_key",
            "guide_version",
            "asset_manifest_version",
            "catalog_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"brand context {field_name} is required")
        if not _SHA256_PATTERN.fullmatch(self.manifest_sha256):
            raise ValueError("brand context manifest_sha256 must be lowercase SHA-256")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{self.manifest_sha256}"

    def to_snapshot(self) -> dict[str, str]:
        return {
            "context_version": self.context_version,
            "manifest_key": self.manifest_key,
            "manifest_sha256": self.manifest_sha256,
            "guide_version": self.guide_version,
            "asset_manifest_version": self.asset_manifest_version,
            "catalog_version": self.catalog_version,
        }

    @classmethod
    def from_snapshot(cls, value: Mapping[str, Any]) -> BrandContextSnapshot:
        return cls(
            context_version=_required_text(value, "context_version"),
            manifest_key=_required_text(value, "manifest_key"),
            manifest_sha256=_required_text(value, "manifest_sha256"),
            guide_version=_required_text(value, "guide_version"),
            asset_manifest_version=_required_text(value, "asset_manifest_version"),
            catalog_version=_required_text(value, "catalog_version"),
        )


@dataclass(frozen=True, slots=True)
class RetrievedBrandDocument:
    document_id: str
    source_kind: str
    source_id: str
    source_version: str
    document_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    s3_key: str | None = None
    distance: float = 0.0

    def lineage(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "distance": round(float(self.distance), 8),
        }


@dataclass(frozen=True, slots=True)
class BrandGuardrails:
    forbidden_terms: tuple[str, ...] = ()
    forbidden_visual_terms: tuple[str, ...] = ()
    approved_colours: tuple[str, ...] = ()
    approved_image_styles: tuple[str, ...] = ()

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[RetrievedBrandDocument],
    ) -> BrandGuardrails:
        metadata = [document.metadata for document in documents]
        return cls(
            forbidden_terms=_unique_rules(
                metadata,
                ("forbidden_terms", "forbidden_phrases", "prohibited_terms"),
            ),
            forbidden_visual_terms=_unique_rules(
                metadata,
                (
                    "forbidden_visual_terms",
                    "forbidden_image_styles",
                    "prohibited_image_styles",
                ),
            ),
            approved_colours=_unique_rules(
                metadata,
                ("approved_colors", "allowed_colors", "brand_colors"),
            ),
            approved_image_styles=_unique_rules(
                metadata,
                ("approved_image_styles", "allowed_image_styles", "image_styles"),
            ),
        )


@dataclass(frozen=True, slots=True)
class RetrievedBrandContext:
    snapshot: BrandContextSnapshot
    documents: tuple[RetrievedBrandDocument, ...]
    query_sha256: str
    guardrails: BrandGuardrails
    selected_asset_id: str | None = None

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.query_sha256):
            raise ValueError("brand context query_sha256 must be lowercase SHA-256")
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("brand context documents must be unique")

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "context_version": self.snapshot.context_version,
            "context_fingerprint": self.snapshot.fingerprint,
            "documents": [
                {
                    "source_kind": document.source_kind,
                    "source_id": document.source_id,
                    "source_version": document.source_version,
                    "text": _bounded_document_text(document.document_text),
                    "rules": _prompt_rule_metadata(document.metadata),
                }
                for document in self.documents
                if document.document_text.strip() or document.metadata
            ],
        }

    def lineage(self, *, provider_request_id: str) -> dict[str, Any]:
        documents = [document.lineage() for document in self.documents]
        lineage: dict[str, Any] = {
            "context_version": self.snapshot.context_version,
            "context_fingerprint": self.snapshot.fingerprint,
            "retrieval_policy_version": BRAND_CONTEXT_RETRIEVAL_POLICY_VERSION,
            "query_version": BRAND_CONTEXT_QUERY_VERSION,
            "query_sha256": self.query_sha256,
            "document_ids": [document["document_id"] for document in documents],
            "documents": documents,
            "provider_request_id": provider_request_id,
        }
        if self.selected_asset_id is not None:
            lineage["selected_asset_id"] = self.selected_asset_id
        return lineage


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        ...


class BrandDocumentReader(Protocol):
    def retrieve(
        self,
        *,
        project_id: str,
        context_version: str,
        channel: ContentChannel,
        query_embedding: Sequence[float],
        limit: int = MAX_RETRIEVAL_DOCUMENTS,
    ) -> list[RetrievedBrandDocument]:
        ...


class BrandContextProvider(Protocol):
    def retrieve(
        self,
        prompt_input: GenerationPromptInput,
        generation_context: GenerationContext,
    ) -> RetrievedBrandContext | None:
        ...


class BrandContextRepository:
    """Read the Generation v1 RAG contract with strict tenant isolation."""

    SELECT_ACTIVE_CONTEXT_SQL = """
        WITH active_context AS (
            SELECT context_version
            FROM generation_rag.retrieval_documents
            WHERE project_id = %(project_id)s
              AND status = 'active'
              AND embedding_model = %(embedding_model)s
              AND embedding_version = %(embedding_version)s
            GROUP BY context_version
            ORDER BY max(updated_at) DESC, context_version DESC
            LIMIT 1
        )
        SELECT
            document_id,
            context_version,
            source_kind,
            source_id,
            source_version,
            s3_key,
            metadata_json,
            content_sha256
        FROM generation_rag.retrieval_documents
        WHERE project_id = %(project_id)s
          AND context_version = (SELECT context_version FROM active_context)
          AND status = 'active'
          AND embedding_model = %(embedding_model)s
          AND embedding_version = %(embedding_version)s
        ORDER BY source_kind, source_id, source_version, chunk_index, document_id
    """

    RETRIEVE_SQL = """
        SELECT
            document_id,
            source_kind,
            source_id,
            source_version,
            s3_key,
            document_text,
            metadata_json,
            embedding <=> %(query_embedding)s::vector AS distance
        FROM generation_rag.retrieval_documents
        WHERE project_id = %(project_id)s
          AND context_version = %(context_version)s
          AND status = 'active'
          AND embedding_model = %(embedding_model)s
          AND embedding_version = %(embedding_version)s
          AND COALESCE(metadata_json ->> 'active', 'true') <> 'false'
          AND COALESCE(metadata_json ->> 'approved', 'true') <> 'false'
          AND (
                metadata_json -> 'channels' IS NULL
                OR jsonb_typeof(metadata_json -> 'channels') <> 'array'
                OR metadata_json -> 'channels' ? %(channel)s
          )
          AND (
                %(channel)s <> 'sms'
                OR source_kind <> 'brand_asset'
          )
        ORDER BY
            CASE
                WHEN source_kind = 'brand_guide'
                     AND metadata_json ->> 'required' = 'true'
                THEN 0
                ELSE 1
            END,
            embedding <=> %(query_embedding)s::vector,
            document_id
        LIMIT %(limit)s
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def resolve_snapshot(self, *, project_id: str) -> BrandContextSnapshot | None:
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.SELECT_ACTIVE_CONTEXT_SQL,
                {
                    "project_id": project_id,
                    "embedding_model": BRAND_CONTEXT_EMBEDDING_MODEL,
                    "embedding_version": BRAND_CONTEXT_EMBEDDING_VERSION,
                },
            )
            rows = cursor.fetchall()
        if not rows:
            return None
        return _snapshot_from_rows(project_id=project_id, rows=rows)

    def retrieve(
        self,
        *,
        project_id: str,
        context_version: str,
        channel: ContentChannel,
        query_embedding: Sequence[float],
        limit: int = MAX_RETRIEVAL_DOCUMENTS,
    ) -> list[RetrievedBrandDocument]:
        vector = _pgvector(query_embedding)
        with self._connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                self.RETRIEVE_SQL,
                {
                    "project_id": project_id,
                    "context_version": context_version,
                    "channel": channel.value,
                    "embedding_model": BRAND_CONTEXT_EMBEDDING_MODEL,
                    "embedding_version": BRAND_CONTEXT_EMBEDDING_VERSION,
                    "query_embedding": vector,
                    "limit": limit,
                },
            )
            rows = cursor.fetchall()
        return [_retrieved_document(row) for row in rows]


class BrandContextRetrievalService:
    def __init__(
        self,
        *,
        repository: BrandDocumentReader,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._repository = repository
        self._embedding_client = embedding_client
        self._cache: dict[tuple[str, str, str, str], RetrievedBrandContext] = {}

    def retrieve(
        self,
        prompt_input: GenerationPromptInput,
        generation_context: GenerationContext,
    ) -> RetrievedBrandContext | None:
        snapshot = prompt_input.brand_context
        if snapshot is None:
            return None
        query = build_retrieval_query(prompt_input, generation_context)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        cache_key = (
            prompt_input.request.project_id,
            snapshot.context_version,
            prompt_input.promotion.channel.value,
            query_sha256,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._embedding_client.embed(query)
        documents = tuple(
            self._repository.retrieve(
                project_id=prompt_input.request.project_id,
                context_version=snapshot.context_version,
                channel=prompt_input.promotion.channel,
                query_embedding=embedding,
            )
        )
        if not any(document.source_kind == "brand_guide" for document in documents):
            raise PermanentGenerationError(
                code="brand_guide_unavailable",
                safe_message="The snapshotted brand guide is unavailable.",
            )
        selected_asset_id = _selected_asset_id(
            channel=prompt_input.promotion.channel,
            documents=documents,
        )
        if (
            prompt_input.promotion.channel != ContentChannel.SMS
            and selected_asset_id is None
        ):
            raise PermanentGenerationError(
                code="brand_asset_unavailable",
                safe_message="The snapshotted approved brand asset is unavailable.",
            )
        result = RetrievedBrandContext(
            snapshot=snapshot,
            documents=documents,
            query_sha256=query_sha256,
            guardrails=BrandGuardrails.from_documents(documents),
            selected_asset_id=selected_asset_id,
        )
        self._cache[cache_key] = result
        return result


class ManagedBrandContextProvider:
    """Open a short DB connection only after the embedding request completes."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any],
        embedding_client: EmbeddingClient,
    ) -> None:
        self._connection_factory = connection_factory
        self._embedding_client = embedding_client
        self._cache: dict[tuple[str, str, str, str], RetrievedBrandContext] = {}

    def retrieve(
        self,
        prompt_input: GenerationPromptInput,
        generation_context: GenerationContext,
    ) -> RetrievedBrandContext | None:
        if prompt_input.brand_context is None:
            return None
        query = build_retrieval_query(prompt_input, generation_context)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        cache_key = (
            prompt_input.request.project_id,
            prompt_input.brand_context.context_version,
            prompt_input.promotion.channel.value,
            query_sha256,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._embedding_client.embed(query)
        connection = self._connection_factory()
        try:
            documents = tuple(
                BrandContextRepository(connection).retrieve(
                    project_id=prompt_input.request.project_id,
                    context_version=prompt_input.brand_context.context_version,
                    channel=prompt_input.promotion.channel,
                    query_embedding=embedding,
                )
            )
        finally:
            connection.close()
        if not any(document.source_kind == "brand_guide" for document in documents):
            raise PermanentGenerationError(
                code="brand_guide_unavailable",
                safe_message="The snapshotted brand guide is unavailable.",
            )
        selected_asset_id = _selected_asset_id(
            channel=prompt_input.promotion.channel,
            documents=documents,
        )
        if (
            prompt_input.promotion.channel != ContentChannel.SMS
            and selected_asset_id is None
        ):
            raise PermanentGenerationError(
                code="brand_asset_unavailable",
                safe_message="The snapshotted approved brand asset is unavailable.",
            )
        result = RetrievedBrandContext(
            snapshot=prompt_input.brand_context,
            documents=documents,
            query_sha256=query_sha256,
            guardrails=BrandGuardrails.from_documents(documents),
            selected_asset_id=selected_asset_id,
        )
        self._cache[cache_key] = result
        return result


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = OPENAI_EMBEDDINGS_URL,
        timeout_seconds: float = 30.0,
        transport: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._transport = transport or _post_embedding_json

    def embed(self, text: str) -> Sequence[float]:
        payload = {
            "model": BRAND_CONTEXT_EMBEDDING_MODEL,
            "input": text,
            "dimensions": BRAND_CONTEXT_EMBEDDING_DIMENSIONS,
            "encoding_format": "float",
        }
        started_at = now_ms()
        log.info(
            "provider_request_prepared",
            {
                "provider": "openai",
                "endpoint": self._endpoint,
                "model": BRAND_CONTEXT_EMBEDDING_MODEL,
                "purpose": "brand_context_retrieval",
            },
        )
        response = self._transport(
            self._endpoint,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self._timeout_seconds,
        )
        try:
            data = response["data"]
            embedding = data[0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentGenerationError(
                code="brand_embedding_response_invalid",
                safe_message="The brand retrieval embedding response was invalid.",
            ) from exc
        vector = _validated_embedding(embedding)
        log.info(
            "provider_request_completed",
            {
                "provider": "openai",
                "endpoint": self._endpoint,
                "model": BRAND_CONTEXT_EMBEDDING_MODEL,
                "purpose": "brand_context_retrieval",
                "durationMs": duration_ms(started_at),
            },
        )
        return vector


def build_retrieval_query(
    prompt_input: GenerationPromptInput,
    generation_context: GenerationContext,
) -> str:
    del generation_context
    payload = {
        "query_version": BRAND_CONTEXT_QUERY_VERSION,
        "channel": prompt_input.promotion.channel.value,
        "promotion": {
            "goal_metric": prompt_input.promotion.goal_metric,
            "goal_basis": prompt_input.promotion.goal_basis,
            "goal_target_value": prompt_input.promotion.goal_target_value,
            "message_brief": prompt_input.promotion.message_brief,
            "offer_type": prompt_input.promotion.offer_type,
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def generation_versions(*, model_version: str) -> dict[str, str]:
    return {
        "content_spec": BRAND_CONTEXT_CONTRACT_VERSION,
        "embedding": BRAND_CONTEXT_EMBEDDING_VERSION,
        "guardrail": BRAND_CONTEXT_CONTRACT_VERSION,
        "model": model_version,
        "prompt": BRAND_CONTEXT_PROMPT_VERSION,
        "renderer": BRAND_CONTEXT_CONTRACT_VERSION,
        "retrieval_policy": BRAND_CONTEXT_RETRIEVAL_POLICY_VERSION,
    }


def retrieval_snapshot_from_candidate_metadata(
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    for candidate in metadata:
        creative = candidate.get("creative")
        if not isinstance(creative, Mapping):
            continue
        lineage = creative.get("lineage")
        if not isinstance(lineage, Mapping):
            continue
        raw_documents = lineage.get("documents")
        if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, str):
            continue
        for raw_document in raw_documents:
            if not isinstance(raw_document, Mapping):
                continue
            document_id = str(raw_document.get("document_id") or "").strip()
            if not document_id:
                continue
            document = {
                "document_id": document_id,
                "source_kind": str(raw_document.get("source_kind") or ""),
                "source_id": str(raw_document.get("source_id") or ""),
                "source_version": str(raw_document.get("source_version") or ""),
                "distance": float(raw_document.get("distance") or 0.0),
            }
            previous = documents.get(document_id)
            if previous is None or document["distance"] < previous["distance"]:
                documents[document_id] = document
    return {
        "query_version": BRAND_CONTEXT_QUERY_VERSION,
        "retrieval_policy_version": BRAND_CONTEXT_RETRIEVAL_POLICY_VERSION,
        "embedding_version": BRAND_CONTEXT_EMBEDDING_VERSION,
        "documents": [documents[key] for key in sorted(documents)],
    }


def validate_brand_guardrails(
    context: RetrievedBrandContext | None,
    *,
    content_values: Mapping[str, Any],
) -> None:
    if context is None:
        return
    text = "\n".join(
        str(content_values.get(field_name) or "")
        for field_name in ("subject", "preheader", "title", "body", "cta", "message")
    ).casefold()
    if any(term.casefold() in text for term in context.guardrails.forbidden_terms):
        raise PermanentGenerationError(
            code="brand_text_guardrail_violation",
            safe_message="Generated copy violated the snapshotted brand rules.",
        )

    image_prompt = str(content_values.get("image_prompt") or "")
    image_prompt_folded = image_prompt.casefold()
    if any(
        term.casefold() in image_prompt_folded
        for term in context.guardrails.forbidden_visual_terms
    ):
        raise PermanentGenerationError(
            code="brand_visual_guardrail_violation",
            safe_message="Generated visual direction violated the snapshotted brand rules.",
        )
    approved_hex = {
        value.casefold()
        for value in context.guardrails.approved_colours
        if _HEX_COLOUR_PATTERN.fullmatch(value)
    }
    generated_hex = {
        value.casefold() for value in _HEX_COLOUR_PATTERN.findall(image_prompt)
    }
    if approved_hex and not generated_hex.issubset(approved_hex):
        raise PermanentGenerationError(
            code="brand_colour_guardrail_violation",
            safe_message="Generated visual colours violated the snapshotted brand rules.",
        )


def _snapshot_from_rows(
    *,
    project_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> BrandContextSnapshot:
    context_versions = {str(row.get("context_version") or "") for row in rows}
    if len(context_versions) != 1 or not next(iter(context_versions), ""):
        raise ValueError("active brand context rows must have one context_version")
    context_version = next(iter(context_versions))
    metadata = [
        row.get("metadata_json")
        for row in rows
        if isinstance(row.get("metadata_json"), Mapping)
    ]
    manifest_key = _consistent_metadata_text(metadata, "manifest_key") or (
        f"brand-context/{project_id}/manifests/{context_version}/manifest.json"
    )
    manifest_sha256 = _consistent_metadata_text(metadata, "manifest_sha256")
    if manifest_sha256 is None or not _SHA256_PATTERN.fullmatch(manifest_sha256):
        manifest_sha256 = _rows_fingerprint(rows)
    guide_version = _consistent_metadata_text(metadata, "guide_version") or (
        _source_version(rows, "brand_guide") or context_version
    )
    asset_manifest_version = _consistent_metadata_text(
        metadata,
        "asset_manifest_version",
    ) or (_source_version(rows, "brand_asset") or context_version)
    catalog_version = (
        _consistent_metadata_text(metadata, "catalog_version") or context_version
    )
    return BrandContextSnapshot(
        context_version=context_version,
        manifest_key=manifest_key,
        manifest_sha256=manifest_sha256,
        guide_version=guide_version,
        asset_manifest_version=asset_manifest_version,
        catalog_version=catalog_version,
    )


def _rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "document_id": str(row.get("document_id") or ""),
            "source_kind": str(row.get("source_kind") or ""),
            "source_id": str(row.get("source_id") or ""),
            "source_version": str(row.get("source_version") or ""),
            "content_sha256": str(row.get("content_sha256") or ""),
        }
        for row in rows
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_version(
    rows: Sequence[Mapping[str, Any]],
    source_kind: str,
) -> str | None:
    versions = sorted(
        {
            str(row.get("source_version") or "").strip()
            for row in rows
            if row.get("source_kind") == source_kind
            and str(row.get("source_version") or "").strip()
        }
    )
    return versions[0] if len(versions) == 1 else None


def _consistent_metadata_text(
    metadata: Sequence[Mapping[str, Any]],
    key: str,
) -> str | None:
    values = {
        str(item.get(key) or "").strip()
        for item in metadata
        if str(item.get(key) or "").strip()
    }
    return next(iter(values)) if len(values) == 1 else None


def _retrieved_document(row: Mapping[str, Any]) -> RetrievedBrandDocument:
    metadata = row.get("metadata_json")
    try:
        distance = float(row.get("distance") or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("brand retrieval distance must be numeric") from exc
    return RetrievedBrandDocument(
        document_id=_required_text(row, "document_id"),
        source_kind=_required_text(row, "source_kind"),
        source_id=_required_text(row, "source_id"),
        source_version=_required_text(row, "source_version"),
        document_text=str(row.get("document_text") or "").strip(),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        s3_key=_optional_text(row.get("s3_key")),
        distance=distance,
    )


def _selected_asset_id(
    *,
    channel: ContentChannel,
    documents: Sequence[RetrievedBrandDocument],
) -> str | None:
    if channel == ContentChannel.SMS:
        return None
    for document in documents:
        if document.source_kind == "brand_asset":
            return document.source_id
    return None


def _pgvector(values: Sequence[float]) -> str:
    vector = _validated_embedding(values)
    return "[" + ",".join(format(value, ".12g") for value in vector) + "]"


def _validated_embedding(value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("brand retrieval embedding must be an array")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("brand retrieval embedding must be numeric") from exc
    if len(vector) != BRAND_CONTEXT_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"brand retrieval embedding must have "
            f"{BRAND_CONTEXT_EMBEDDING_DIMENSIONS} dimensions"
        )
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("brand retrieval embedding must contain finite values")
    return vector


def _unique_rules(
    metadata: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for item in metadata:
        sources = [item]
        for container_key in ("rules", "guardrails", "design_rules"):
            nested = item.get(container_key)
            if isinstance(nested, Mapping):
                sources.append(nested)
        for source in sources:
            for key in keys:
                raw = source.get(key)
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                    values.extend(str(value).strip() for value in raw)
    return tuple(dict.fromkeys(value for value in values if value))


def _prompt_rule_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "required",
        "channels",
        "role",
        "locale",
        "brand_description",
        "core_messages",
        "tone_voice",
        "tone_and_voice",
        "forbidden_terms",
        "forbidden_phrases",
        "prohibited_terms",
        "approved_colors",
        "allowed_colors",
        "brand_colors",
        "approved_image_styles",
        "allowed_image_styles",
        "image_styles",
        "forbidden_visual_terms",
        "forbidden_image_styles",
        "channel_constraints",
        "rules",
        "guardrails",
        "design_rules",
    }
    return {key: item for key, item in value.items() if key in allowed_keys}


def _bounded_document_text(value: str) -> str:
    compacted = " ".join(value.split())
    return compacted[:MAX_DOCUMENT_TEXT_LENGTH]


def _required_text(value: Mapping[str, Any], key: str) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise ValueError(f"brand context {key} is required")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _post_embedding_json(
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429 or 500 <= exc.code <= 599:
            raise RetryableGenerationError(
                code="brand_embedding_provider_unavailable",
                safe_message="The brand retrieval embedding provider is unavailable.",
                status_code=exc.code,
            ) from exc
        raise PermanentGenerationError(
            code="brand_embedding_request_rejected",
            safe_message="The brand retrieval embedding request was rejected.",
            status_code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise RetryableGenerationError(
            code="brand_embedding_network_error",
            safe_message="The brand retrieval embedding request failed temporarily.",
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermanentGenerationError(
            code="brand_embedding_response_invalid",
            safe_message="The brand retrieval embedding response was invalid.",
        ) from exc
    if not isinstance(parsed, Mapping):
        raise PermanentGenerationError(
            code="brand_embedding_response_invalid",
            safe_message="The brand retrieval embedding response was invalid.",
        )
    return parsed
