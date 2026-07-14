from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from app.generation.brand_context import (
    BrandContextSnapshot,
    generation_versions,
)
from app.generation.prompt_builder import (
    GenerationInputBuilder,
    GenerationPromptInput,
    PromotionPromptInput,
    TargetSegmentPromptInput,
)
from app.generation.repositories import (
    GenerationIdempotencyMismatch,
    GenerationRunRecord,
)
from app.generation.schemas import (
    ContentChannel,
    GenerationAcceptedResponse,
    GenerationRequest,
    GenerationStatus,
)
from app.logging import log


GENERATION_REQUEST_SCHEMA_VERSION = "generation.request.v1"
MAX_IDEMPOTENCY_KEY_LENGTH = 200
MAX_GENERATION_ID_LENGTH = 100
INTERNAL_IDEMPOTENCY_KEY_PREFIX = "loopad-internal:"


class GenerationSubmissionRepository(Protocol):
    def create_or_get_idempotent(
        self,
        record: GenerationRunRecord,
    ) -> tuple[dict[str, Any], bool]:
        ...


class GenerationSubmissionInputReader(Protocol):
    def get_promotion_input(
        self,
        request: GenerationRequest,
    ) -> PromotionPromptInput | None:
        ...

    def list_target_segment_inputs(
        self,
        request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        ...


class BrandContextSnapshotReader(Protocol):
    def resolve_snapshot(self, *, project_id: str) -> BrandContextSnapshot | None:
        ...


class GenerationWakeCoordinator(Protocol):
    @property
    def accepting(self) -> bool:
        ...

    def wake(self) -> None:
        ...


class GenerationSubmissionConnection(Protocol):
    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...


class GenerationInputUnavailable(RuntimeError):
    """Raised when confirmed recommendation inputs are not ready."""


class GenerationIdempotencyConflict(RuntimeError):
    """Raised when one idempotency key is reused with a different request."""


class GenerationSubmissionUnavailable(RuntimeError):
    """Raised while the application is shutting down and no longer accepts jobs."""


class GenerationSnapshotError(ValueError):
    """Raised when a durable Generation input snapshot is malformed."""


class GenerationSubmissionService:
    def __init__(
        self,
        *,
        connection: GenerationSubmissionConnection,
        generation_run_repository: GenerationSubmissionRepository,
        generation_input_reader: GenerationSubmissionInputReader,
        brand_context_repository: BrandContextSnapshotReader | None = None,
        model_version: str = "generation-default",
        coordinator: GenerationWakeCoordinator | None = None,
    ) -> None:
        self._connection = connection
        self._generation_run_repository = generation_run_repository
        self._generation_input_reader = generation_input_reader
        self._brand_context_repository = brand_context_repository
        self._model_version = model_version
        self._coordinator = coordinator

    def submit(
        self,
        request: GenerationRequest,
        *,
        idempotency_key: str,
    ) -> GenerationAcceptedResponse:
        key = normalize_idempotency_key(idempotency_key)
        if self._coordinator is not None and not self._coordinator.accepting:
            raise GenerationSubmissionUnavailable(
                "generation worker is shutting down"
            )

        brand_context = (
            self._brand_context_repository.resolve_snapshot(
                project_id=request.project_id,
            )
            if self._brand_context_repository is not None
            else None
        )
        promotion = self._generation_input_reader.get_promotion_input(request)
        if promotion is None:
            raise GenerationInputUnavailable(
                "promotion input was not found for generation"
            )
        target_segments = self._generation_input_reader.list_target_segment_inputs(
            request
        )
        if not target_segments:
            raise GenerationInputUnavailable(
                "confirmed promotion_target_segments are required for generation"
            )
        snapshot = build_generation_input_snapshot(
            request=request,
            promotion=promotion,
            target_segments=target_segments,
            brand_context=brand_context,
            model_version=self._model_version,
        )
        fingerprint = generation_request_fingerprint(snapshot)
        generation_id = generation_id_for_request(
            promotion_id=request.promotion_id,
            project_id=request.project_id,
            idempotency_key=key,
        )
        record = GenerationRunRecord(
            generation_id=generation_id,
            analysis_id=request.analysis_id,
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            content_option_count=request.content_option_count,
            operator_instruction=request.operator_instruction,
            input_json=snapshot,
            output_json=None,
            generation_report_json={
                "status": GenerationStatus.REQUESTED.value,
                "schema_version": GENERATION_REQUEST_SCHEMA_VERSION,
            },
            status=GenerationStatus.REQUESTED.value,
            idempotency_key=key,
            request_fingerprint=fingerprint,
        )

        try:
            persisted, created = self._generation_run_repository.create_or_get_idempotent(
                record
            )
            persisted_fingerprint = str(
                persisted.get("request_fingerprint") or ""
            )
            if persisted_fingerprint != fingerprint:
                raise GenerationIdempotencyConflict(
                    "idempotency key was already used for a different generation request"
                )
            self._connection.commit()
        except GenerationIdempotencyMismatch as exc:
            self._connection.rollback()
            raise GenerationIdempotencyConflict(
                "idempotency key was already used for a different generation request"
            ) from exc
        except Exception:
            self._connection.rollback()
            raise

        if self._coordinator is not None:
            try:
                self._coordinator.wake()
            except Exception as exc:
                # The durable requested row is the source of truth. Periodic polling
                # must recover a lost in-process wake-up signal.
                log.warn(
                    "generation_wakeup_failed",
                    {"generationId": persisted.get("generation_id"), "err": exc},
                )

        status = GenerationStatus(str(persisted["status"]))
        response = GenerationAcceptedResponse(
            generation_id=str(persisted["generation_id"]),
            promotion_id=str(persisted["promotion_id"]),
            status=status,
        )
        log.info(
            "generation_request_accepted",
            {
                "generationId": response.generation_id,
                "status": response.status.value,
                "created": created,
            },
        )
        return response


def normalize_idempotency_key(value: str) -> str:
    key = str(value).strip()
    if not key:
        raise ValueError("Idempotency-Key header is required")
    if key.startswith(INTERNAL_IDEMPOTENCY_KEY_PREFIX):
        raise ValueError("Idempotency-Key uses a reserved internal prefix")
    if len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise ValueError(
            f"Idempotency-Key must be at most {MAX_IDEMPOTENCY_KEY_LENGTH} characters"
        )
    return key


def build_generation_input_snapshot(
    *,
    request: GenerationRequest,
    promotion: PromotionPromptInput,
    target_segments: Sequence[TargetSegmentPromptInput],
    brand_context: BrandContextSnapshot | None = None,
    model_version: str = "generation-default",
) -> dict[str, Any]:
    GenerationInputBuilder().build(
        request=request,
        promotion=promotion,
        target_segments=target_segments,
        brand_context=brand_context,
    )
    targets = sorted(target_segments, key=lambda item: item.segment_id)
    if len({target.segment_id for target in targets}) != len(targets):
        raise GenerationSnapshotError("target_segments must not contain duplicates")
    snapshot: dict[str, Any] = {
        "schema_version": GENERATION_REQUEST_SCHEMA_VERSION,
        "project_id": request.project_id,
        "campaign_id": request.campaign_id,
        "promotion_id": request.promotion_id,
        "analysis_id": request.analysis_id,
        "content_option_count": request.content_option_count,
        "operator_instruction": request.operator_instruction,
        "channel": promotion.channel.value,
        "promotion": _promotion_snapshot(promotion),
        "target_segments": [_target_segment_snapshot(item) for item in targets],
        "placement": _placement_snapshot(promotion.channel),
        "offer": {
            "type": promotion.offer_type,
            "message_brief": promotion.message_brief,
        },
        "landing": {
            "url": promotion.landing_url,
            "type": promotion.landing_type,
        },
        "versions": generation_versions(model_version=model_version),
    }
    if brand_context is not None:
        snapshot["brand_context"] = brand_context.to_snapshot()
    return snapshot


def generation_request_fingerprint(snapshot: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generation_id_for_request(
    *,
    promotion_id: str,
    project_id: str,
    idempotency_key: str,
) -> str:
    promotion_slug = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        promotion_id.removeprefix("promo_"),
    ).strip("_") or "content"
    digest = hashlib.sha256(
        f"{project_id}\x1f{idempotency_key}".encode("utf-8")
    ).hexdigest()[:16]
    max_slug_length = MAX_GENERATION_ID_LENGTH - len("generation__") - len(digest)
    return f"generation_{promotion_slug[:max_slug_length]}_{digest}"


def prompt_inputs_from_snapshot(
    value: Mapping[str, Any],
) -> list[GenerationPromptInput]:
    if value.get("schema_version") != GENERATION_REQUEST_SCHEMA_VERSION:
        raise GenerationSnapshotError("unsupported generation input schema_version")
    request = GenerationRequest(
        project_id=_required_text(value, "project_id"),
        campaign_id=_required_text(value, "campaign_id"),
        promotion_id=_required_text(value, "promotion_id"),
        analysis_id=_required_text(value, "analysis_id"),
        content_option_count=_required_positive_int(value, "content_option_count"),
        operator_instruction=_optional_text(value.get("operator_instruction")),
    )
    promotion_value = _required_mapping(value, "promotion")
    promotion = PromotionPromptInput(
        project_id=_required_text(promotion_value, "project_id"),
        campaign_id=_required_text(promotion_value, "campaign_id"),
        promotion_id=_required_text(promotion_value, "promotion_id"),
        channel=ContentChannel(_required_text(promotion_value, "channel")),
        goal_metric=_required_text(promotion_value, "goal_metric"),
        goal_target_value=_required_text(promotion_value, "goal_target_value"),
        goal_basis=_required_text(promotion_value, "goal_basis"),
        message_brief=_optional_text(promotion_value.get("message_brief")),
        landing_url=_optional_text(promotion_value.get("landing_url")),
        offer_type=_optional_text(promotion_value.get("offer_type")),
        landing_type=_optional_text(promotion_value.get("landing_type")),
    )
    raw_targets = value.get("target_segments")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise GenerationSnapshotError("target_segments must be a non-empty array")
    targets = [_target_segment_from_snapshot(item) for item in raw_targets]
    if len({item.segment_id for item in targets}) != len(targets):
        raise GenerationSnapshotError("target_segments must not contain duplicates")
    brand_context_value = value.get("brand_context")
    if brand_context_value is not None and not isinstance(
        brand_context_value,
        Mapping,
    ):
        raise GenerationSnapshotError("brand_context must be an object")
    try:
        brand_context = (
            BrandContextSnapshot.from_snapshot(brand_context_value)
            if isinstance(brand_context_value, Mapping)
            else None
        )
    except ValueError as exc:
        raise GenerationSnapshotError(str(exc)) from exc
    return GenerationInputBuilder().build(
        request=request,
        promotion=promotion,
        target_segments=targets,
        brand_context=brand_context,
    )


def _promotion_snapshot(value: PromotionPromptInput) -> dict[str, Any]:
    snapshot = asdict(value)
    snapshot["channel"] = value.channel.value
    return snapshot


def _target_segment_snapshot(value: TargetSegmentPromptInput) -> dict[str, Any]:
    snapshot = asdict(value)
    snapshot["content_brief_json"] = dict(value.content_brief_json)
    if value.source_content_brief_json is not None:
        snapshot["content_brief"] = dict(value.source_content_brief_json)
        snapshot["data_evidence"] = dict(value.data_evidence_json)
    return snapshot


def _target_segment_from_snapshot(value: object) -> TargetSegmentPromptInput:
    if not isinstance(value, Mapping):
        raise GenerationSnapshotError("target_segments entries must be objects")
    content_brief = value.get("content_brief_json")
    if not isinstance(content_brief, Mapping):
        raise GenerationSnapshotError("content_brief_json must be an object")
    source_content_brief = value.get("content_brief")
    if source_content_brief is not None and not isinstance(
        source_content_brief,
        Mapping,
    ):
        raise GenerationSnapshotError("content_brief must be an object")
    data_evidence = value.get("data_evidence")
    if data_evidence is not None and not isinstance(data_evidence, Mapping):
        raise GenerationSnapshotError("data_evidence must be an object")
    return TargetSegmentPromptInput(
        analysis_id=_required_text(value, "analysis_id"),
        promotion_id=_required_text(value, "promotion_id"),
        segment_id=_required_text(value, "segment_id"),
        segment_name=_required_text(value, "segment_name"),
        content_brief_json=dict(content_brief),
        segment_vector_id=_optional_text(value.get("segment_vector_id")),
        estimated_size=_nonnegative_int(value.get("estimated_size")),
        priority=_optional_text(value.get("priority")),
        content_slug=_optional_text(value.get("content_slug")),
        natural_language_query=_optional_text(value.get("natural_language_query")),
        generated_sql=_optional_text(value.get("generated_sql")),
        sample_ratio=_optional_text(value.get("sample_ratio")),
        source=_optional_text(value.get("source")),
        query_preview_id=_optional_text(value.get("query_preview_id")),
        status=_optional_text(value.get("status")),
        source_content_brief_json=(
            dict(source_content_brief)
            if isinstance(source_content_brief, Mapping)
            else None
        ),
        data_evidence_json=(
            dict(data_evidence) if isinstance(data_evidence, Mapping) else {}
        ),
    )


def _placement_snapshot(channel: ContentChannel) -> dict[str, str]:
    if channel == ContentChannel.EMAIL:
        return {"type": "email_body"}
    if channel == ContentChannel.SMS:
        return {"type": "sms_message"}
    return {"type": "onsite_banner", "slot_id": "C1_MAIN_TOP"}


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise GenerationSnapshotError(f"{key} must be an object")
    return item


def _required_text(value: Mapping[str, Any], key: str) -> str:
    text = _optional_text(value.get(key))
    if text is None:
        raise GenerationSnapshotError(f"{key} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_positive_int(value: Mapping[str, Any], key: str) -> int:
    number = _nonnegative_int(value.get(key))
    if number < 1:
        raise GenerationSnapshotError(f"{key} must be at least 1")
    return number


def _nonnegative_int(value: object) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise GenerationSnapshotError("expected an integer") from exc
    if number < 0:
        raise GenerationSnapshotError("expected a non-negative integer")
    return number
