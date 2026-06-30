from __future__ import annotations

from contextlib import nullcontext
from types import TracebackType
from typing import Iterable, Protocol

from app.contents.types import GeneratedContentDraft, GeneratedContentRecord, RecommendationActionTarget


class GenerationLock(Protocol):
    def __enter__(self) -> None:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        ...


class ContentRepository(Protocol):
    def list_generation_targets(
        self,
        *,
        project_id: int | str,
        analysis_date: str,
        eligible_statuses: tuple[str, ...],
    ) -> Iterable[RecommendationActionTarget]:
        ...

    def generation_lock(
        self,
        *,
        project_id: int | str,
        recommendation_action_id: int,
    ) -> GenerationLock:
        ...

    def get_generated_content(
        self,
        *,
        project_id: int | str,
        recommendation_action_id: int,
        variant_key: str,
    ) -> GeneratedContentRecord | None:
        ...

    def upsert_generated_content(
        self,
        *,
        draft: GeneratedContentDraft,
        force: bool,
    ) -> GeneratedContentRecord:
        ...

    def mark_action_content_generated(self, *, recommendation_action_id: int) -> None:
        ...

    def mark_action_failed(
        self,
        *,
        recommendation_action_id: int,
        error_type: str,
        error_message: str,
    ) -> None:
        ...


def no_op_generation_lock() -> GenerationLock:
    return nullcontext()
