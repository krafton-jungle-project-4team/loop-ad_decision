from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.db import create_clickhouse_client
from app.dependencies import get_settings, require_internal_key
from app.internal.schemas import (
    UserBehaviorVectorBuildRequest,
    UserBehaviorVectorBuildResponse,
)
from app.internal.user_behavior_vectors import (
    UserBehaviorVectorBatchService,
    UserBehaviorVectorBuildRepository,
)


router = APIRouter(
    prefix="/internal/decision/v1/batches",
    tags=["internal-batches"],
    dependencies=[Depends(require_internal_key)],
)


@router.post(
    "/user-behavior-vectors/build",
    response_model=UserBehaviorVectorBuildResponse,
    status_code=status.HTTP_200_OK,
)
def build_user_behavior_vectors(
    payload: UserBehaviorVectorBuildRequest,
    request: Request,
) -> UserBehaviorVectorBuildResponse:
    settings = get_settings(request)
    clickhouse_client = create_clickhouse_client(settings)
    try:
        batch_service = UserBehaviorVectorBatchService(
            repository=UserBehaviorVectorBuildRepository(clickhouse_client),
        )
        result = batch_service.build(payload)
    finally:
        _close_clickhouse_client(clickhouse_client)

    return UserBehaviorVectorBuildResponse(
        project_id=result.project_id,
        vector_version=result.vector_version,
        source=result.source,
        vector_dim=result.vector_dim,
        processed_user_count=result.processed_user_count,
        window_start=result.window_start,
        window_end=result.window_end,
        status=result.status,
    )


def _close_clickhouse_client(clickhouse_client: object | None) -> None:
    if clickhouse_client is None:
        return
    close = getattr(clickhouse_client, "close", None)
    if callable(close):
        close()
