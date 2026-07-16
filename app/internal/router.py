from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.analysis.repositories import PsycopgPostgresExecutor
from app.db import create_clickhouse_client, create_postgres_connection
from app.dependencies import get_settings, require_internal_key
from app.internal.schemas import (
    UserBehaviorVectorBuildRequest,
    UserBehaviorVectorBuildResponse,
    UserBehaviorVectorSearchSyncRequest,
    UserBehaviorVectorSearchSyncResponse,
)
from app.internal.user_behavior_vector_search_sync import (
    UserBehaviorVectorSearchSyncRepository,
    UserBehaviorVectorSearchSyncService,
)
from app.internal.user_behavior_vectors import (
    HOTEL_BEHAVIOR_V2,
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
    connection = None
    try:
        batch_service = UserBehaviorVectorBatchService(
            repository=UserBehaviorVectorBuildRepository(clickhouse_client),
        )
        result = batch_service.build(payload)
        if payload.vector_version == HOTEL_BEHAVIOR_V2:
            connection = create_postgres_connection(settings)
            UserBehaviorVectorSearchSyncRepository(
                clickhouse=clickhouse_client,
                postgres=PsycopgPostgresExecutor(connection),
            ).register_generation(
                vector_generation_id=result.vector_generation_id,
                project_id=result.project_id,
                vector_version=result.vector_version,
                manifest_hash=result.manifest_hash,
                window_start=result.window_start,
                window_end=result.window_end,
                expected_user_count=result.expected_user_count,
                source_revision_cutoff=result.source_revision_cutoff,
            )
            connection.commit()
    except Exception:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if connection is not None:
            connection.close()
        _close_clickhouse_client(clickhouse_client)

    return UserBehaviorVectorBuildResponse(
        project_id=result.project_id,
        vector_version=result.vector_version,
        source=result.source,
        vector_dim=result.vector_dim,
        processed_user_count=result.processed_user_count,
        vector_generation_id=result.vector_generation_id,
        expected_user_count=result.expected_user_count,
        manifest_hash=result.manifest_hash,
        window_start=result.window_start,
        window_end=result.window_end,
        status=result.status,
    )


@router.post(
    "/user-behavior-vector-search/sync",
    response_model=UserBehaviorVectorSearchSyncResponse,
    status_code=status.HTTP_200_OK,
)
def sync_user_behavior_vector_search(
    payload: UserBehaviorVectorSearchSyncRequest,
    request: Request,
) -> UserBehaviorVectorSearchSyncResponse:
    settings = get_settings(request)
    clickhouse_client = create_clickhouse_client(settings)
    connection = create_postgres_connection(settings)
    try:
        service = UserBehaviorVectorSearchSyncService(
            UserBehaviorVectorSearchSyncRepository(
                clickhouse=clickhouse_client,
                postgres=PsycopgPostgresExecutor(connection),
            )
        )
        result = service.sync(
            project_id=payload.project_id,
            vector_version=payload.vector_version,
            vector_generation_id=payload.vector_generation_id,
            batch_size=payload.batch_size,
            max_batches=payload.max_batches,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        _close_clickhouse_client(clickhouse_client)
    return UserBehaviorVectorSearchSyncResponse(
        project_id=result.project_id,
        vector_version=result.vector_version,
        vector_generation_id=result.vector_generation_id,
        synced_user_count=result.synced_user_count,
        expected_user_count=result.expected_user_count,
        active_generation_id=result.active_generation_id,
        source_cutoff=result.source_cutoff,
        status=result.status,
    )


def _close_clickhouse_client(clickhouse_client: object | None) -> None:
    if clickhouse_client is None:
        return
    close = getattr(clickhouse_client, "close", None)
    if callable(close):
        close()
