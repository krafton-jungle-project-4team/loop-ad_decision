from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.db import create_postgres_connection
from app.decision.repositories import PsycopgPostgresExecutor
from app.dependencies import get_settings, require_internal_key
from app.uplift.registry import (
    UpliftModelActivationError,
    UpliftModelLifecycleService,
    UpliftModelNotFoundError,
    UpliftModelRegistryRepository,
)


router = APIRouter(
    prefix="/internal/decision/v1/uplift",
    tags=["internal-uplift"],
    dependencies=[Depends(require_internal_key)],
)


class UpliftModelActivationRequest(BaseModel):
    approved_by: str = Field(min_length=1, max_length=100)


class UpliftModelActivationResponse(BaseModel):
    model_version_id: str
    lifecycle_status: str
    serving_eligible: bool
    approved_by: str
    approved_at: datetime
    activated_at: datetime


def get_uplift_model_lifecycle_service(
    request: Request,
) -> Iterator[UpliftModelLifecycleService]:
    connection = create_postgres_connection(get_settings(request))
    try:
        yield UpliftModelLifecycleService(
            UpliftModelRegistryRepository(
                PsycopgPostgresExecutor(connection)
            )
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@router.post(
    "/models/{model_version_id}/activate",
    response_model=UpliftModelActivationResponse,
)
def activate_uplift_model(
    model_version_id: str,
    payload: UpliftModelActivationRequest,
    lifecycle_service: UpliftModelLifecycleService = Depends(
        get_uplift_model_lifecycle_service
    ),
) -> UpliftModelActivationResponse:
    try:
        activated = lifecycle_service.activate_validated_model(
            model_version_id,
            approved_by=payload.approved_by,
        )
    except UpliftModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="uplift model version was not found",
        ) from exc
    except (UpliftModelActivationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if (
        activated.approved_by is None
        or activated.approved_at is None
        or activated.activated_at is None
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="activated uplift model is missing approval provenance",
        )
    return UpliftModelActivationResponse(
        model_version_id=activated.model_version_id,
        lifecycle_status=activated.lifecycle_status,
        serving_eligible=activated.serving_eligible,
        approved_by=activated.approved_by,
        approved_at=activated.approved_at,
        activated_at=activated.activated_at,
    )
