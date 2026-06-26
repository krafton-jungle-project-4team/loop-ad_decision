from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.automation.schemas import AutomationPolicyResponse, AutomationPolicyUpsertRequest
from app.db.postgres import get_postgres_session
from app.persistence.repository import PostgresRepository

router = APIRouter(prefix="/automation-policies", tags=["automation-policies"])


def get_automation_policy_repository(
    session: Annotated[Session, Depends(get_postgres_session)],
) -> PostgresRepository:
    return PostgresRepository(session)


@router.get("/{project_id}", response_model=AutomationPolicyResponse)
def get_automation_policy(
    project_id: str,
    repository: Annotated[PostgresRepository, Depends(get_automation_policy_repository)],
) -> AutomationPolicyResponse:
    policy = repository.get_automation_policy(project_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="automation policy not found",
        )
    return AutomationPolicyResponse.model_validate(policy)


@router.put("/{project_id}", response_model=AutomationPolicyResponse)
def put_automation_policy(
    project_id: str,
    request: AutomationPolicyUpsertRequest,
    repository: Annotated[PostgresRepository, Depends(get_automation_policy_repository)],
) -> AutomationPolicyResponse:
    try:
        policy = repository.upsert_automation_policy(
            project_id=project_id,
            values=request.model_dump(),
        )
        repository.commit()
    except Exception:
        repository.rollback()
        raise
    return AutomationPolicyResponse.model_validate(policy)
