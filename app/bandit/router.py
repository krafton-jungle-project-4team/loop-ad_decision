from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.bandit.schemas import BanditActionProbabilityResponse
from app.bandit.service import get_action_probabilities
from app.db.postgres import get_postgres_session
from app.persistence.repository import PostgresRepository

router = APIRouter(prefix="/bandit", tags=["bandit"])


def get_bandit_repository(
    session: Annotated[Session, Depends(get_postgres_session)],
) -> PostgresRepository:
    return PostgresRepository(session)


@router.get(
    "/policies/{bandit_policy_id}/action-probabilities",
    response_model=BanditActionProbabilityResponse,
)
def get_bandit_action_probabilities(
    bandit_policy_id: int,
    repository: Annotated[PostgresRepository, Depends(get_bandit_repository)],
    samples: int = Query(default=10000, ge=1, le=100000),
    seed: int = Query(default=42),
) -> BanditActionProbabilityResponse:
    return get_action_probabilities(
        repository=repository,
        bandit_policy_id=bandit_policy_id,
        samples=samples,
        seed=seed,
    )
