from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.analysis.schemas import (
    FunnelRecommendationAnalysisRequest,
    FunnelRecommendationAnalysisResponse,
)
from app.analysis.service import run_funnel_recommendation_analysis
from app.db.clickhouse import ClickHouseClientFactory, get_clickhouse_client_factory
from app.db.postgres import get_postgres_session
from app.metrics.repository import FunnelMetricsRepository
from app.persistence.repository import PostgresRepository
from app.root_causes.repository import RootCauseRepository

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/funnel/recommend", response_model=FunnelRecommendationAnalysisResponse)
def recommend_for_funnel_analysis(
    request: FunnelRecommendationAnalysisRequest,
    client_factory: Annotated[ClickHouseClientFactory, Depends(get_clickhouse_client_factory)],
    postgres_session: Annotated[Session, Depends(get_postgres_session)],
) -> FunnelRecommendationAnalysisResponse:
    with client_factory() as client:
        return run_funnel_recommendation_analysis(
            request=request,
            metrics_repository=FunnelMetricsRepository(client),
            root_cause_repository=RootCauseRepository(client),
            persistence_repository=PostgresRepository(postgres_session),
        )
