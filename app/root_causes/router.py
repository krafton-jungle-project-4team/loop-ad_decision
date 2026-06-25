from typing import Annotated

from fastapi import APIRouter, Depends

from app.db.clickhouse import ClickHouseClientFactory, get_clickhouse_client_factory
from app.root_causes.repository import RootCauseRepository
from app.root_causes.schemas import RootCauseAnalysisRequest, RootCauseAnalysisResponse
from app.root_causes.service import calculate_root_causes

router = APIRouter(prefix="/root-causes", tags=["root-causes"])


@router.post("/funnel", response_model=RootCauseAnalysisResponse)
def get_funnel_root_causes(
    request: RootCauseAnalysisRequest,
    client_factory: Annotated[ClickHouseClientFactory, Depends(get_clickhouse_client_factory)],
) -> RootCauseAnalysisResponse:
    with client_factory() as client:
        repository = RootCauseRepository(client)
        return calculate_root_causes(request, repository)
