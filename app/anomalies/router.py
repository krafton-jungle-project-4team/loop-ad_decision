from typing import Annotated

from fastapi import APIRouter, Depends

from app.anomalies.schemas import FunnelAnomalyRequest, FunnelAnomalyResponse
from app.anomalies.service import calculate_funnel_anomalies
from app.db.clickhouse import ClickHouseClientFactory, get_clickhouse_client_factory
from app.metrics.repository import FunnelMetricsRepository

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.post("/funnel", response_model=FunnelAnomalyResponse)
def get_funnel_anomalies(
    request: FunnelAnomalyRequest,
    client_factory: Annotated[ClickHouseClientFactory, Depends(get_clickhouse_client_factory)],
) -> FunnelAnomalyResponse:
    with client_factory() as client:
        repository = FunnelMetricsRepository(client)
        return calculate_funnel_anomalies(request, repository)
