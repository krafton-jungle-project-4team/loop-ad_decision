from typing import Annotated

from fastapi import APIRouter, Depends

from app.db.clickhouse import ClickHouseClientFactory, get_clickhouse_client_factory
from app.metrics.repository import FunnelMetricsRepository
from app.metrics.schemas import FunnelMetricRequest, FunnelMetricResponse
from app.metrics.service import calculate_funnel_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/funnel", response_model=FunnelMetricResponse)
def get_funnel_metrics(
    request: FunnelMetricRequest,
    client_factory: Annotated[ClickHouseClientFactory, Depends(get_clickhouse_client_factory)],
) -> FunnelMetricResponse:
    with client_factory() as client:
        repository = FunnelMetricsRepository(client)
        return calculate_funnel_metrics(request, repository)
