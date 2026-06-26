from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.anomalies.router import router as anomalies_router
from app.actions.router import router as actions_router
from app.core.config import get_settings
from app.db.postgres import create_postgres_tables
from app.metrics.router import router as metrics_router
from app.root_causes.router import router as root_causes_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.postgres_auto_create_tables:
        try:
            create_postgres_tables(settings)
        except Exception:
            logger.warning("PostgreSQL table auto-create skipped.", exc_info=True)
    yield


app = FastAPI(title="LoopAd AI Analysis Server", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(metrics_router)
app.include_router(anomalies_router)
app.include_router(root_causes_router)
app.include_router(actions_router)
