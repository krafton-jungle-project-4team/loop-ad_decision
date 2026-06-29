import uvicorn

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.analysis.router import router as analysis_router
from app.anomalies.router import router as anomalies_router
from app.actions.router import router as actions_router
from app.automation.router import router as automation_router
from app.bandit.router import router as bandit_router
from app.contents.router import router as contents_router
from app.core.config import get_settings
from app.metrics.router import router as metrics_router
from app.recommendations.router import (
    ad_mappings_router,
    recommendations_router,
)
from app.root_causes.router import router as root_causes_router

app = FastAPI(title="LoopAd AI Analysis Server")


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
app.include_router(automation_router)
app.include_router(analysis_router)
app.include_router(bandit_router)
app.include_router(recommendations_router)
app.include_router(ad_mappings_router)
app.include_router(contents_router)


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
    )


if __name__ == "__main__":
    main()
