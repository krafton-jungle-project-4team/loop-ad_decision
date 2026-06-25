from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.anomalies.router import router as anomalies_router
from app.actions.router import router as actions_router
from app.metrics.router import router as metrics_router
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
