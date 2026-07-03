from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.router import decision_router


def create_app() -> FastAPI:
    app = FastAPI(title="Loop-Ad Decision API", version="0.1.0")
    app.include_router(decision_router)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        _request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "decision-api"}

    return app


app = create_app()

