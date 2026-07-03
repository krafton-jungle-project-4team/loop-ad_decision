from fastapi import FastAPI

from app.analysis.router import router as analysis_router
from app.generation.router import router as generation_router


def create_app() -> FastAPI:
    app = FastAPI(title="Loop-Ad Decision API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "decision-api"}

    app.include_router(analysis_router)
    app.include_router(generation_router)
    return app


app = create_app()
