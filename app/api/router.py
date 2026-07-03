from fastapi import APIRouter

from app.api.generation import router as generation_router


decision_router = APIRouter(prefix="/decision/v1")
decision_router.include_router(generation_router)

