from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.contents.ai.gemini_image_provider import GeminiImageProvider
from app.contents.ai.image_provider import ImageProvider
from app.contents.ai.mock_image_provider import MockImageProvider
from app.contents.contents_service import ContentsService
from app.contents.errors import ContentGenerationError
from app.contents.schemas import GenerateContentRequest, GenerateContentResponse
from app.contents.storage.s3_asset_storage_service import AssetStorage, S3AssetStorage
from app.core.config import Settings, get_settings
from app.db.postgres import get_postgres_session
from app.persistence.repository import PostgresRepository

router = APIRouter(prefix="/contents", tags=["contents"])


def get_content_repository(
    session: Annotated[Session, Depends(get_postgres_session)],
) -> PostgresRepository:
    return PostgresRepository(session)


def get_image_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ImageProvider:
    if settings.image_provider == "gemini":
        return GeminiImageProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_image_model,
        )
    return MockImageProvider()


def get_asset_storage(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AssetStorage:
    return S3AssetStorage(settings)


def get_content_generation_service(
    repository: Annotated[PostgresRepository, Depends(get_content_repository)],
    image_provider: Annotated[ImageProvider, Depends(get_image_provider)],
    storage: Annotated[AssetStorage, Depends(get_asset_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ContentsService:
    return ContentsService(
        repository=repository,
        image_provider=image_provider,
        storage=storage,
        asset_key_prefix=settings.loopad_genai_generated_assets_prefix,
    )


@router.post("/generate", response_model=GenerateContentResponse)
def generate_content(
    request: GenerateContentRequest,
    service: Annotated[ContentsService, Depends(get_content_generation_service)],
) -> GenerateContentResponse | JSONResponse:
    try:
        return service.generate_content(request)
    except ContentGenerationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_response_body(),
        )
