from fastapi.testclient import TestClient

from app.contents.errors import ContentGenerationError
from app.contents.router import get_content_generation_service
from app.contents.schemas import GenerateContentResponse
from app.main import app


class SuccessfulContentService:
    def generate_content(self, request):
        return GenerateContentResponse(
            creative_id="1",
            action_id=request.action_id,
            content_url="https://cdn.example.com/final/banner.png",
            recommendation_action_id=10,
            mapping_id=20,
        )


class FailingContentService:
    def generate_content(self, request):
        raise ContentGenerationError(
            code="AD_CREATIVE_NOT_FOUND",
            message="creative missing",
            status_code=404,
        )


def test_generate_content_api_returns_contract_response() -> None:
    app.dependency_overrides[get_content_generation_service] = lambda: SuccessfulContentService()
    try:
        response = TestClient(app).post(
            "/contents/generate",
            json={
                "recommendation_result_id": 1,
                "action_id": "act_free_shipping_coupon",
                "force": False,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "creative_id": "1",
        "action_id": "act_free_shipping_coupon",
        "content_url": "https://cdn.example.com/final/banner.png",
        "recommendation_action_id": 10,
        "mapping_id": 20,
    }


def test_generate_content_api_returns_required_error_shape() -> None:
    app.dependency_overrides[get_content_generation_service] = lambda: FailingContentService()
    try:
        response = TestClient(app).post(
            "/contents/generate",
            json={
                "recommendation_result_id": 1,
                "action_id": "act_free_shipping_coupon",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "AD_CREATIVE_NOT_FOUND",
            "message": "creative missing",
        },
    }


def test_app_main_registers_contents_router() -> None:
    paths: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path:
            paths.add(path)

        router = getattr(route, "original_router", None)
        if router is not None:
            paths.update(getattr(child_route, "path", "") for child_route in router.routes)

    assert "/contents/generate" in paths
