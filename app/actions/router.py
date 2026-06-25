from fastapi import APIRouter

from app.actions.schemas import ActionRecommendationRequest, ActionRecommendationResponse
from app.actions.service import recommend_actions

router = APIRouter(prefix="/actions", tags=["actions"])


@router.post("/recommend", response_model=ActionRecommendationResponse)
def recommend_marketing_actions(
    request: ActionRecommendationRequest,
) -> ActionRecommendationResponse:
    return recommend_actions(request)
