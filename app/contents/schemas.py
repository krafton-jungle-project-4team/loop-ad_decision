from pydantic import BaseModel, Field


class GenerateContentRequest(BaseModel):
    recommendation_result_id: int = Field(gt=0)
    action_id: str = Field(min_length=1)
    force: bool = False


class GenerateContentResponse(BaseModel):
    creative_id: str
    action_id: str
    content_url: str
    recommendation_action_id: int | None = None
    mapping_id: int | None = None
