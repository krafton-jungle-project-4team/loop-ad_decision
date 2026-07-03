from fastapi import APIRouter, HTTPException

from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    ContentBriefResponse,
    TargetSegmentResponse,
)

router = APIRouter(prefix="/decision/v1/promotions", tags=["analysis"])


@router.post("/{promotion_id}/analysis", response_model=AnalysisResponse)
def analyze_promotion(
    promotion_id: str,
    request: AnalysisRequest,
) -> AnalysisResponse:
    if promotion_id != request.promotion_id:
        raise HTTPException(
            status_code=400,
            detail="path promotion_id must match request promotion_id",
        )

    return AnalysisResponse(
        analysis_id=f"analysis_{promotion_id}",
        promotion_id=promotion_id,
        status=AnalysisStatus.COMPLETED,
        target_segments=[
            TargetSegmentResponse(
                segment_id="seg_repeat_hotel_no_booking",
                segment_name="Repeat hotel viewers without booking",
                segment_vector_id="segvec_repeat_hotel_no_booking_v1",
                estimated_size=1342,
                content_brief=ContentBriefResponse(
                    message_direction=(
                        "Emphasize free cancellation, same-day availability, "
                        "and breakfast benefits."
                    ),
                    keywords=[
                        "free cancellation",
                        "same-day availability",
                        "breakfast included",
                    ],
                ),
            )
        ],
    )
