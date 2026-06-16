"""
routers/recommendations.py — AI book recommendation endpoints.
All endpoints return 501 Not Implemented until Phase 1.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.recommendations import RecommendationRequest

router = APIRouter()


@router.post("", summary="Get AI book recommendations")
def recommend(request: RecommendationRequest):
    """
    POST /recommendations
    TODO: Return personalized / query-driven book recommendations.
    """
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented — Phase 1"},
    )
