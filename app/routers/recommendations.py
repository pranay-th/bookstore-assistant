"""
routers/recommendations.py — AI book recommendation endpoints.
"""
from fastapi import APIRouter, HTTPException

from app.schemas.recommendations import RecommendationRequest, RecommendationResponse
from app.services.agent_service import AgentError
from app.services.recommendation_service import RecommendationService

router = APIRouter()


@router.post("", response_model=RecommendationResponse, summary="Get AI book recommendations")
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """
    POST /recommendations

    Uses the agentic tool-calling loop to gather candidate books from the
    catalog, then ranks and explains the best picks.
    """
    service = RecommendationService()
    try:
        return service.recommend(request)
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
