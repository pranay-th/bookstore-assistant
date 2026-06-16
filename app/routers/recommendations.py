"""
routers/recommendations.py — AI book recommendation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import AuthenticatedUser, require_user
from app.schemas.recommendations import RecommendationRequest, RecommendationResponse
from app.services.agent_service import AgentError
from app.services.recommendation_service import RecommendationService

router = APIRouter()


@router.post("", response_model=RecommendationResponse, summary="Get AI book recommendations")
def recommend(
    request: RecommendationRequest,
    user: AuthenticatedUser = Depends(require_user),
) -> RecommendationResponse:
    """
    POST /recommendations

    Requires a valid Bearer access token (issued by the Django backend).
    Uses the agentic tool-calling loop to gather candidate books from the
    catalog, then ranks and explains the best picks.
    """
    # Trust the authenticated identity over any client-supplied user_id.
    if user.user_id and user.user_id != "anonymous":
        request.user_id = user.user_id

    service = RecommendationService()
    try:
        return service.recommend(request)
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
