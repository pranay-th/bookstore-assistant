"""
services/recommendation_service.py — AI book recommendation logic.

TODO: Drive recommendations through the agentic loop / backend tools —
fetch user history and trending data, then let the LLM rank and explain picks.
"""
from app.schemas.recommendations import RecommendationRequest, RecommendationResponse


class RecommendationService:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        TODO:
          1. Fetch user history / catalog and trending data via backend tools.
          2. Let the LLM rank candidates and explain each pick.
          3. Return top `request.limit` books with reasons.
        """
        raise NotImplementedError("Phase 1 — recommendations not implemented")
