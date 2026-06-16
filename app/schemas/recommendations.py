"""
schemas/recommendations.py — Pydantic schemas for AI book recommendations.
"""
from typing import Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request for personalized or query-driven book recommendations."""
    user_id:  Optional[str] = None
    query:    Optional[str] = Field(None, description="Free-text intent, e.g. 'cozy mysteries set in winter'")
    limit:    int = Field(5, ge=1, le=50)


class RecommendedBook(BaseModel):
    """A single recommended book entry."""
    book_id: str
    title:   str
    reason:  Optional[str] = None
    score:   float = 0.0


class RecommendationResponse(BaseModel):
    """List of recommended books."""
    results: list[RecommendedBook] = Field(default_factory=list)
