"""
schemas/chat.py — Pydantic schemas for the conversational assistant.
"""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single turn in a conversation."""
    role:    Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Incoming chat request from the storefront."""
    message:         str = Field(..., min_length=1, description="User's latest message")
    history:         list[ChatMessage] = Field(default_factory=list)
    session_id:      Optional[str] = None
    user_id:         Optional[str] = None


class ChatResponse(BaseModel):
    """Assistant reply."""
    reply:      str
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # TODO: Add citations / referenced book_ids once RAG is wired up
