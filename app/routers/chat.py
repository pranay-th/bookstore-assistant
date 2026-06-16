"""
routers/chat.py — Conversational assistant endpoints.
All endpoints return 501 Not Implemented until Phase 1.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.chat import ChatRequest

router = APIRouter()


@router.post("", summary="Chat with the bookstore assistant")
def chat(request: ChatRequest):
    """
    POST /chat
    TODO: Return a grounded, catalog-aware assistant reply.
    """
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented — Phase 1"},
    )
