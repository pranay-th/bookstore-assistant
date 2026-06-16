"""
routers/chat.py — Conversational assistant endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import AuthenticatedUser, require_user
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent_service import AgentError, AgentService

router = APIRouter()


@router.post("", response_model=ChatResponse, summary="Chat with the bookstore assistant")
def chat(
    request: ChatRequest,
    user: AuthenticatedUser = Depends(require_user),
) -> ChatResponse:
    """
    POST /chat

    Requires a valid Bearer access token (issued by the Django backend).
    Runs the agentic tool-calling loop: the LLM may call backend catalog tools
    to ground its answer, then returns a natural-language reply.
    """
    # Trust the authenticated identity over any client-supplied user_id.
    if user.user_id and user.user_id != "anonymous":
        request.user_id = user.user_id

    service = AgentService()
    try:
        return service.run(request)
    except AgentError as exc:
        # Config/backend/LLM failures — surface as a clean 503, not a 500 crash.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
