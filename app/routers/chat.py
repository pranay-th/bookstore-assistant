"""
routers/chat.py — Conversational assistant endpoints.
"""
from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent_service import AgentError, AgentService

router = APIRouter()


@router.post("", response_model=ChatResponse, summary="Chat with the bookstore assistant")
def chat(request: ChatRequest) -> ChatResponse:
    """
    POST /chat

    Runs the agentic tool-calling loop: the LLM may call backend catalog tools
    to ground its answer, then returns a natural-language reply.
    """
    service = AgentService()
    try:
        return service.run(request)
    except AgentError as exc:
        # Config/backend/LLM failures — surface as a clean 503, not a 500 crash.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
