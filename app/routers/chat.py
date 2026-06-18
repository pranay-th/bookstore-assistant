"""
routers/chat.py — Conversational assistant endpoints.
"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

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


def _sse(event: str, data: str) -> str:
    """Format one Server-Sent Event frame.

    Data is JSON-encoded so multi-line text and special characters survive the
    line-based SSE wire format.
    """
    payload = json.dumps({"type": event, "data": data})
    return f"data: {payload}\n\n"


@router.post("/stream", summary="Chat with the assistant (streaming, SSE)")
def chat_stream(
    request: ChatRequest,
    user: AuthenticatedUser = Depends(require_user),
):
    """
    POST /chat/stream

    Same as POST /chat, but streams the reply as Server-Sent Events so the UI
    can render tokens as they arrive and show progress during tool calls.

    Event frames (each a JSON object on a `data:` line):
        {"type": "status", "data": "Searching the catalog…"}
        {"type": "token",  "data": "partial text "}
        {"type": "done",   "data": "the full final reply"}
        {"type": "error",  "data": "message"}
    """
    if user.user_id and user.user_id != "anonymous":
        request.user_id = user.user_id

    service = AgentService()

    def event_source():
        try:
            for event, data in service.stream(request):
                yield _sse(event, data)
        except Exception as exc:  # noqa: BLE001 — never break the stream uncleanly
            yield _sse("error", f"Something went wrong: {exc}")

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx/render)
            "Connection": "keep-alive",
        },
    )
