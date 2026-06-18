"""
tests/test_streaming.py — Tests for the SSE streaming chat path.

The LLM client is a FakeLLM and backend tools are patched; no network is used.
"""
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.chat import ChatRequest
from app.services.agent_service import AgentService
from app.tests.helpers import (
    FakeLLM,
    auth_header,
    text_completion,
    tool_call_completion,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit — AgentService.stream generator
# ---------------------------------------------------------------------------

def test_stream_no_tool_emits_tokens_and_done():
    fake = FakeLLM([text_completion("Here is a great book for you.")])
    service = AgentService(llm_client=fake)

    events = list(service.stream(ChatRequest(message="hi")))

    types = [e[0] for e in events]
    assert "token" in types
    assert types[-1] == "done"

    # Concatenated tokens reconstruct the full reply.
    streamed = "".join(d for t, d in events if t == "token").strip()
    assert streamed == "Here is a great book for you."
    # The done event carries the full reply too.
    assert events[-1][1] == "Here is a great book for you."


def test_stream_emits_status_during_tool_round():
    fake = FakeLLM([
        tool_call_completion("search_books", '{"query": "dune"}'),
        text_completion("Found Dune by Frank Herbert."),
    ])
    service = AgentService(llm_client=fake)

    with patch(
        "app.services.tools.backend_client.search_books",
        return_value=[{"id": "1", "title": "Dune", "author": "Frank Herbert"}],
    ):
        events = list(service.stream(ChatRequest(message="find dune")))

    types = [e[0] for e in events]
    assert "status" in types  # a progress update was emitted for the tool round
    assert types[-1] == "done"
    assert "Dune" in events[-1][1]


def test_stream_missing_api_key_emits_error():
    service = AgentService()  # no injected client
    with patch(
        "app.services.agent_service.get_llm_client",
        side_effect=RuntimeError("LLM_API_KEY is not set"),
    ):
        events = list(service.stream(ChatRequest(message="hi")))

    assert len(events) == 1
    assert events[0][0] == "error"
    assert "LLM_API_KEY" in events[0][1]


# ---------------------------------------------------------------------------
# Endpoint — POST /chat/stream (SSE)
# ---------------------------------------------------------------------------

def _parse_sse(body: str):
    """Parse SSE 'data: {json}' frames into a list of dicts."""
    events = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_stream_endpoint_requires_auth():
    resp = client.post("/chat/stream", json={"message": "hi"})
    assert resp.status_code == 401


def test_stream_endpoint_returns_sse_events():
    fake = FakeLLM([text_completion("A streamed answer.")])
    with patch("app.routers.chat.AgentService") as MockService:
        MockService.return_value = AgentService(llm_client=fake)
        resp = client.post(
            "/chat/stream",
            json={"message": "hi"},
            headers=auth_header(),
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"
    assert events[-1]["data"] == "A streamed answer."
