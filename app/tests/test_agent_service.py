"""
tests/test_agent_service.py — Unit tests for the agentic tool-calling loop.

Everything external is mocked: the LLM client is a FakeLLM and the backend
tool callables are patched, so no real network or API calls happen.
"""
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.schemas.chat import ChatMessage, ChatRequest
from app.services.agent_service import AgentError, AgentService
from app.tests.helpers import FakeLLM, text_completion, tool_call_completion


def test_no_tool_reply():
    """The model answers directly without any tool calls."""
    fake = FakeLLM([text_completion("Hello, happy to help!")])
    service = AgentService(llm_client=fake)

    resp = service.run(ChatRequest(message="hi"))

    assert resp.reply == "Hello, happy to help!"
    # Exactly one LLM call, and tools were advertised.
    assert len(fake.calls) == 1
    assert "tools" in fake.calls[0]


def test_single_tool_call_round_trip():
    """The model calls search_books once, then answers using the result."""
    fake = FakeLLM([
        tool_call_completion("search_books", '{"query": "dune"}'),
        text_completion("I found Dune by Frank Herbert."),
    ])
    service = AgentService(llm_client=fake)

    with patch(
        "app.services.tools.backend_client.search_books",
        return_value=[{"id": "1", "title": "Dune", "author": "Frank Herbert"}],
    ) as mock_search:
        resp = service.run(ChatRequest(message="find dune"))

    assert resp.reply == "I found Dune by Frank Herbert."
    mock_search.assert_called_once_with(query="dune")
    # Two LLM round trips: tool request + final answer.
    assert len(fake.calls) == 2

    # The second call's messages must include the tool result.
    second_messages = fake.calls[1]["messages"]
    roles = [m["role"] for m in second_messages]
    assert "tool" in roles
    tool_msg = next(m for m in second_messages if m["role"] == "tool")
    assert "Dune" in tool_msg["content"]


def test_history_is_included_in_messages():
    """Prior conversation turns are forwarded to the model."""
    fake = FakeLLM([text_completion("Sure!")])
    service = AgentService(llm_client=fake)

    request = ChatRequest(
        message="and a sequel?",
        history=[
            ChatMessage(role="user", content="recommend a fantasy book"),
            ChatMessage(role="assistant", content="Try The Hobbit."),
        ],
    )
    service.run(request)

    messages = fake.calls[0]["messages"]
    # system + 2 history + 1 user
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "recommend a fantasy book"
    assert messages[2]["content"] == "Try The Hobbit."
    assert messages[-1]["content"] == "and a sequel?"


def test_max_iteration_guard():
    """If the model keeps calling tools, the loop stops at the cap and still
    returns a final answer (via the no-tools fallback call)."""
    # One tool-call completion per iteration, plus the forced final answer.
    completions = [
        tool_call_completion("search_books", '{"query": "x"}', call_id=f"call_{i}")
        for i in range(settings.AGENT_MAX_ITERATIONS)
    ]
    completions.append(text_completion("Here's my best answer so far."))
    fake = FakeLLM(completions)
    service = AgentService(llm_client=fake)

    with patch(
        "app.services.tools.backend_client.search_books",
        return_value=[],
    ):
        resp = service.run(ChatRequest(message="loop forever"))

    assert resp.reply == "Here's my best answer so far."
    # MAX_ITERATIONS tool rounds + 1 forced final-answer call.
    assert len(fake.calls) == settings.AGENT_MAX_ITERATIONS + 1
    # The final call must NOT advertise tools.
    assert "tools" not in fake.calls[-1]


def test_unknown_tool_returns_error_to_model():
    """An unknown tool name yields a JSON error the model can read, and the
    loop continues to a final answer."""
    fake = FakeLLM([
        tool_call_completion("nonexistent_tool", "{}"),
        text_completion("Sorry, I couldn't do that."),
    ])
    service = AgentService(llm_client=fake)

    resp = service.run(ChatRequest(message="do something weird"))

    assert resp.reply == "Sorry, I couldn't do that."
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert "unknown tool" in tool_msg["content"]


def test_backend_error_is_caught():
    """A backend HTTP error becomes a JSON error message, not a crash."""
    import httpx

    fake = FakeLLM([
        tool_call_completion("search_books", '{"query": "dune"}'),
        text_completion("The catalog seems unavailable right now."),
    ])
    service = AgentService(llm_client=fake)

    with patch(
        "app.services.tools.backend_client.search_books",
        side_effect=httpx.ConnectError("connection refused"),
    ):
        resp = service.run(ChatRequest(message="find dune"))

    assert resp.reply == "The catalog seems unavailable right now."
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert "backend request failed" in tool_msg["content"]


def test_missing_api_key_raises_agent_error():
    """With no injected client and no API key, run() raises AgentError."""
    service = AgentService()  # no client injected -> resolves get_llm_client()

    with patch("app.services.agent_service.get_llm_client",
               side_effect=RuntimeError("LLM_API_KEY is not set")):
        with pytest.raises(AgentError, match="LLM_API_KEY"):
            service.run(ChatRequest(message="hi"))
