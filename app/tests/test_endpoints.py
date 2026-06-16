"""
tests/test_endpoints.py — Endpoint integration tests (LLM + backend mocked).

The chat and recommendation endpoints run the agentic loop, so we patch the
AgentService LLM client and the backend tool calls — no real network in CI.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.agent_service import AgentError
from app.tests.helpers import FakeLLM, auth_header, text_completion

client = TestClient(app)


def test_chat_returns_reply():
    """POST /chat returns a 200 with the assistant's reply text."""
    fake = FakeLLM([text_completion("Try 'Dune' by Frank Herbert.")])
    with patch("app.routers.chat.AgentService") as MockService:
        from app.services.agent_service import AgentService
        MockService.return_value = AgentService(llm_client=fake)

        resp = client.post(
            "/chat",
            json={"message": "Recommend a sci-fi book"},
            headers=auth_header(),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Try 'Dune' by Frank Herbert."


def test_chat_carries_session_id():
    """session_id from the request is echoed back on the response."""
    fake = FakeLLM([text_completion("Hello!")])
    with patch("app.routers.chat.AgentService") as MockService:
        from app.services.agent_service import AgentService
        MockService.return_value = AgentService(llm_client=fake)

        resp = client.post(
            "/chat",
            json={"message": "hi", "session_id": "sess-123"},
            headers=auth_header(),
        )

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "sess-123"


def test_chat_missing_api_key_returns_503():
    """A missing LLM_API_KEY surfaces as a clean 503, not a crash."""
    with patch("app.routers.chat.AgentService") as MockService:
        instance = MockService.return_value
        instance.run.side_effect = AgentError("LLM_API_KEY is not set")

        resp = client.post("/chat", json={"message": "hi"}, headers=auth_header())

    assert resp.status_code == 503
    assert "LLM_API_KEY" in resp.json()["detail"]


def test_recommendations_returns_results():
    """POST /recommendations returns parsed, ranked book picks."""
    reply = (
        '{"results": [{"book_id": "1", "title": "Dune", '
        '"reason": "Classic sci-fi", "score": 0.95}]}'
    )
    fake = FakeLLM([text_completion(reply)])
    with patch("app.routers.recommendations.RecommendationService") as MockService:
        from app.services.recommendation_service import RecommendationService
        MockService.return_value = RecommendationService(llm_client=fake)

        resp = client.post(
            "/recommendations",
            json={"query": "space opera", "limit": 5},
            headers=auth_header(),
        )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["book_id"] == "1"
    assert results[0]["title"] == "Dune"
    assert results[0]["score"] == 0.95
