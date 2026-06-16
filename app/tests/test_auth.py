"""
tests/test_auth.py — JWT authentication tests.

Covers token verification (core/auth.py) and endpoint enforcement. The LLM is
mocked where an endpoint would otherwise call it, so no network is used.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.auth import AuthError, decode_token
from app.core.config import settings
from app.main import app
from app.tests.helpers import (
    FakeLLM,
    auth_header,
    make_access_token,
    text_completion,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests — decode_token
# ---------------------------------------------------------------------------

def test_decode_valid_token():
    token = make_access_token(user_id="user-abc")
    user = decode_token(token)
    assert user.user_id == "user-abc"
    assert user.token_type == "access"


def test_decode_expired_token_raises():
    token = make_access_token(expired=True)
    with pytest.raises(AuthError, match="expired"):
        decode_token(token)


def test_decode_bad_signature_raises():
    token = make_access_token(secret="some-other-secret")
    with pytest.raises(AuthError, match="Invalid authentication token"):
        decode_token(token)


def test_decode_refresh_token_rejected():
    token = make_access_token(token_type="refresh")
    with pytest.raises(AuthError, match="access token is required"):
        decode_token(token)


def test_decode_missing_user_id_claim():
    # Forge a token whose user id lives under a different claim name.
    token = make_access_token(user_id_claim="sub")
    with pytest.raises(AuthError, match="missing the user identity"):
        decode_token(token)


def test_decode_without_secret_configured_fails_closed():
    original = settings.JWT_SECRET
    settings.JWT_SECRET = ""
    try:
        with pytest.raises(AuthError, match="not configured"):
            decode_token(make_access_token())
    finally:
        settings.JWT_SECRET = original


# ---------------------------------------------------------------------------
# Endpoint enforcement
# ---------------------------------------------------------------------------

def test_chat_without_token_is_401():
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_recommendations_without_token_is_401():
    resp = client.post("/recommendations", json={"limit": 3})
    assert resp.status_code == 401


def test_chat_with_invalid_token_is_401():
    resp = client.post(
        "/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


def test_chat_with_valid_token_passes_and_sets_user_id():
    fake = FakeLLM([text_completion("Hi there!")])
    token = make_access_token(user_id="user-xyz")

    with patch("app.routers.chat.AgentService") as MockService:
        from app.services.agent_service import AgentService
        service = AgentService(llm_client=fake)
        MockService.return_value = service

        with patch.object(service, "run", wraps=service.run) as spy:
            resp = client.post(
                "/chat",
                json={"message": "hi", "user_id": "spoofed"},
                headers=auth_header(token),
            )

    assert resp.status_code == 200
    # The authenticated identity overrides any client-supplied user_id.
    called_request = spy.call_args.args[0]
    assert called_request.user_id == "user-xyz"


def test_health_does_not_require_auth():
    """Health probe must stay open for Railway/Render checks."""
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Optional-auth mode (REQUIRE_AUTH=False)
# ---------------------------------------------------------------------------

def test_optional_auth_allows_missing_token(monkeypatch):
    monkeypatch.setattr(settings, "REQUIRE_AUTH", False)
    fake = FakeLLM([text_completion("Anonymous hello")])

    with patch("app.routers.chat.AgentService") as MockService:
        from app.services.agent_service import AgentService
        MockService.return_value = AgentService(llm_client=fake)

        resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 200


def test_optional_auth_still_rejects_bad_token(monkeypatch):
    """Even with auth optional, a present-but-invalid token is rejected."""
    monkeypatch.setattr(settings, "REQUIRE_AUTH", False)
    resp = client.post(
        "/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer garbage"},
    )
    assert resp.status_code == 401
