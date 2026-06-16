"""
tests/helpers.py — Test doubles for the LLM client.

These mimic just enough of the OpenAI SDK response shape that the agent loop
relies on (`response.choices[0].message` with `.content` and `.tool_calls`,
where each tool call has `.id` and `.function.name` / `.function.arguments`).
"""
import datetime

import jwt
from types import SimpleNamespace

# Shared test secret — also patched into settings.JWT_SECRET by the auth tests.
TEST_JWT_SECRET = "test-secret-key"


def make_access_token(
    user_id: str = "user-123",
    secret: str = TEST_JWT_SECRET,
    algorithm: str = "HS256",
    token_type: str = "access",
    expired: bool = False,
    user_id_claim: str = "user_id",
    extra_claims: dict | None = None,
) -> str:
    """Forge a simplejwt-style access token for tests."""
    now = datetime.datetime.now(datetime.timezone.utc)
    exp = now - datetime.timedelta(minutes=5) if expired else now + datetime.timedelta(minutes=30)
    claims = {
        user_id_claim: user_id,
        "token_type": token_type,
        "iat": now,
        "exp": exp,
        "jti": "test-jti",
    }
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, secret, algorithm=algorithm)


def auth_header(token: str | None = None) -> dict:
    """Build an Authorization header dict for the test client."""
    return {"Authorization": f"Bearer {token or make_access_token()}"}


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def text_completion(content: str) -> SimpleNamespace:
    """A final assistant message with no tool calls."""
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call_completion(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    """An assistant message that requests a single tool call."""
    message = SimpleNamespace(
        content="",
        tool_calls=[_tool_call(call_id, name, arguments)],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    """Stand-in for the OpenAI client.

    Returns the queued completions in order on each
    ``chat.completions.create(...)`` call. Records calls for assertions.
    """

    def __init__(self, completions):
        self._completions = list(completions)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        self.calls.append(kwargs)
        if not self._completions:
            raise AssertionError("FakeLLM ran out of queued completions")
        return self._completions.pop(0)
