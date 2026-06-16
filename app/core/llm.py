"""
core/llm.py — LLM client provider.

Centralizes construction of the LLM client used by the agentic tool-calling
loop so routers and services depend on a single configured instance.

TODO: Add retry/backoff, timeout tuning, and streaming support.
TODO: Support alternative providers (Anthropic, Azure OpenAI) behind this interface.
"""
from functools import lru_cache

from app.core.config import settings


@lru_cache
def get_llm_client():
    """
    Return a configured OpenAI client.

    Lazily imported so the service can boot (e.g. for /health) even when the
    SDK or API key is not yet configured.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — configure it in .env before using the assistant."
        )

    from openai import OpenAI

    return OpenAI(api_key=settings.OPENAI_API_KEY)
