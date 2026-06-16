"""
core/llm.py — LLM client provider.

Centralizes construction of the LLM client used by the agentic tool-calling
loop so routers and services depend on a single configured instance.

Uses an OpenAI-compatible client pointed at OpenRouter (LLM_BASE_URL), so the
same code works with DeepSeek V3.2 or any other OpenRouter-hosted model.

TODO: Add retry/backoff, timeout tuning, and streaming support.
"""
from functools import lru_cache

from app.core.config import settings


@lru_cache
def get_llm_client():
    """
    Return a configured OpenAI-compatible client (OpenRouter).

    Lazily imported so the service can boot (e.g. for /health) even when the
    SDK or API key is not yet configured.
    """
    if not settings.LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is not set — configure your OpenRouter API key in .env "
            "before using the assistant."
        )

    from openai import OpenAI

    return OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
