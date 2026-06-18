"""
core/config.py — Application settings via pydantic-settings.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV:   str = 'development'
    APP_DEBUG: bool = True
    PORT:      int = 8002

    CORS_ORIGINS:          str = 'http://localhost:3000'
    DJANGO_API_URL:        str = 'http://localhost:8000'
    ANALYTICS_SERVICE_URL: str = 'http://localhost:8001'

    # LLM provider (agentic tool-calling loop) — OpenAI-compatible API.
    # Defaults target OpenRouter with DeepSeek V3.2.
    LLM_API_KEY:  str = ''
    LLM_BASE_URL: str = 'https://openrouter.ai/api/v1'
    LLM_MODEL:    str = 'deepseek/deepseek-v3.2'

    # Cap on tokens generated per LLM response. Without this, some models/
    # providers (e.g. DeepSeek on OpenRouter) default to a very large
    # max_tokens (64k+), which OpenRouter rejects with HTTP 402 when the key's
    # remaining credit can't cover the reservation. Keep modest — chat replies
    # and recommendation JSON are short.
    LLM_MAX_TOKENS: int = 1024

    # Safety bound on tool-calling iterations per chat turn. Each iteration is
    # a full LLM call, so keeping this low directly limits credit spend. 3-4 is
    # plenty: typically one search round + a final answer.
    AGENT_MAX_ITERATIONS: int = 4

    # ------------------------------------------------------------------
    # Auth — validates JWT access tokens issued by the Django backend.
    #
    # The backend uses djangorestframework-simplejwt (HS256) signing with
    # Django's SECRET_KEY. To verify those tokens here statelessly, JWT_SECRET
    # MUST match the backend's SECRET_KEY, and the algorithm / claim names must
    # match the backend's SIMPLE_JWT config.
    # ------------------------------------------------------------------
    JWT_SECRET:       str = ''           # == Django SECRET_KEY
    JWT_ALGORITHM:    str = 'HS256'
    JWT_USER_ID_CLAIM: str = 'user_id'   # == SIMPLE_JWT['USER_ID_CLAIM']

    # When True, /chat and /recommendations require a valid Bearer token.
    # When False, auth is optional (token is decoded if present, ignored if not)
    # — handy for local development.
    REQUIRE_AUTH: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(',')]

    class Config:
        env_file = '.env'


settings = Settings()
