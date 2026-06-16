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

    # LLM provider (agentic tool-calling loop)
    OPENAI_API_KEY: str = ''
    OPENAI_MODEL:   str = 'gpt-4o-mini'

    # Safety bound on tool-calling iterations per chat turn
    AGENT_MAX_ITERATIONS: int = 6

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(',')]

    class Config:
        env_file = '.env'


settings = Settings()
