"""
main.py — FastAPI application entrypoint for the AI assistant service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import health, chat, recommendations

app = FastAPI(
    title="Enterprise Book Store — AI Assistant Service",
    description="AI-native features: conversational assistant, recommendations, and semantic search.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router,            prefix="/chat",            tags=["Chat"])
app.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])
