"""
routers/health.py — Health check endpoint.
Used by Railway's health probe.
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check():
    """
    GET /health
    Returns 200 OK — used by Railway health probes.
    """
    return {"status": "ok"}
