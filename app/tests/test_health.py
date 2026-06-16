"""
tests/test_health.py — Health endpoint tests.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    """GET /health should return 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    """GET /health should return {'status': 'ok'}."""
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
