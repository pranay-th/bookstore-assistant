"""
tests/test_endpoints.py — Phase 0 placeholder endpoint tests.

All AI endpoints should return 501 Not Implemented until Phase 1.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_chat_not_implemented():
    response = client.post("/chat", json={"message": "Recommend a sci-fi book"})
    assert response.status_code == 501


def test_recommendations_not_implemented():
    response = client.post("/recommendations", json={"limit": 5})
    assert response.status_code == 501
