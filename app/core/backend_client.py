"""
core/backend_client.py — Thin HTTP client for the Django backend.

The agent's tools call into the existing REST API rather than maintaining a
separate data store or embedding index.

TODO: Add auth header forwarding, retries, and timeout tuning.
"""
import httpx

from app.core.config import settings


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.DJANGO_API_URL, timeout=10.0)


def search_books(query: str, limit: int = 10) -> list[dict]:
    """
    Search the catalog via the Django backend.
    TODO: Confirm the exact query param/endpoint with the backend team.
    """
    with _client() as client:
        resp = client.get("/api/books/", params={"search": query, "page_size": limit})
        resp.raise_for_status()
        return resp.json().get("results", [])


def get_book(book_id: str) -> dict:
    """Fetch full details for a single book."""
    with _client() as client:
        resp = client.get(f"/api/books/{book_id}/")
        resp.raise_for_status()
        return resp.json()
