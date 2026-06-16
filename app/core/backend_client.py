"""
core/backend_client.py — Thin HTTP client for the Django backend.

The agent's tools call into the existing REST API rather than maintaining a
separate data store or embedding index.

The Django backend wraps every response in a standard envelope:

    {"status": {"success": bool, "code": int, "message": str},
     "data":   <payload> | null}

For list endpoints the payload itself is paginated:

    {"results": [...], "count": int, "num_pages": int, ...}

The helpers below unwrap that envelope so tools receive plain Python data.

TODO: Add auth header forwarding, retries, and timeout tuning.
"""
import httpx

from app.core.config import settings


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.DJANGO_API_URL, timeout=10.0)


def _unwrap(payload: dict) -> dict:
    """Return the ``data`` section of the backend's response envelope.

    Falls back to the raw payload if it isn't enveloped (defensive — keeps
    the client working if an endpoint returns bare data).
    """
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data") or {}
    return payload


def search_books(query: str, limit: int = 10) -> list[dict]:
    """
    Search the catalog via the Django backend.

    Hits ``GET /api/books/?search=<query>&page_size=<limit>``. The backend
    filters by title/author (case-insensitive) and returns a paginated,
    enveloped response; we unwrap it down to the ``results`` list.
    """
    with _client() as client:
        resp = client.get("/api/books/", params={"search": query, "page_size": limit})
        resp.raise_for_status()
        data = _unwrap(resp.json())

    if isinstance(data, list):
        return data
    return data.get("results", [])


def get_book(book_id: str) -> dict:
    """Fetch full details for a single book by id."""
    with _client() as client:
        resp = client.get(f"/api/books/{book_id}/")
        resp.raise_for_status()
        return _unwrap(resp.json())


def list_books_by_author(author: str, limit: int = 10) -> list[dict]:
    """
    List books by a given author.

    The backend's ``?search=`` matches both title and author, so we reuse it
    and then narrow to rows whose author actually contains the term — handy
    when the model wants an author's catalog specifically.
    """
    results = search_books(author, limit=limit)
    needle = author.strip().lower()
    filtered = [b for b in results if needle in (b.get("author") or "").lower()]
    return filtered or results
