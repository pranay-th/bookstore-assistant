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


# Fields worth showing the model in a *list* result. We deliberately drop
# heavy/irrelevant fields (description, isbn, cover_url, created_at, is_active)
# because list results are re-sent to the model on every subsequent loop
# iteration — keeping them lean is the single biggest token saving.
_LIST_FIELDS = ("id", "title", "author", "price", "stock")

# Max characters of a single book's description we feed back to the model.
_DESCRIPTION_CAP = 400


def _slim_list_book(book: dict) -> dict:
    """Project a book row down to the fields useful for browsing/ranking."""
    if not isinstance(book, dict):
        return book
    return {k: book[k] for k in _LIST_FIELDS if k in book}


def _slim_detail_book(book: dict) -> dict:
    """Trim a single-book detail payload: keep useful fields, cap description."""
    if not isinstance(book, dict):
        return book
    keep = ("id", "title", "author", "price", "stock", "published_year",
            "language", "category", "description")
    slim = {k: book[k] for k in keep if k in book}
    desc = slim.get("description")
    if isinstance(desc, str) and len(desc) > _DESCRIPTION_CAP:
        slim["description"] = desc[:_DESCRIPTION_CAP].rstrip() + "…"
    return slim


def search_books(query: str, limit: int = 5) -> list[dict]:
    """
    Search the catalog via the Django backend.

    Hits ``GET /api/books/?search=<query>&page_size=<limit>``. The backend
    filters by title/author (case-insensitive) and returns a paginated,
    enveloped response; we unwrap it and project each row down to a few light
    fields so the loop stays token-cheap.
    """
    # Hard cap to keep token usage (and credit spend) bounded.
    limit = max(1, min(int(limit or 5), 10))
    with _client() as client:
        resp = client.get("/api/books/", params={"search": query, "page_size": limit})
        resp.raise_for_status()
        data = _unwrap(resp.json())

    rows = data if isinstance(data, list) else data.get("results", [])
    return [_slim_list_book(b) for b in rows]


def get_book(book_id: str) -> dict:
    """Fetch details for a single book by id (trimmed, description capped)."""
    with _client() as client:
        resp = client.get(f"/api/books/{book_id}/")
        resp.raise_for_status()
        return _slim_detail_book(_unwrap(resp.json()))


def list_books_by_author(author: str, limit: int = 5) -> list[dict]:
    """
    List books by a given author.

    The backend's ``?search=`` matches both title and author, so we reuse it
    and then narrow to rows whose author actually contains the term — handy
    when the model wants an author's catalog specifically. Results are already
    slimmed by ``search_books``.
    """
    results = search_books(author, limit=limit)
    needle = author.strip().lower()
    filtered = [b for b in results if needle in (b.get("author") or "").lower()]
    return filtered or results
