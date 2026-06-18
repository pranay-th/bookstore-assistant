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


def _auth_client(access_token: str) -> httpx.Client:
    """A client that forwards the user's JWT for user-scoped endpoints
    (cart, orders). Both services share SECRET_KEY, so the token validates."""
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    return httpx.Client(base_url=settings.DJANGO_API_URL, timeout=10.0, headers=headers)


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


# ---------------------------------------------------------------------------
# Cart + orders — user-scoped. These require the caller's JWT, forwarded via
# _auth_client. All endpoints are scoped to the authenticated user server-side.
# ---------------------------------------------------------------------------

# Cart line items can be large; project to what the model needs to reason/act.
_CART_ITEM_FIELDS = ("id", "book_id", "title", "author", "quantity", "price", "subtotal")


def _slim_cart(data: dict) -> dict:
    """Trim a cart payload to id, totals, and lean line items."""
    if not isinstance(data, dict):
        return {"items": [], "total_quantity": 0, "total_price": "0"}
    items = []
    for it in data.get("items", []) or []:
        if isinstance(it, dict):
            items.append({k: it[k] for k in _CART_ITEM_FIELDS if k in it})
    return {
        "items": items,
        "total_quantity": data.get("total_quantity", len(items)),
        "total_price": data.get("total_price", "0"),
    }


def get_cart(access_token: str) -> dict:
    """Return the authenticated user's cart (lean)."""
    with _auth_client(access_token) as client:
        resp = client.get("/api/cart/")
        resp.raise_for_status()
        return _slim_cart(_unwrap(resp.json()))


def add_to_cart(access_token: str, book_id: str, quantity: int = 1) -> dict:
    """Add a book to the cart (or increment if already present)."""
    quantity = max(1, int(quantity or 1))
    with _auth_client(access_token) as client:
        resp = client.post("/api/cart/add/", json={"book_id": book_id, "quantity": quantity})
        resp.raise_for_status()
        return _slim_cart(_unwrap(resp.json()))


def remove_cart_item(access_token: str, item_id: str) -> dict:
    """Remove a single line item from the cart by its cart-item id.

    Note: this is the cart *item* id (from get_cart's items[].id), not a book id.
    """
    with _auth_client(access_token) as client:
        resp = client.delete(f"/api/cart/{item_id}/remove/")
        resp.raise_for_status()
        return _slim_cart(_unwrap(resp.json()))


def clear_cart(access_token: str) -> dict:
    """Empty the authenticated user's cart."""
    with _auth_client(access_token) as client:
        resp = client.delete("/api/cart/clear/")
        resp.raise_for_status()
        return _slim_cart(_unwrap(resp.json()))


def place_order(access_token: str, payment_method: str = "card") -> dict:
    """Check out the user's current cart into a confirmed order.

    Reads the cart server-side, sends its items to the checkout endpoint
    (which simulates payment, creates the order, deducts stock, and clears the
    cart). Returns the order summary.
    """
    cart = get_cart(access_token)
    items = [
        {"book_id": it["book_id"], "quantity": it["quantity"]}
        for it in cart.get("items", [])
        if it.get("book_id")
    ]
    if not items:
        return {"error": "Your cart is empty — add a book before placing an order."}

    method = payment_method if payment_method in ("card", "paypal", "bank_transfer") else "card"
    with _auth_client(access_token) as client:
        resp = client.post(
            "/api/orders/checkout/",
            json={"items": items, "payment_method": method},
        )
        resp.raise_for_status()
        return _unwrap(resp.json())


def list_orders(access_token: str, limit: int = 5) -> list[dict]:
    """List the authenticated user's recent orders (lean)."""
    with _auth_client(access_token) as client:
        resp = client.get("/api/orders/")
        resp.raise_for_status()
        data = _unwrap(resp.json())

    rows = data if isinstance(data, list) else data.get("results", [])
    lean = []
    for o in rows[: max(1, int(limit or 5))]:
        if not isinstance(o, dict):
            continue
        lean.append({
            "order_id": o.get("id"),
            "status": o.get("status"),
            "total_amount": o.get("total_amount"),
            "item_count": len(o.get("items", []) or []),
            "created_at": o.get("created_at"),
        })
    return lean
