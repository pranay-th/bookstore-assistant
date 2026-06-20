"""
services/tools.py — Tool definitions for the agentic loop.

Each tool pairs an OpenAI-style JSON schema (advertised to the model) with a
Python callable that executes it against the backend. The agent loop in
agent_service.py reads TOOL_SPECS to build the request and TOOL_IMPLS to run
whatever the model decides to call.

TODO: Add tools for inventory, orders, recommendations (analytics service).
"""
from app.core import backend_client

# ---------------------------------------------------------------------------
# Tool schemas advertised to the LLM
# ---------------------------------------------------------------------------
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": (
                "Search the bookstore catalog by title, author, or keyword. "
                "Returns a short list of matching books (id, title, author, "
                "price, stock). Call this at most once or twice per question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "limit": {"type": "integer", "description": "Max results (1-10)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_book",
            "description": (
                "Get details for a single book by its id. Only call this when "
                "the shopper asks about one specific book — search_books already "
                "returns enough to list and compare titles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {"type": "string", "description": "The book's unique id"},
                },
                "required": ["book_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_books_by_author",
            "description": (
                "List books written by a specific author. Use this when the "
                "shopper asks for an author's catalog or 'more like <author>'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "Author name"},
                    "limit": {"type": "integer", "description": "Max results (1-10)", "default": 5},
                },
                "required": ["author"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": (
                "View the shopper's current shopping cart: line items (each with "
                "a cart-item id, title, quantity, price) and totals."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": (
                "Add a book to the shopper's cart by book id (use search_books "
                "first to find the id). If it's already in the cart, the quantity "
                "increases. Confirm the title with the shopper before adding if "
                "there's any ambiguity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {"type": "string", "description": "The book's unique id"},
                    "quantity": {"type": "integer", "description": "Copies to add", "default": 1},
                },
                "required": ["book_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": (
                "Remove a line item from the cart. Pass the cart-item id (the "
                "'id' field from view_cart's items — NOT the book id). Call "
                "view_cart first if you don't have it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "The cart-item id from view_cart"},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_cart",
            "description": "Remove everything from the shopper's cart. Confirm with the shopper first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Place an order for everything currently in the shopper's cart "
                "(simulated payment — no real charge). Before calling this: make "
                "sure the cart has items (use view_cart), confirm the shopper "
                "wants to check out, ask which payment method they'd like, AND "
                "collect their delivery details (see the `delivery` argument). "
                "An order placed without delivery details cannot be tracked, so "
                "always gather the address first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_method": {
                        "type": "string",
                        "enum": ["card", "paypal", "bank_transfer"],
                        "description": "Payment method the shopper chose",
                        "default": "card",
                    },
                    "delivery": {
                        "type": "object",
                        "description": (
                            "Delivery contact + shipping address, collected from "
                            "the shopper. Ask for any missing required fields "
                            "before placing the order."
                        ),
                        "properties": {
                            "full_name": {"type": "string", "description": "Recipient's full name"},
                            "email": {"type": "string", "description": "Contact email"},
                            "phone": {"type": "string", "description": "Contact phone (optional)"},
                            "line1": {"type": "string", "description": "Address line 1 — house/flat, street"},
                            "line2": {"type": "string", "description": "Address line 2 — area/landmark (optional)"},
                            "city": {"type": "string", "description": "City"},
                            "state": {"type": "string", "description": "State / province"},
                            "postal_code": {"type": "string", "description": "Postal / PIN code"},
                            "country": {"type": "string", "description": "ISO-2 country code (default IN)"},
                            "notes": {"type": "string", "description": "Delivery instructions (optional)"},
                        },
                        "required": ["full_name", "email", "line1", "city", "state", "postal_code"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_order_tracking",
            "description": (
                "Open the live delivery tracking page for one of the shopper's "
                "orders inside the app. Call this ONLY when the shopper asks to "
                "track an order / see where it is / view tracking, OR after you "
                "offer to show tracking and they confirm. Use the order's id "
                "(from a just-placed order or from list_orders)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order id to open tracking for"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "List the shopper's recent orders with status and totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max orders (1-10)", "default": 5},
                },
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Name -> callable mapping the agent loop dispatches to.
#
# Each value is a thin wrapper that resolves the backend_client function at
# call time (rather than binding the reference at import). This keeps the
# indirection trivial while making the backend calls straightforward to mock
# in tests (patch app.core.backend_client.<fn>).
# ---------------------------------------------------------------------------


def _search_books(**kwargs):
    return backend_client.search_books(**kwargs)


def _get_book(**kwargs):
    return backend_client.get_book(**kwargs)


def _list_books_by_author(**kwargs):
    return backend_client.list_books_by_author(**kwargs)


# ── User-scoped tools — they receive the shopper's access_token (injected by
#    the agent loop, never supplied by the model) as the first argument. ──
def _view_cart(access_token, **kwargs):
    return backend_client.get_cart(access_token)


def _add_to_cart(access_token, **kwargs):
    return backend_client.add_to_cart(access_token, **kwargs)


def _remove_from_cart(access_token, **kwargs):
    return backend_client.remove_cart_item(access_token, **kwargs)


def _clear_cart(access_token, **kwargs):
    return backend_client.clear_cart(access_token)


def _place_order(access_token, **kwargs):
    return backend_client.place_order(access_token, **kwargs)


def _open_order_tracking(**kwargs):
    """Signal the UI to open the tracking page. No backend call — the agent
    loop turns this into a navigation 'action' event for the client."""
    order_id = str(kwargs.get("order_id") or "").strip()
    if not order_id:
        return {"error": "open_order_tracking needs the order_id."}
    return {"ok": True, "order_id": order_id, "message": "Opening the tracking page."}


def _list_orders(access_token, **kwargs):
    return backend_client.list_orders(access_token, **kwargs)


TOOL_IMPLS = {
    "search_books": _search_books,
    "get_book": _get_book,
    "list_books_by_author": _list_books_by_author,
    "view_cart": _view_cart,
    "add_to_cart": _add_to_cart,
    "remove_from_cart": _remove_from_cart,
    "clear_cart": _clear_cart,
    "place_order": _place_order,
    "open_order_tracking": _open_order_tracking,
    "list_orders": _list_orders,
}

# Tools that act on the authenticated user's data. The agent loop injects the
# access_token for these; they are unavailable when the request is anonymous.
USER_SCOPED_TOOLS = {
    "view_cart",
    "add_to_cart",
    "remove_from_cart",
    "clear_cart",
    "place_order",
    "list_orders",
}
