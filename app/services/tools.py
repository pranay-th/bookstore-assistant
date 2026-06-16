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
            "description": "Search the bookstore catalog by title, author, or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_book",
            "description": "Get full details for a single book by its id.",
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
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["author"],
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


TOOL_IMPLS = {
    "search_books": _search_books,
    "get_book": _get_book,
    "list_books_by_author": _list_books_by_author,
}
