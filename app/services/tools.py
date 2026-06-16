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
]

# ---------------------------------------------------------------------------
# Name -> callable mapping the agent loop dispatches to
# ---------------------------------------------------------------------------
TOOL_IMPLS = {
    "search_books": backend_client.search_books,
    "get_book": backend_client.get_book,
}
