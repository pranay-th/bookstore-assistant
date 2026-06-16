"""
services/agent_service.py — Agentic tool-calling loop.

The assistant answers by letting the LLM call backend tools (search_books,
get_book, ...) in a loop until it produces a final natural-language reply.
No embeddings or vector store — the model reasons over live catalog data
fetched through tools.

Loop shape (Phase 1):
    1. Send system + history + user message, advertising TOOL_SPECS.
    2. If the model requests tool calls, execute them via TOOL_IMPLS and
       append the results as tool messages.
    3. Repeat until the model returns a final message or AGENT_MAX_ITERATIONS
       is reached.
"""
import json

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import tools

SYSTEM_PROMPT = (
    "You are the Enterprise Book Store assistant. Help shoppers find books, "
    "answer questions about the catalog, and make recommendations. Use the "
    "provided tools to look up real catalog data instead of guessing."
)


class AgentService:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def run(self, request: ChatRequest) -> ChatResponse:
        """
        Execute the tool-calling loop for one chat turn.

        TODO (Phase 1):
          - Build the messages list: system + request.history + user message.
          - Call self.llm.chat.completions.create(..., tools=tools.TOOL_SPECS).
          - While the response has tool_calls and iterations < AGENT_MAX_ITERATIONS:
              dispatch each call to tools.TOOL_IMPLS, append tool results, re-call.
          - Return the final assistant message as ChatResponse.
        """
        raise NotImplementedError("Phase 1 — agent loop not implemented")

    def _dispatch_tool(self, name: str, arguments: str) -> str:
        """Execute a single tool call and return a JSON-serialized result."""
        impl = tools.TOOL_IMPLS.get(name)
        if impl is None:
            return json.dumps({"error": f"unknown tool: {name}"})
        kwargs = json.loads(arguments or "{}")
        return json.dumps(impl(**kwargs))
