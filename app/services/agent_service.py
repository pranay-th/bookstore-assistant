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
import logging

import httpx

from app.core.config import settings
from app.core.llm import get_llm_client
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the Enterprise Book Store assistant. Help shoppers find books, "
    "answer questions about the catalog, and make recommendations. Use the "
    "provided tools to look up real catalog data instead of guessing. When you "
    "reference a book, prefer details returned by the tools (title, author, "
    "price). If a search returns nothing, say so honestly rather than inventing "
    "titles. Keep replies concise and friendly.\n\n"
    "Be economical with tools: usually a single search_books call answers the "
    "question. Do not call tools repeatedly with similar queries, and do not "
    "fetch per-book details unless the shopper asks about one specific book. "
    "Once you have enough information, answer directly."
)


class AgentError(Exception):
    """Raised when the agent loop cannot complete (config/backend failure)."""


def _normalize_llm_error(exc: Exception) -> str:
    """Turn an LLM/provider exception into a user-safe message.

    OpenRouter returns HTTP 402 when the key lacks credit for the requested
    max_tokens — call that out explicitly so it's actionable rather than a
    generic failure.
    """
    status = getattr(exc, "status_code", None)
    if status == 402 or "402" in str(exc):
        return (
            "The assistant is temporarily unavailable due to an LLM credit "
            "limit. Please try again later."
        )
    return f"LLM request failed: {exc}"


def _status_for_tools(names: str) -> str:
    """Map tool names to a friendly progress message for streaming clients."""
    if "search_books" in names or "list_books_by_author" in names:
        return "Searching the catalog…"
    if "get_book" in names:
        return "Looking up book details…"
    return "Checking the shelves…"


class AgentService:
    """Runs the LLM tool-calling loop for one chat turn."""

    def __init__(self, llm_client=None):
        # Allow injection (tests pass a mock); fall back to the shared client.
        self._llm = llm_client

    @property
    def llm(self):
        """Lazily resolve the LLM client so the service can be constructed
        without a configured API key (e.g. at import time)."""
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self, request: ChatRequest) -> ChatResponse:
        """Execute the tool-calling loop and return the assistant reply."""
        messages = self._build_messages(SYSTEM_PROMPT, request.history, request.message)
        reply_text = self._run_loop(messages)
        return ChatResponse(reply=reply_text, session_id=request.session_id)

    # ------------------------------------------------------------------
    # Streaming entrypoint
    # ------------------------------------------------------------------
    def stream(self, request: ChatRequest):
        """Run the agentic loop, yielding events for the UI as they happen.

        This is a generator of (event_type, data) tuples:
            ("status", str)  — human-readable progress (e.g. "Searching…")
            ("token",  str)  — a chunk of the final answer's text
            ("done",   str)  — the complete final answer (for convenience)
            ("error",  str)  — a terminal error message

        Tool-call rounds are not streamed token-by-token (they aren't
        user-facing prose); instead we emit a "status" event per round and
        stream only the model's final, tool-free answer.
        """
        try:
            llm = self.llm
        except RuntimeError as exc:
            yield ("error", str(exc))
            return

        messages = self._build_messages(SYSTEM_PROMPT, request.history, request.message)

        try:
            yield from self._stream_loop(llm, messages)
        except AgentError as exc:
            yield ("error", str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming loop failed")
            yield ("error", f"Something went wrong: {exc}")

    def _stream_loop(self, llm, messages: list[dict]):
        """Tool loop that streams the final answer's tokens."""
        for iteration in range(settings.AGENT_MAX_ITERATIONS):
            # Non-streaming call to decide whether tools are needed this round.
            try:
                response = llm.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    tools=tools.TOOL_SPECS,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM request failed during streaming")
                raise AgentError(_normalize_llm_error(exc)) from exc

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                # Final answer reached. Stream it token-by-token for a snappy UX.
                final = (message.content or "").strip()
                if final:
                    yield from self._stream_text(final)
                    yield ("done", final)
                    return
                # Empty content with no tools — fall through to a streamed call.
                break

            # Announce the tool round, then execute the tools.
            names = ", ".join(c.function.name for c in tool_calls)
            yield ("status", _status_for_tools(names))

            messages.append(self._assistant_tool_message(message, tool_calls))
            for call in tool_calls:
                result = self._dispatch_tool(call.function.name, call.function.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            yield ("status", "Reading the results…")

        # Hit the iteration cap (or empty final) — stream one tool-free answer.
        yield ("status", "Putting it together…")
        text = self._force_final_answer(llm, messages)
        yield from self._stream_text(text)
        yield ("done", text)

    @staticmethod
    def _stream_text(text: str):
        """Yield text in small word-grouped chunks as ('token', chunk).

        The decision call above is not a streaming completion, so we already
        have the full final text; chunking it here keeps the UI's typing
        animation smooth without a second model round trip.
        """
        words = text.split(" ")
        chunk = []
        for i, word in enumerate(words):
            chunk.append(word)
            # Emit every few words to balance smoothness vs. event volume.
            if len(chunk) >= 3 or i == len(words) - 1:
                piece = " ".join(chunk)
                if i != len(words) - 1:
                    piece += " "
                yield ("token", piece)
                chunk = []

    # ------------------------------------------------------------------
    # Core loop (shared with recommendation service)
    # ------------------------------------------------------------------
    def _run_loop(self, messages: list[dict]) -> str:
        """Drive the LLM <-> tool conversation until a final answer or the
        iteration cap is hit. Returns the final assistant text."""
        try:
            llm = self.llm
        except RuntimeError as exc:
            # Missing API key, etc. — surface as a clean error, not a crash.
            raise AgentError(str(exc)) from exc

        for iteration in range(settings.AGENT_MAX_ITERATIONS):
            try:
                response = llm.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    tools=tools.TOOL_SPECS,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
            except Exception as exc:  # noqa: BLE001 — normalize SDK/network errors
                logger.exception("LLM request failed")
                raise AgentError(_normalize_llm_error(exc)) from exc

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            # No tool calls -> the model produced its final answer.
            if not tool_calls:
                return (message.content or "").strip()

            # Record the assistant's tool-call turn, then run each tool.
            messages.append(self._assistant_tool_message(message, tool_calls))
            for call in tool_calls:
                result = self._dispatch_tool(call.function.name, call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        # Exhausted the iteration budget — ask for a final answer without tools
        # so the user still gets a usable reply instead of an error.
        return self._force_final_answer(llm, messages)

    def _force_final_answer(self, llm, messages: list[dict]) -> str:
        """Make one last call with no tools to squeeze out a final reply after
        the iteration cap is reached."""
        messages.append(
            {
                "role": "system",
                "content": (
                    "You have reached the tool-call limit. Provide the best final "
                    "answer you can using the information gathered so far."
                ),
            }
        )
        try:
            response = llm.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:  # noqa: BLE001
            logger.exception("Final-answer LLM request failed after max iterations")

        return (
            "I wasn't able to finish looking that up just now. Could you try "
            "rephrasing or narrowing your request?"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_messages(system_prompt: str, history, user_message: str) -> list[dict]:
        """Assemble the OpenAI-style messages list: system + history + user."""
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for turn in history or []:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _assistant_tool_message(message, tool_calls) -> dict:
        """Serialize the assistant message that requested tool calls so it can
        be appended back into the conversation."""
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }

    def _dispatch_tool(self, name: str, arguments: str) -> str:
        """Execute a single tool call and return a JSON-serialized result.

        Errors are returned as JSON (not raised) so the model can read them and
        recover (e.g. retry with a different query) within the loop.
        """
        impl = tools.TOOL_IMPLS.get(name)
        if impl is None:
            return json.dumps({"error": f"unknown tool: {name}"})

        try:
            kwargs = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid tool arguments: {exc}"})

        try:
            return json.dumps(impl(**kwargs))
        except httpx.HTTPError as exc:
            logger.warning("Tool %s backend call failed: %s", name, exc)
            return json.dumps({"error": f"backend request failed: {exc}"})
        except TypeError as exc:
            # Wrong/missing arguments for the tool callable.
            return json.dumps({"error": f"bad arguments for {name}: {exc}"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s raised", name)
            return json.dumps({"error": f"tool {name} failed: {exc}"})
