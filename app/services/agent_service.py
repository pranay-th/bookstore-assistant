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
    "answer questions about the catalog, manage their cart, and place orders. "
    "Use the provided tools to look up real catalog data and act on the "
    "shopper's account instead of guessing. When you reference a book, prefer "
    "details returned by the tools (title, author, price). If a search returns "
    "nothing, say so honestly rather than inventing titles. Keep replies "
    "concise and friendly.\n\n"
    "Be economical with tools: usually a single search_books call answers a "
    "question. Don't call tools repeatedly with similar queries, and don't "
    "fetch per-book details unless the shopper asks about one specific book. "
    "Once you have enough information, answer directly.\n\n"
    "Cart & orders:\n"
    "- To add a book you need its id; use search_books first if you only have a "
    "title, and confirm you have the right book.\n"
    "- To remove an item, call view_cart to get the cart-item id, then "
    "remove_from_cart with that id.\n"
    "- Before placing an order, confirm the shopper wants to check out and ask "
    "which payment method they'd like (card, paypal, or bank_transfer). Only "
    "then call place_order. Payment is simulated — no real charge — so tell the "
    "shopper that. After ordering, share the order id, total, and status.\n"
    "- Never invent cart contents, prices, order ids, or totals: always read "
    "them from the tools."
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
    if "place_order" in names:
        return "Placing your order…"
    if "add_to_cart" in names:
        return "Adding to your cart…"
    if "remove_from_cart" in names or "clear_cart" in names:
        return "Updating your cart…"
    if "view_cart" in names:
        return "Checking your cart…"
    if "list_orders" in names:
        return "Looking up your orders…"
    if "search_books" in names or "list_books_by_author" in names:
        return "Searching the catalog…"
    if "get_book" in names:
        return "Looking up book details…"
    return "Checking the shelves…"


class AgentService:
    """Runs the LLM tool-calling loop for one chat turn."""

    def __init__(self, llm_client=None, access_token=None):
        # Allow injection (tests pass a mock); fall back to the shared client.
        self._llm = llm_client
        # The shopper's JWT, forwarded to user-scoped tools (cart, orders).
        self._access_token = access_token

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
        """Agentic loop with real token streaming.

        Each turn is a streaming completion advertising the tools. As chunks
        arrive we:
          - emit any assistant *content* deltas immediately as ('token', …),
          - accumulate any *tool_call* deltas (which arrive split across chunks).

        Models in this stack don't mix prose with tool calls in one turn: if the
        model wants tools, the turn carries tool-call deltas and (near-)empty
        content. So when a turn finishes:
          - tool calls present  -> execute them, append results, continue;
          - otherwise           -> the streamed content was the final answer.
        """
        for iteration in range(settings.AGENT_MAX_ITERATIONS):
            try:
                stream = llm.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    tools=tools.TOOL_SPECS,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    stream=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM request failed during streaming")
                raise AgentError(_normalize_llm_error(exc)) from exc

            content_parts: list[str] = []
            tool_acc: dict[int, dict] = {}
            streamed_any = False

            try:
                for chunk in stream:
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    if delta is None:
                        continue

                    piece = getattr(delta, "content", None)
                    if piece:
                        content_parts.append(piece)
                        streamed_any = True
                        yield ("token", piece)

                    for tc in (getattr(delta, "tool_calls", None) or []):
                        self._accumulate_tool_call(tool_acc, tc)
            except Exception:  # noqa: BLE001
                logger.exception("Error reading streamed turn")

            # No tool calls -> the streamed content was the final answer.
            if not tool_acc:
                final = "".join(content_parts).strip()
                if streamed_any and final:
                    yield ("done", final)
                    return
                # Edge case: empty turn, nudge once more without tools.
                yield ("status", "Writing your answer…")
                full = yield from self._stream_completion(llm, messages)
                yield ("done", full or "I couldn't quite finish that — try rephrasing?")
                return

            # Tool turn — announce, record the assistant turn, run the tools.
            calls = self._finalize_tool_calls(tool_acc)
            names = ", ".join(c["function"]["name"] for c in calls)
            yield ("status", _status_for_tools(names))

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    "tool_calls": calls,
                }
            )
            for call in calls:
                result = self._dispatch_tool(
                    call["function"]["name"], call["function"]["arguments"]
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result}
                )
            yield ("status", "Reading the results…")

        # Hit the iteration cap — make a final, tool-free streaming answer.
        yield ("status", "Putting it together…")
        messages.append(
            {
                "role": "system",
                "content": (
                    "You have reached the tool-call limit. Provide the best "
                    "final answer you can using the information gathered so far."
                ),
            }
        )
        full = yield from self._stream_completion(llm, messages)
        yield ("done", full or "I couldn't quite finish that — try rephrasing?")

    @staticmethod
    def _accumulate_tool_call(acc: dict, tc) -> None:
        """Merge a streamed tool_call delta into the accumulator by index.

        Tool calls stream as partial deltas: the id/name usually arrive first,
        then the arguments string builds up across subsequent chunks.
        """
        idx = getattr(tc, "index", 0) or 0
        slot = acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["name"] = fn.name
            if getattr(fn, "arguments", None):
                slot["arguments"] += fn.arguments

    @staticmethod
    def _finalize_tool_calls(acc: dict) -> list[dict]:
        """Turn the accumulator into ordered OpenAI-style tool_call dicts."""
        calls = []
        for idx in sorted(acc):
            slot = acc[idx]
            calls.append(
                {
                    "id": slot["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": slot["name"] or "",
                        "arguments": slot["arguments"] or "{}",
                    },
                }
            )
        return calls

    def _stream_completion(self, llm, messages: list[dict]):
        """Make a streaming (no-tools) completion, yielding ('token', chunk)
        for each delta. Returns the full concatenated text."""
        collected = []
        try:
            stream = llm.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                max_tokens=settings.LLM_MAX_TOKENS,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming completion request failed")
            raise AgentError(_normalize_llm_error(exc)) from exc

        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    collected.append(piece)
                    yield ("token", piece)
        except Exception:  # noqa: BLE001
            logger.exception("Error while reading completion stream")

        return "".join(collected).strip()

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

        User-scoped tools (cart, orders) receive the shopper's access_token,
        injected here — never supplied by the model. If there's no token
        (anonymous request), those tools return a clear error instead of acting.
        """
        impl = tools.TOOL_IMPLS.get(name)
        if impl is None:
            return json.dumps({"error": f"unknown tool: {name}"})

        try:
            kwargs = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid tool arguments: {exc}"})

        if name in tools.USER_SCOPED_TOOLS:
            if not self._access_token:
                return json.dumps(
                    {"error": "Please sign in to manage your cart or place an order."}
                )
            args = (self._access_token,)
        else:
            args = ()

        try:
            return json.dumps(impl(*args, **kwargs))
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 401:
                return json.dumps({"error": "Your session has expired — please sign in again."})
            if status_code == 404:
                return json.dumps({"error": "That item could not be found."})
            logger.warning("Tool %s backend call failed: %s", name, exc)
            return json.dumps({"error": f"backend request failed ({status_code})"})
        except httpx.HTTPError as exc:
            logger.warning("Tool %s backend call failed: %s", name, exc)
            return json.dumps({"error": f"backend request failed: {exc}"})
        except TypeError as exc:
            # Wrong/missing arguments for the tool callable.
            return json.dumps({"error": f"bad arguments for {name}: {exc}"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s raised", name)
            return json.dumps({"error": f"tool {name} failed: {exc}"})
