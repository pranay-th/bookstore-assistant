"""
services/recommendation_service.py — AI book recommendation logic.

Recommendations run through the same agentic tool-calling loop as chat: the
LLM uses backend tools to gather candidate books, then ranks and explains the
best picks. The model is asked to return a strict JSON object which we parse
into the RecommendationResponse schema.
"""
import json
import logging
import re

from app.schemas.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedBook,
)
from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

RECO_SYSTEM_PROMPT = (
    "You are the Enterprise Book Store recommendation engine. Use the provided "
    "tools to find real books in the catalog that match the shopper's request, "
    "then choose the best picks. Only recommend books returned by the tools — "
    "never invent titles or ids.\n\n"
    "Respond with ONLY a JSON object (no markdown, no prose) of the form:\n"
    '{"results": [{"book_id": "<id>", "title": "<title>", '
    '"reason": "<why it fits>", "score": <0..1>}]}\n'
    "Order results best-first and include at most the requested number."
)


def _build_user_prompt(request: RecommendationRequest) -> str:
    parts = [f"Recommend up to {request.limit} books."]
    if request.query:
        parts.append(f"The shopper is interested in: {request.query!r}.")
    else:
        parts.append("The shopper gave no specific query — suggest popular, broadly appealing titles.")
    if request.user_id:
        parts.append(f"(user_id: {request.user_id})")
    parts.append(
        "Use search_books / list_books_by_author to find candidates, then return "
        "the JSON object described in the system prompt."
    )
    return " ".join(parts)


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from the model's reply.

    Handles bare JSON as well as JSON wrapped in ```fences``` or surrounded by
    stray prose, which smaller models occasionally emit.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the first {...} block in the string.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


class RecommendationService:
    """Generates ranked, explained book recommendations via the agent loop."""

    def __init__(self, llm_client=None):
        self._agent = AgentService(llm_client=llm_client)

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Run the tool-calling loop and parse the model's JSON into the
        recommendation response schema."""
        messages = self._agent._build_messages(
            RECO_SYSTEM_PROMPT, [], _build_user_prompt(request)
        )
        reply = self._agent._run_loop(messages)  # may raise AgentError

        parsed = _extract_json(reply)
        raw_results = parsed.get("results", []) if isinstance(parsed, dict) else []

        results: list[RecommendedBook] = []
        for item in raw_results[: request.limit]:
            if not isinstance(item, dict):
                continue
            book_id = item.get("book_id")
            title = item.get("title")
            if book_id is None or title is None:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            results.append(
                RecommendedBook(
                    book_id=str(book_id),
                    title=str(title),
                    reason=item.get("reason"),
                    score=score,
                )
            )

        return RecommendationResponse(results=results)
