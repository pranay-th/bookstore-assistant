"""
tests/test_recommendation_service.py — Unit tests for recommendations.

The LLM client is a FakeLLM and backend tools are patched; no real network.
"""
from unittest.mock import patch

from app.schemas.recommendations import RecommendationRequest
from app.services.recommendation_service import RecommendationService, _extract_json
from app.tests.helpers import FakeLLM, text_completion, tool_call_completion


def test_recommend_parses_json_reply():
    """A clean JSON reply is parsed into RecommendedBook entries."""
    reply = (
        '{"results": ['
        '{"book_id": "1", "title": "Dune", "reason": "Epic sci-fi", "score": 0.9},'
        '{"book_id": "2", "title": "Foundation", "reason": "Galactic scope", "score": 0.8}'
        ']}'
    )
    fake = FakeLLM([text_completion(reply)])
    service = RecommendationService(llm_client=fake)

    resp = service.recommend(RecommendationRequest(query="space opera", limit=5))

    assert len(resp.results) == 2
    assert resp.results[0].book_id == "1"
    assert resp.results[0].title == "Dune"
    assert resp.results[0].score == 0.9


def test_recommend_with_tool_round_trip():
    """The model searches the catalog, then returns ranked JSON."""
    fake = FakeLLM([
        tool_call_completion("search_books", '{"query": "mystery"}'),
        text_completion('{"results": [{"book_id": "7", "title": "Gone Girl", "score": 0.7}]}'),
    ])
    service = RecommendationService(llm_client=fake)

    with patch(
        "app.services.tools.backend_client.search_books",
        return_value=[{"id": "7", "title": "Gone Girl", "author": "Gillian Flynn"}],
    ) as mock_search:
        resp = service.recommend(RecommendationRequest(query="twisty mystery", limit=3))

    mock_search.assert_called_once_with(query="mystery")
    assert len(resp.results) == 1
    assert resp.results[0].title == "Gone Girl"


def test_recommend_respects_limit():
    """No more than `limit` results are returned even if the model over-produces."""
    items = ",".join(
        f'{{"book_id": "{i}", "title": "Book {i}", "score": 0.5}}' for i in range(10)
    )
    fake = FakeLLM([text_completion(f'{{"results": [{items}]}}')])
    service = RecommendationService(llm_client=fake)

    resp = service.recommend(RecommendationRequest(limit=3))

    assert len(resp.results) == 3


def test_recommend_handles_unparseable_reply():
    """A non-JSON reply yields an empty result list rather than crashing."""
    fake = FakeLLM([text_completion("Sorry, I have no good picks.")])
    service = RecommendationService(llm_client=fake)

    resp = service.recommend(RecommendationRequest(query="???", limit=5))

    assert resp.results == []


def test_extract_json_handles_fenced_block():
    """_extract_json strips ```json fences and surrounding prose."""
    text = 'Here you go:\n```json\n{"results": []}\n```\nHope that helps!'
    assert _extract_json(text) == {"results": []}
