"""
tests/test_backend_client.py — Unit tests for the Django backend client.

httpx is mocked; these verify we correctly unwrap the backend's
{"status": ..., "data": {"results": [...]}} envelope.
"""
from unittest.mock import MagicMock, patch

from app.core import backend_client


def _mock_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


def _patch_client(resp):
    """Patch backend_client._client() to yield a context-managed fake client."""
    fake_client = MagicMock()
    fake_client.get.return_value = resp
    ctx = MagicMock()
    ctx.__enter__.return_value = fake_client
    ctx.__exit__.return_value = False
    return patch.object(backend_client, "_client", return_value=ctx), fake_client


def test_search_books_unwraps_envelope():
    body = {
        "status": {"success": True, "code": 200, "message": "ok"},
        "data": {"results": [{"id": "1", "title": "Dune"}], "count": 1},
    }
    patcher, fake_client = _patch_client(_mock_response(body))
    with patcher:
        results = backend_client.search_books("dune", limit=5)

    assert results == [{"id": "1", "title": "Dune"}]
    fake_client.get.assert_called_once_with(
        "/api/books/", params={"search": "dune", "page_size": 5}
    )


def test_get_book_unwraps_data():
    body = {
        "status": {"success": True, "code": 200, "message": "ok"},
        "data": {"id": "1", "title": "Dune", "author": "Frank Herbert"},
    }
    patcher, _ = _patch_client(_mock_response(body))
    with patcher:
        book = backend_client.get_book("1")

    assert book["title"] == "Dune"
    assert book["author"] == "Frank Herbert"


def test_list_books_by_author_filters_results():
    body = {
        "status": {"success": True, "code": 200, "message": "ok"},
        "data": {
            "results": [
                {"id": "1", "title": "Dune", "author": "Frank Herbert"},
                {"id": "2", "title": "Some Other Book", "author": "Someone Else"},
            ]
        },
    }
    patcher, _ = _patch_client(_mock_response(body))
    with patcher:
        results = backend_client.list_books_by_author("Frank Herbert")

    assert len(results) == 1
    assert results[0]["author"] == "Frank Herbert"
