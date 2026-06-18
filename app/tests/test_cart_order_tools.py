"""
tests/test_cart_order_tools.py — cart/order tool dispatch + backend client.

Verifies the agent injects the access token into user-scoped tools, guards
anonymous requests, and that the backend_client cart/order helpers forward the
Bearer token and unwrap the envelope. httpx is mocked; no network.
"""
import json
from unittest.mock import MagicMock, patch

from app.core import backend_client
from app.services import tools
from app.services.agent_service import AgentService


# ---------------------------------------------------------------------------
# Dispatch — token injection + anonymous guard
# ---------------------------------------------------------------------------

def test_user_scoped_tool_receives_access_token():
    service = AgentService(llm_client=object(), access_token="tok-123")
    with patch(
        "app.services.tools.backend_client.add_to_cart",
        return_value={"items": [], "total_quantity": 0, "total_price": "0"},
    ) as mock_add:
        out = service._dispatch_tool("add_to_cart", '{"book_id": "b1", "quantity": 2}')

    # token injected as first positional arg; model args passed through.
    mock_add.assert_called_once_with("tok-123", book_id="b1", quantity=2)
    assert "items" in json.loads(out)


def test_user_scoped_tool_without_token_is_guarded():
    service = AgentService(llm_client=object(), access_token=None)
    out = service._dispatch_tool("view_cart", "{}")
    assert "sign in" in json.loads(out)["error"].lower()


def test_catalog_tool_does_not_receive_token():
    service = AgentService(llm_client=object(), access_token="tok-123")
    with patch(
        "app.services.tools.backend_client.search_books",
        return_value=[{"id": "1", "title": "Dune"}],
    ) as mock_search:
        service._dispatch_tool("search_books", '{"query": "dune"}')
    mock_search.assert_called_once_with(query="dune")  # no token


def test_all_user_scoped_tools_are_registered():
    for name in tools.USER_SCOPED_TOOLS:
        assert name in tools.TOOL_IMPLS
    # And every new tool is advertised in the schema.
    advertised = {t["function"]["name"] for t in tools.TOOL_SPECS}
    for name in ("view_cart", "add_to_cart", "remove_from_cart",
                 "clear_cart", "place_order", "list_orders"):
        assert name in advertised


# ---------------------------------------------------------------------------
# backend_client — cart/order helpers
# ---------------------------------------------------------------------------

def _mock_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


def _patch_auth_client(resp):
    fake_client = MagicMock()
    fake_client.get.return_value = resp
    fake_client.post.return_value = resp
    fake_client.delete.return_value = resp
    ctx = MagicMock()
    ctx.__enter__.return_value = fake_client
    ctx.__exit__.return_value = False
    return patch.object(backend_client, "_auth_client", return_value=ctx), fake_client


def test_add_to_cart_forwards_token_and_slims():
    book_id = "1c6343bd-032c-48a2-9dff-53541ef0167f"
    body = {
        "status": {"success": True},
        "data": {
            "id": "cart1",
            "total_quantity": 1,
            "total_price": "499.00",
            "items": [
                {
                    "id": "item1", "book_id": book_id, "title": "Dune",
                    "author": "Frank Herbert", "quantity": 1, "price": "499.00",
                    "subtotal": "499.00", "cover_url": "x", "created_at": "y",
                }
            ],
        },
    }
    patcher, fake_client = _patch_auth_client(_mock_response(body))
    with patcher as mock_auth_client:
        cart = backend_client.add_to_cart("tok-123", book_id=book_id, quantity=1)
        mock_auth_client.assert_called_once_with("tok-123")

    fake_client.post.assert_called_once_with(
        "/api/cart/add/", json={"book_id": book_id, "quantity": 1}
    )
    # Slimmed: heavy fields dropped from line items.
    item = cart["items"][0]
    assert "cover_url" not in item and "created_at" not in item
    assert item["title"] == "Dune"


def test_place_order_checks_out_cart_items():
    cart_body = {
        "status": {"success": True},
        "data": {
            "total_quantity": 2, "total_price": "998.00",
            "items": [
                {"id": "i1", "book_id": "b1", "title": "Dune", "quantity": 2,
                 "price": "499.00", "subtotal": "998.00"},
            ],
        },
    }
    checkout_body = {
        "status": {"success": True},
        "data": {
            "order_id": "o1", "status": "confirmed",
            "total_amount": "998.00", "item_count": 1,
            "payment_method": "card",
        },
    }
    fake_client = MagicMock()
    fake_client.get.return_value = _mock_response(cart_body)
    fake_client.post.return_value = _mock_response(checkout_body)
    ctx = MagicMock()
    ctx.__enter__.return_value = fake_client
    ctx.__exit__.return_value = False

    with patch.object(backend_client, "_auth_client", return_value=ctx):
        result = backend_client.place_order("tok-123", payment_method="card")

    # Checkout called with the cart's items.
    fake_client.post.assert_called_once_with(
        "/api/orders/checkout/",
        json={"items": [{"book_id": "b1", "quantity": 2}], "payment_method": "card"},
    )
    assert result["order_id"] == "o1"
    assert result["status"] == "confirmed"


def test_place_order_empty_cart_returns_error():
    cart_body = {"status": {"success": True}, "data": {"items": [], "total_quantity": 0}}
    patcher, _ = _patch_auth_client(_mock_response(cart_body))
    with patcher:
        result = backend_client.place_order("tok-123")
    assert "empty" in result["error"].lower()


def test_add_to_cart_rejects_non_uuid_book_id():
    """A title or malformed id returns a clear error without hitting the API."""
    # No _auth_client patch needed — it should bail before any HTTP call.
    result = backend_client.add_to_cart("tok-123", book_id="The Hobbit", quantity=1)
    assert "error" in result
    assert "search_books" in result["error"]
