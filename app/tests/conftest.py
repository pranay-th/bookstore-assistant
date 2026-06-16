"""
tests/conftest.py — shared pytest fixtures.

Configures the JWT secret so the auth dependency can verify the test tokens
forged by ``helpers.make_access_token``.
"""
import pytest

from app.core.config import settings
from app.tests.helpers import TEST_JWT_SECRET


@pytest.fixture(autouse=True)
def _configure_auth():
    """Point the app's JWT settings at the shared test secret for every test.

    autouse so endpoint tests authenticate without extra wiring. Original
    values are restored afterwards.
    """
    original_secret = settings.JWT_SECRET
    original_require = settings.REQUIRE_AUTH
    settings.JWT_SECRET = TEST_JWT_SECRET
    settings.REQUIRE_AUTH = True
    yield
    settings.JWT_SECRET = original_secret
    settings.REQUIRE_AUTH = original_require
