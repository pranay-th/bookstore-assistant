"""
core/auth.py — JWT authentication against the Django backend's tokens.

The Django backend issues access tokens with djangorestframework-simplejwt:
    - algorithm: HS256
    - signing key: Django SECRET_KEY
    - sent as: ``Authorization: Bearer <token>``
    - claims: ``user_id`` (USER_ID_CLAIM), ``token_type`` ("access"),
      ``exp``, ``iat``, ``jti``.

We verify those tokens statelessly here by sharing the same secret — no extra
network round trip to the backend. This module exposes:

    - ``AuthenticatedUser``: the decoded identity attached to a request.
    - ``require_user``: FastAPI dependency that 401s without a valid token.
    - ``optional_user``: dependency that returns the user if present, else None.

Set ``REQUIRE_AUTH=False`` in the environment to make auth optional (local dev).
"""
import logging
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

logger = logging.getLogger(__name__)

# auto_error=False so we can craft our own 401s and support optional auth.
# scheme_name="jwtAuth" mirrors the Django backend's Swagger (drf-spectacular
# names the simplejwt scheme "jwtAuth"), so both API docs look identical.
_bearer = HTTPBearer(scheme_name="jwtAuth", auto_error=False)


@dataclass
class AuthenticatedUser:
    """The identity decoded from a verified access token."""

    user_id: str
    token_type: Optional[str] = None
    raw_claims: Optional[dict] = None
    # The raw access token, kept so the agent can forward it to the Django
    # backend for user-scoped actions (cart, orders). Both services share the
    # same SECRET_KEY, so this token is valid against Django too.
    access_token: Optional[str] = None


class AuthError(HTTPException):
    """401 with a WWW-Authenticate header, as expected for bearer auth."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def decode_token(token: str) -> AuthenticatedUser:
    """
    Verify and decode a backend-issued access token.

    Raises:
        AuthError: if the secret is unconfigured, the signature/expiry is
            invalid, the token isn't an access token, or the user_id claim
            is missing.
    """
    if not settings.JWT_SECRET:
        # Misconfiguration — fail closed rather than accepting unverified tokens.
        logger.error("JWT_SECRET is not set — cannot verify access tokens.")
        raise AuthError("Authentication is not configured on the server.")

    try:
        claims = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        # Covers bad signature, malformed token, wrong algorithm, etc.
        logger.info("Rejected invalid JWT: %s", exc)
        raise AuthError("Invalid authentication token.") from exc

    # simplejwt tags access tokens with token_type="access". Reject refresh
    # tokens (or anything else) being used as a bearer credential.
    token_type = claims.get("token_type")
    if token_type and token_type != "access":
        raise AuthError("Invalid token type — an access token is required.")

    user_id = claims.get(settings.JWT_USER_ID_CLAIM)
    if not user_id:
        raise AuthError("Token is missing the user identity claim.")

    return AuthenticatedUser(
        user_id=str(user_id),
        token_type=token_type,
        raw_claims=claims,
        access_token=token,
    )


def require_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthenticatedUser:
    """
    FastAPI dependency: require a valid Bearer access token.

    With ``REQUIRE_AUTH=False`` a missing token is allowed (returns an
    anonymous user); a *present* token is still verified so a bad token always
    fails. With ``REQUIRE_AUTH=True`` a token is mandatory.
    """
    if credentials is None or not credentials.credentials:
        if settings.REQUIRE_AUTH:
            raise AuthError("Authentication credentials were not provided.")
        return AuthenticatedUser(user_id="anonymous")

    return decode_token(credentials.credentials)


def optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[AuthenticatedUser]:
    """
    FastAPI dependency: decode the user if a token is present, else return None.

    A present-but-invalid token still raises (so clients get clear feedback),
    but no token simply yields ``None``.
    """
    if credentials is None or not credentials.credentials:
        return None
    return decode_token(credentials.credentials)
