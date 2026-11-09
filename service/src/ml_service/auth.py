"""Service tokens: verifying the ones of core-service and obtaining our own (ADR-007).

Incoming tokens: RS256 only (no ``alg=none`` or algorithm substitution, SECURITY.md §2), key from
the JWKS of core, issuer, audience ``inspecto-internal``, subject ``core-service`` and scope
``ml.classify``. Our token: client credentials at core, cached until shortly before expiry.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

CORE_CLIENT = "core-service"
CLASSIFY_SCOPE = "ml.classify"
OWN_CLIENT = "ml-service"
RENEW_MARGIN_SECONDS = 60

KeyResolver = Callable[[str], Any]


class InvalidTokenError(Exception):
    """The token is missing, malformed, expired, not signed by core or lacks the scope."""


def jwks_resolver(url: str) -> KeyResolver:
    """Signing key of a token from the JWKS of core (cached by PyJWT, refreshed on a new kid)."""
    client = jwt.PyJWKClient(url, cache_keys=True, lifespan=300, timeout=2)
    return lambda token: client.get_signing_key_from_jwt(token).key


@dataclass(frozen=True)
class TokenVerifier:
    """Checks that a bearer token is a service token of core-service for classification."""

    resolve_key: KeyResolver
    issuer: str
    audience: str

    def verify(self, authorization: str | None) -> dict[str, Any]:
        """Claims of a valid token from an ``Authorization: Bearer …`` header."""
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            msg = "bearer token required"
            raise InvalidTokenError(msg)
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                self.resolve_key(token),
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except (jwt.PyJWTError, ValueError) as error:
            raise InvalidTokenError(str(error)) from error
        if (
            claims["sub"] != CORE_CLIENT
            or CLASSIFY_SCOPE not in str(claims.get("scope", "")).split()
        ):
            msg = f"only {CORE_CLIENT} with scope {CLASSIFY_SCOPE} may call ml-service"
            raise InvalidTokenError(msg)
        return claims


class ServiceTokenSource:
    """Our own service token, obtained from core and cached until shortly before it expires."""

    def __init__(
        self,
        http: httpx.Client,
        core_url: str,
        secret: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = http
        self._url = f"{core_url}/internal/v1/auth/service-token"
        self._secret = secret
        self._clock = clock
        self._token = ""
        self._expires = 0.0

    def token(self) -> str:
        """A valid token, from the cache or a new one."""
        if self._token and self._clock() < self._expires:
            return self._token
        response = self._http.post(
            self._url,
            data={
                "grant_type": "client_credentials",
                "client_id": OWN_CLIENT,
                "client_secret": self._secret,
            },
        )
        response.raise_for_status()
        body = response.json()
        self._token = str(body["access_token"])
        self._expires = self._clock() + int(body["expires_in"]) - RENEW_MARGIN_SECONDS
        return self._token

    def invalidate(self) -> None:
        """Forget the cached token (core may have restarted with a new key)."""
        self._token = ""
