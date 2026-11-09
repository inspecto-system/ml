"""Configuration from ``ML_*`` environment variables (names follow media-service)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

MIN_SECRET_LENGTH = 32


class SettingsError(ValueError):
    """A required variable is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Everything ml-service needs to start."""

    port: int
    jwks_url: str
    jwt_issuer: str
    internal_audience: str
    core_url: str
    client_secret: str
    media_url: str
    model_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Read the settings; the secret is required and must be at least 32 characters."""
        values = os.environ if env is None else env
        secret = values.get("ML_CORE_CLIENT_SECRET", "")
        if len(secret) < MIN_SECRET_LENGTH:
            msg = f"ML_CORE_CLIENT_SECRET must be at least {MIN_SECRET_LENGTH} characters"
            raise SettingsError(msg)
        try:
            port = int(values.get("ML_PORT", "8084"))
        except ValueError as error:
            msg = "ML_PORT must be a number"
            raise SettingsError(msg) from error
        core = values.get("ML_CORE_URL", "http://127.0.0.1:8081").rstrip("/")
        return cls(
            port=port,
            jwks_url=values.get("ML_JWKS_URL", f"{core}/api/v1/.well-known/jwks.json"),
            jwt_issuer=values.get("ML_JWT_ISSUER", "inspecto-core"),
            internal_audience=values.get("ML_JWT_INTERNAL_AUDIENCE", "inspecto-internal"),
            core_url=core,
            client_secret=secret,
            media_url=values.get("ML_MEDIA_URL", "http://127.0.0.1:8082").rstrip("/"),
            model_dir=Path(values.get("ML_MODEL_DIR", "/models/current")),
        )
