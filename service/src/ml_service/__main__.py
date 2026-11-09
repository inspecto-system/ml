"""Entry point: ``python -m ml_service`` — reads ``ML_*`` settings and serves on ``ML_PORT``."""

import logging

import httpx
import uvicorn

from ml_service.app import Dependencies, create_app
from ml_service.auth import ServiceTokenSource, TokenVerifier, jwks_resolver
from ml_service.classifier import Classifier
from ml_service.media import MediaClient
from ml_service.settings import Settings

# media-service answers in up to 10 s for image bytes (API_DOCUMENTATION.md); core waits 3 s for
# the whole classification, so a shorter limit fails fast instead of working for nobody.
MEDIA_TIMEOUT_SECONDS = 2.5

log = logging.getLogger("ml_service")


def _load_classifier(settings: Settings) -> Classifier | None:
    try:
        classifier = Classifier(settings.model_dir)
    except (OSError, ValueError, KeyError) as error:
        log.error("model not loaded from %s: %s", settings.model_dir, error)  # noqa: TRY400 — no traceback needed
        return None
    log.info("model %s loaded", classifier.info.version)
    return classifier


def main() -> None:
    """Start the HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    http = httpx.Client(timeout=MEDIA_TIMEOUT_SECONDS)
    tokens = ServiceTokenSource(http, settings.core_url, settings.client_secret)
    dependencies = Dependencies(
        verifier=TokenVerifier(
            jwks_resolver(settings.jwks_url), settings.jwt_issuer, settings.internal_audience
        ),
        media=MediaClient(http, settings.media_url, tokens),
        classifier=_load_classifier(settings),
    )
    uvicorn.run(create_app(dependencies), host="0.0.0.0", port=settings.port)  # noqa: S104 — container port


if __name__ == "__main__":
    main()
