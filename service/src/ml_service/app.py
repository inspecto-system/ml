"""HTTP API of ml-service (ml.yaml 0.5.0) with RFC 9457 errors like the other services."""

import logging
import re
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from ml_service.auth import InvalidTokenError, TokenVerifier
from ml_service.classifier import Classifier, InvalidImageError
from ml_service.media import MediaClient, MediaUnavailableError, PhotoNotFoundError

REQUEST_ID_HEADER = "X-Request-Id"
PROBLEM_BASE = "https://inspecto.dev/problems/"
VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

log = logging.getLogger("ml_service")


class ClassifyRequest(BaseModel):
    """``ClassifyRequest`` of ml.yaml."""

    model_config = ConfigDict(extra="forbid")
    photoId: uuid.UUID  # noqa: N815 — field name of the contract


@dataclass
class Dependencies:
    """Collaborators of the API; ``classifier`` is ``None`` while the model is not loaded."""

    verifier: TokenVerifier
    media: MediaClient
    classifier: Classifier | None


class ProblemError(Exception):
    """An error answered as ``application/problem+json``."""

    def __init__(self, status: HTTPStatus, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def _request_id(request: Request) -> str:
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    return incoming if VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex


def _problem(request: Request, status: HTTPStatus, code: str, detail: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", _request_id(request))
    body = {
        "type": PROBLEM_BASE + code.lower().replace("_", "-"),
        "title": status.phrase,
        "status": status.value,
        "code": code,
        "detail": detail,
        "requestId": request_id,
    }
    headers = {REQUEST_ID_HEADER: request_id}
    return JSONResponse(body, status.value, headers, media_type="application/problem+json")


def _install_errors(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id(request: Request, call_next: Any) -> Any:  # noqa: ANN401 — ASGI types
        request.state.request_id = _request_id(request)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

    @app.exception_handler(ProblemError)
    async def problem(request: Request, error: ProblemError) -> JSONResponse:
        return _problem(request, error.status, error.code, error.detail)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, _: RequestValidationError) -> JSONResponse:
        return _problem(request, HTTPStatus.BAD_REQUEST, "VALIDATION_ERROR", "invalid request body")


class _Endpoints:
    """Handlers of ml.yaml bound to the dependencies."""

    def __init__(self, dependencies: Dependencies) -> None:
        self._dependencies = dependencies

    def authorised(self, authorization: Annotated[str | None, Header()] = None) -> None:
        try:
            self._dependencies.verifier.verify(authorization)
        except InvalidTokenError as error:
            log.info("rejected token: %s", error)
            raise ProblemError(
                HTTPStatus.UNAUTHORIZED, "UNAUTHENTICATED", "invalid token"
            ) from error

    def classifier(self) -> Classifier:
        if self._dependencies.classifier is None:
            raise ProblemError(HTTPStatus.SERVICE_UNAVAILABLE, "DEPENDENCY_UNAVAILABLE", "no model")
        return self._dependencies.classifier

    def classify(self, body: ClassifyRequest) -> dict[str, Any]:
        model = self.classifier()
        try:
            content = self._dependencies.media.preview(body.photoId)
            predictions = model.classify(content)
        except PhotoNotFoundError as error:
            raise ProblemError(HTTPStatus.NOT_FOUND, "NOT_FOUND", "photo not found") from error
        except (MediaUnavailableError, InvalidImageError) as error:
            log.warning("classification failed: %s", error)
            raise ProblemError(
                HTTPStatus.SERVICE_UNAVAILABLE, "DEPENDENCY_UNAVAILABLE", "photo unavailable"
            ) from error
        return {
            "modelVersion": model.info.version,
            "predictions": [{"code": p.code, "probability": p.probability} for p in predictions],
        }

    def model_info(self) -> dict[str, Any]:
        info = self.classifier().info
        return {
            "modelVersion": info.version,
            "classes": list(info.classes),
            "inputSize": info.input_size,
            "threshold": info.threshold,
        }

    def ready(self) -> dict[str, str]:
        self.classifier()
        return {"status": "UP"}


def _live() -> dict[str, str]:
    return {"status": "UP"}


def create_app(dependencies: Dependencies) -> FastAPI:
    """The FastAPI application; no OpenAPI pages — the contract lives in platform/api."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    _install_errors(app)
    endpoints = _Endpoints(dependencies)
    authorised = [Depends(endpoints.authorised)]
    app.add_api_route(
        "/internal/v1/classify", endpoints.classify, methods=["POST"], dependencies=authorised
    )
    app.add_api_route(
        "/internal/v1/model", endpoints.model_info, methods=["GET"], dependencies=authorised
    )
    app.add_api_route("/health/live", _live, methods=["GET"])
    app.add_api_route("/health/ready", endpoints.ready, methods=["GET"])
    return app
