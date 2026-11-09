from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ml_service.app import Dependencies, create_app
from ml_service.auth import ServiceTokenSource
from ml_service.classifier import Classifier
from ml_service.media import MediaClient
from ml_service.settings import Settings, SettingsError
from tests.support import Issuer, fake_services, jpeg, write_model

PHOTO = "00000000-0000-0000-0000-000000000001"


def _app(
    tmp_path: Path, media_answer: httpx.Response, *, model: bool = True
) -> tuple[TestClient, Issuer]:
    issuer = Issuer()
    http = fake_services(lambda _: media_answer)
    media = MediaClient(http, "http://media", ServiceTokenSource(http, "http://core", "s" * 32))
    classifier = Classifier(write_model(tmp_path)) if model else None
    app = create_app(Dependencies(issuer.verifier(), media, classifier))
    return TestClient(app), issuer


def _auth(issuer: Issuer, **claims: object) -> dict[str, str]:
    return {"Authorization": f"Bearer {issuer.token(**claims)}"}


def test_classify_returns_predictions_and_model_version(tmp_path: Path) -> None:
    client, issuer = _app(tmp_path, httpx.Response(200, content=jpeg((10, 10, 250))))

    response = client.post("/internal/v1/classify", json={"photoId": PHOTO}, headers=_auth(issuer))

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "colour-test"
    assert [p["code"] for p in body["predictions"]] == ["STAIN", "CRACK"]
    assert response.headers["X-Request-Id"]


def test_no_defect_gives_an_empty_list(tmp_path: Path) -> None:
    client, issuer = _app(tmp_path, httpx.Response(200, content=jpeg((10, 250, 10))))

    response = client.post("/internal/v1/classify", json={"photoId": PHOTO}, headers=_auth(issuer))

    assert response.json()["predictions"] == []


def test_rejects_tokens_of_other_services(tmp_path: Path) -> None:
    client, issuer = _app(tmp_path, httpx.Response(200, content=jpeg((250, 0, 0))))

    response = client.post(
        "/internal/v1/classify",
        json={"photoId": PHOTO},
        headers=_auth(issuer, sub="report-service") | {"X-Request-Id": "abc-1"},
    )

    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "https://inspecto.dev/problems/unauthenticated",
        "title": "Unauthorized",
        "status": 401,
        "code": "UNAUTHENTICATED",
        "detail": "invalid token",
        "requestId": "abc-1",
    }


@pytest.mark.parametrize("body", [{}, {"photoId": "nope"}, {"photoId": PHOTO, "extra": 1}])
def test_validates_the_request(tmp_path: Path, body: dict[str, object]) -> None:
    client, issuer = _app(tmp_path, httpx.Response(200))

    response = client.post("/internal/v1/classify", json=body, headers=_auth(issuer))

    assert response.status_code == 400
    assert response.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    ("media", "status", "code"),
    [
        (httpx.Response(404), 404, "NOT_FOUND"),
        (httpx.Response(503), 503, "DEPENDENCY_UNAVAILABLE"),
        (httpx.Response(200, content=b"broken"), 503, "DEPENDENCY_UNAVAILABLE"),
    ],
    ids=["no photo", "media down", "not an image"],
)
def test_classify_errors(tmp_path: Path, media: httpx.Response, status: int, code: str) -> None:
    client, issuer = _app(tmp_path, media)

    response = client.post("/internal/v1/classify", json={"photoId": PHOTO}, headers=_auth(issuer))

    assert response.status_code == status
    assert response.json()["code"] == code


def test_model_info(tmp_path: Path) -> None:
    client, issuer = _app(tmp_path, httpx.Response(200))

    response = client.get("/internal/v1/model", headers=_auth(issuer))

    assert response.json() == {
        "modelVersion": "colour-test",
        "classes": ["CRACK", "NO_DEFECT", "STAIN"],
        "inputSize": 8,
        "threshold": 0.5,
    }


def test_without_a_model_the_service_is_live_but_not_ready(tmp_path: Path) -> None:
    client, issuer = _app(tmp_path, httpx.Response(200), model=False)

    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    response = client.post("/internal/v1/classify", json={"photoId": PHOTO}, headers=_auth(issuer))
    assert response.status_code == 503


def test_ready_with_a_model(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, httpx.Response(200))

    assert client.get("/health/ready").json() == {"status": "UP"}


def test_settings_from_environment() -> None:
    settings = Settings.from_env(
        {"ML_CORE_CLIENT_SECRET": "x" * 32, "ML_CORE_URL": "http://core:8081/"}
    )

    assert settings.jwks_url == "http://core:8081/api/v1/.well-known/jwks.json"
    assert settings.port == 8084
    assert settings.internal_audience == "inspecto-internal"


@pytest.mark.parametrize(
    "env",
    [{}, {"ML_CORE_CLIENT_SECRET": "short"}, {"ML_CORE_CLIENT_SECRET": "x" * 32, "ML_PORT": "a"}],
)
def test_settings_errors(env: dict[str, str]) -> None:
    with pytest.raises(SettingsError):
        Settings.from_env(env)
