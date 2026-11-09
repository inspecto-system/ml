import uuid

import httpx
import pytest

from ml_service.auth import InvalidTokenError, ServiceTokenSource
from ml_service.media import MediaClient, MediaUnavailableError, PhotoNotFoundError
from tests.support import Issuer, fake_services

PHOTO = uuid.UUID("00000000-0000-0000-0000-000000000001")


def test_accepts_a_classification_token_of_core() -> None:
    issuer = Issuer()

    claims = issuer.verifier().verify(f"Bearer {issuer.token()}")

    assert claims["sub"] == "core-service"


@pytest.mark.parametrize(
    "overrides",
    [
        {"sub": "report-service"},
        {"scope": "photos.read"},
        {"aud": "inspecto-api"},
        {"iss": "someone"},
        {"exp": 1},
    ],
    ids=["other service", "no scope", "user audience", "other issuer", "expired"],
)
def test_rejects_other_tokens(overrides: dict[str, object]) -> None:
    issuer = Issuer()

    with pytest.raises(InvalidTokenError):
        issuer.verifier().verify(f"Bearer {issuer.token(**overrides)}")


@pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer ", "Bearer not.a.jwt"])
def test_rejects_missing_or_malformed_header(header: str | None) -> None:
    with pytest.raises(InvalidTokenError):
        Issuer().verifier().verify(header)


def test_rejects_a_token_signed_by_another_key() -> None:
    with pytest.raises(InvalidTokenError):
        Issuer().verifier().verify(f"Bearer {Issuer().token()}")


def test_own_token_is_cached_until_shortly_before_expiry() -> None:
    calls = []

    def core(request: httpx.Request) -> httpx.Response:
        calls.append(request.content.decode())
        return httpx.Response(200, json={"access_token": f"t{len(calls)}", "expires_in": 120})

    now = [0.0]
    tokens = ServiceTokenSource(
        httpx.Client(transport=httpx.MockTransport(core)), "http://core", "s" * 32, lambda: now[0]
    )

    first, again = tokens.token(), tokens.token()
    now[0] = 61.0  # 120 s minus the 60 s renewal margin have passed
    renewed = tokens.token()

    assert (first, again, renewed) == ("t1", "t1", "t2")
    assert "client_id=ml-service" in calls[0]
    assert "grant_type=client_credentials" in calls[0]


def _client(media: httpx.Client) -> MediaClient:
    return MediaClient(media, "http://media", ServiceTokenSource(media, "http://core", "s" * 32))


def test_reads_the_preview_with_own_token() -> None:
    seen = []

    def media(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"jpeg")

    assert _client(fake_services(media)).preview(PHOTO) == b"jpeg"
    assert seen[0].url.path == f"/internal/v1/photos/{PHOTO}/content"
    assert seen[0].url.params["variant"] == "preview"
    assert seen[0].headers["Authorization"] == "Bearer own-token"


def test_retries_once_with_a_new_token_after_401() -> None:
    answers = iter([httpx.Response(401), httpx.Response(200, content=b"jpeg")])

    assert _client(fake_services(lambda _: next(answers))).preview(PHOTO) == b"jpeg"


@pytest.mark.parametrize(
    ("answer", "error"),
    [(404, PhotoNotFoundError), (500, MediaUnavailableError), (401, MediaUnavailableError)],
)
def test_media_errors(answer: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        _client(fake_services(lambda _: httpx.Response(answer))).preview(PHOTO)


def test_network_failure_is_unavailability() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(MediaUnavailableError):
        _client(fake_services(broken)).preview(PHOTO)
