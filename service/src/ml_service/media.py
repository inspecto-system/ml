"""Reading the photo to classify from media-service (``preview`` variant, scope ``photos.read``)."""

from uuid import UUID

import httpx

from ml_service.auth import ServiceTokenSource


class PhotoNotFoundError(Exception):
    """media-service has no such photo."""


class MediaUnavailableError(Exception):
    """media-service did not answer or answered with an error."""


class MediaClient:
    """Downloads photo bytes with our service token; renews the token once on ``401``."""

    def __init__(self, http: httpx.Client, media_url: str, tokens: ServiceTokenSource) -> None:
        self._http = http
        self._media_url = media_url
        self._tokens = tokens

    def preview(self, photo_id: UUID) -> bytes:
        """Bytes of the ``preview`` variant of the photo."""
        try:
            response = self._get(photo_id)
            if response.status_code == httpx.codes.UNAUTHORIZED:
                self._tokens.invalidate()
                response = self._get(photo_id)
        except httpx.HTTPError as error:
            raise MediaUnavailableError(str(error)) from error
        if response.status_code == httpx.codes.NOT_FOUND:
            raise PhotoNotFoundError(str(photo_id))
        if response.status_code != httpx.codes.OK:
            msg = f"media-service answered {response.status_code}"
            raise MediaUnavailableError(msg)
        return response.content

    def _get(self, photo_id: UUID) -> httpx.Response:
        return self._http.get(
            f"{self._media_url}/internal/v1/photos/{photo_id}/content",
            params={"variant": "preview"},
            headers={"Authorization": f"Bearer {self._tokens.token()}"},
        )
