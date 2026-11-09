"""Test doubles: an RSA issuer like core-service, a tiny ONNX model, a fake media-service."""

import io
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import jwt
import numpy as np
import onnx
from cryptography.hazmat.primitives.asymmetric import rsa
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from ml_service.auth import TokenVerifier

ISSUER = "inspecto-core"
AUDIENCE = "inspecto-internal"
CLASSES = ["CRACK", "NO_DEFECT", "STAIN"]


class Issuer:
    """Signs tokens like core-service; ``verifier`` trusts only its key."""

    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def token(self, **overrides: Any) -> str:  # noqa: ANN401 — any claim value
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "core-service",
            "scope": "ml.classify",
            "exp": int(time.time()) + 300,
        } | overrides
        return jwt.encode(claims, self.key, algorithm="RS256")

    def verifier(self) -> TokenVerifier:
        public = self.key.public_key()
        return TokenVerifier(lambda _: public, ISSUER, AUDIENCE)


def write_model(directory: Path, threshold: float = 0.5) -> Path:
    """A model that answers by the dominant colour: red → CRACK, green → NO_DEFECT, blue → STAIN.

    Graph: mean over pixels (N, 3), times 20, softmax; preprocessing without normalisation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    weight = numpy_helper.from_array(np.eye(3, dtype=np.float32) * 20, "weight")
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes")
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["image", "axes"], ["mean"], keepdims=0),
            helper.make_node("MatMul", ["mean", "weight"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "colour",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 8, 8])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 3])],
        [weight, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, str(directory / "model.onnx"))
    info = {
        "modelVersion": "colour-test",
        "classes": CLASSES,
        "noDefectClass": "NO_DEFECT",
        "inputSize": 8,
        "preprocessing": {"resize": 10, "crop": 8, "mean": [0, 0, 0], "std": [1, 1, 1]},
        "threshold": threshold,
    }
    (directory / "model.json").write_text(json.dumps(info), encoding="utf-8")
    return directory


def jpeg(colour: tuple[int, int, int], size: tuple[int, int] = (40, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


Handler = Callable[[httpx.Request], httpx.Response]


def token_endpoint(request: httpx.Request) -> httpx.Response | None:
    """Answer of core's service-token endpoint, or ``None`` for other requests."""
    if request.url.path.endswith("/internal/v1/auth/service-token"):
        return httpx.Response(200, json={"access_token": "own-token", "expires_in": 900})
    return None


def fake_services(media: Handler) -> httpx.Client:
    """An HTTP client whose requests go to a fake core (token) and the given media handler."""

    def route(request: httpx.Request) -> httpx.Response:
        return token_endpoint(request) or media(request)

    return httpx.Client(transport=httpx.MockTransport(route))
