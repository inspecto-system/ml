"""The ONNX model and its answer: up to three categories, or nothing (ml.yaml 0.5.0).

``model.json`` next to ``model.onnx`` describes the model (written by ``inspecto_ml.export_cli``):
version, classes, the "no defect" class, preprocessing and the confidence threshold.
Preprocessing repeats ``inspecto_ml.export.preprocess``; the export checks it against training.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
from PIL import Image

TOP_K = 3


class InvalidImageError(ValueError):
    """The photo bytes are not a decodable image."""


@dataclass(frozen=True)
class Prediction:
    """A category and its probability."""

    code: str
    probability: float


@dataclass(frozen=True)
class ModelInfo:
    """What ``GET /internal/v1/model`` reports and how to preprocess an image."""

    version: str
    classes: tuple[str, ...]
    no_defect: str
    input_size: int
    resize: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    threshold: float

    @classmethod
    def parse(cls, document: dict[str, Any]) -> "ModelInfo":
        """From the content of ``model.json``."""
        preprocessing = document["preprocessing"]
        return cls(
            version=str(document["modelVersion"]),
            classes=tuple(document["classes"]),
            no_defect=str(document["noDefectClass"]),
            input_size=int(document["inputSize"]),
            resize=int(preprocessing["resize"]),
            mean=tuple(preprocessing["mean"]),
            std=tuple(preprocessing["std"]),
            threshold=float(document["threshold"]),
        )


def preprocess(image: Image.Image, info: ModelInfo) -> np.ndarray:
    """``(3, S, S)`` float32: bilinear resize of the shorter side, centre crop, normalisation."""
    rgb = image.convert("RGB")
    short, long = sorted(rgb.size)
    other = int(info.resize * long / short)  # truncation, as in torchvision.transforms.Resize
    size = (info.resize, other) if rgb.width <= rgb.height else (other, info.resize)
    resized = rgb.resize(size, Image.Resampling.BILINEAR)
    crop = info.input_size
    left = round((resized.width - crop) / 2)
    top = round((resized.height - crop) / 2)
    array = np.asarray(resized.crop((left, top, left + crop, top + crop)), dtype=np.float32) / 255
    mean = np.asarray(info.mean, dtype=np.float32)
    std = np.asarray(info.std, dtype=np.float32)
    return np.ascontiguousarray(((array - mean) / std).transpose(2, 0, 1))


def suggestions(probabilities: np.ndarray, info: ModelInfo) -> list[Prediction]:
    """Up to three categories; none if "no defect" wins or the best category is below threshold."""
    order = np.argsort(-probabilities)
    best = info.classes[int(order[0])]
    if best == info.no_defect or float(probabilities[order[0]]) < info.threshold:
        return []
    ranked = [
        Prediction(info.classes[int(index)], round(float(probabilities[index]), 4))
        for index in order
        if info.classes[int(index)] != info.no_defect
    ]
    return ranked[:TOP_K]


class Classifier:
    """ONNX Runtime session on CPU and the model description."""

    def __init__(self, directory: Path) -> None:
        document = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        self.info = ModelInfo.parse(document)
        self._session = onnxruntime.InferenceSession(
            str(directory / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def classify(self, content: bytes) -> list[Prediction]:
        """Suggestions for the image bytes."""
        try:
            with Image.open(io.BytesIO(content)) as image:
                batch = preprocess(image, self.info)[np.newaxis]
        except (OSError, ValueError) as error:
            raise InvalidImageError(str(error)) from error
        probabilities = np.asarray(self._session.run(None, {self._input: batch})[0])[0]
        return suggestions(probabilities, self.info)
