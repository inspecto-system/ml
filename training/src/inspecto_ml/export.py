"""Export of a classifier to ONNX for ml-service, with an equivalence check (ML_DOCUMENTATION §4).

The exported graph takes a normalised image batch ``(N, 3, S, S)`` and returns class
probabilities ``(N, C)``. Preprocessing (resize, centre crop, normalisation) is done by
ml-service with Pillow and NumPy as described in ``model.json``; ``preprocess`` here is the same
procedure and is checked against the torchvision transform used in training.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from PIL import Image
from torch import nn

from inspecto_ml.metrics import evaluate

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(20))  # 0.0 … 0.95


@dataclass(frozen=True)
class Preprocessing:
    """How an image becomes the network input: shorter side → ``resize``, centre crop ``crop``."""

    resize: int
    crop: int
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def preprocess(image: Image.Image, spec: Preprocessing) -> np.ndarray:
    """``(3, crop, crop)`` float32 array: bilinear resize of the shorter side, crop, normalise."""
    rgb = image.convert("RGB")
    short, long = sorted(rgb.size)
    other = int(spec.resize * long / short)  # truncation, as in torchvision.transforms.Resize
    size = (spec.resize, other) if rgb.width <= rgb.height else (other, spec.resize)
    resized = rgb.resize(size, Image.Resampling.BILINEAR)
    left = round((resized.width - spec.crop) / 2)
    top = round((resized.height - spec.crop) / 2)
    cropped = resized.crop((left, top, left + spec.crop, top + spec.crop))
    array = np.asarray(cropped, dtype=np.float32) / 255
    normalised = (array - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(
        spec.std, dtype=np.float32
    )
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))


class LinearProbe(nn.Module):
    """Frozen backbone, standardisation and a linear layer with softmax — the ML-03 baseline."""

    def __init__(
        self,
        backbone: nn.Module,
        mean: np.ndarray,
        scale: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(weight.astype(np.float32)))
            self.linear.bias.copy_(torch.from_numpy(bias.astype(np.float32)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Class probabilities of a normalised image batch."""
        features = self.backbone(images).flatten(1)
        logits = self.linear((features - self.mean) / self.scale)
        return torch.softmax(logits, dim=1)


def export_onnx(model: nn.Module, path: Path, crop: int) -> None:
    """Write the model as ONNX with a dynamic batch dimension."""
    model.eval()
    example = torch.zeros(1, 3, crop, crop)
    batch = torch.export.Dim("batch", min=1, max=64)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_shapes={"images": {0: batch}},
        external_data=False,
    )


@dataclass(frozen=True)
class Equivalence:
    """Agreement of PyTorch and ONNX Runtime on the same inputs."""

    images: int
    max_abs_difference: float
    top1_agreement: float


@torch.inference_mode()
def check_equivalence(
    model: nn.Module, path: Path, torch_batch: np.ndarray, onnx_batch: np.ndarray
) -> Equivalence:
    """Compare PyTorch on ``torch_batch`` with ONNX Runtime on ``onnx_batch`` (same images).

    The batches may come from different preprocessing implementations: the check then covers
    both the export and the preprocessing of ml-service.
    """
    model.eval()
    expected = model(torch.from_numpy(torch_batch)).numpy()
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {"image": onnx_batch})[0])
    return Equivalence(
        images=len(onnx_batch),
        max_abs_difference=float(np.max(np.abs(expected - actual))),
        top1_agreement=float(np.mean(expected.argmax(1) == actual.argmax(1))),
    )


def apply_threshold(
    probabilities: np.ndarray, classes: Sequence[str], threshold: float, no_defect: str
) -> np.ndarray:
    """Probabilities as ml-service answers: a top category below the threshold → "no defect"."""
    adjusted = probabilities.copy()
    rows = np.max(probabilities, axis=1) < threshold
    adjusted[rows] = 0.0
    adjusted[rows, list(classes).index(no_defect)] = 1.0
    return adjusted


def choose_threshold(
    probabilities: np.ndarray, truth: Sequence[str], classes: Sequence[str], no_defect: str
) -> tuple[float, dict[str, float]]:
    """Threshold with the best validation macro-F1 (ties → the smaller one) and the whole search."""
    search = {
        str(threshold): float(
            evaluate(truth, apply_threshold(probabilities, classes, threshold, no_defect), classes)[
                "macro_f1"
            ]
        )
        for threshold in THRESHOLDS
    }
    best = max(THRESHOLDS, key=lambda threshold: (search[str(threshold)], -threshold))
    return best, search
