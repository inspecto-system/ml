"""Image features from a frozen pretrained network (the linear-probe baseline of ML-03)."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn

Transform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class Backbone:
    """A network that maps a preprocessed batch to feature vectors, and its preprocessing."""

    name: str
    network: nn.Module
    transform: Transform


def resnet50() -> Backbone:
    """ResNet-50 with ImageNet weights (torchvision ``IMAGENET1K_V2``) without the classifier.

    The weights are downloaded once into ``TORCH_HOME``.
    """
    from torchvision.models import ResNet50_Weights  # noqa: PLC0415 — heavy import, used here only
    from torchvision.models import resnet50 as build  # noqa: PLC0415

    weights = ResNet50_Weights.IMAGENET1K_V2
    network = build(weights=weights)
    network.fc = nn.Identity()
    return Backbone("resnet50/IMAGENET1K_V2", network, weights.transforms())


@torch.inference_mode()
def extract(backbone: Backbone, images: Iterable[Image.Image], batch_size: int = 64) -> np.ndarray:
    """Feature matrix ``(N, D)`` of the images, computed in batches in evaluation mode."""
    backbone.network.eval()
    batches: list[np.ndarray] = []
    pending: list[torch.Tensor] = []

    def flush() -> None:
        output = backbone.network(torch.stack(pending))
        batches.append(output.flatten(1).numpy().astype(np.float32))
        pending.clear()

    for image in images:
        pending.append(backbone.transform(image))
        if len(pending) == batch_size:
            flush()
    if pending:
        flush()
    if not batches:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(batches)
