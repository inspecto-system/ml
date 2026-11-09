import numpy as np
import torch
from PIL import Image
from torch import nn

from inspecto_ml.features import Backbone


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def tiny_backbone() -> Backbone:
    """Mean colour of the image as a 3-number feature: no weights to download."""
    return Backbone("tiny", nn.AdaptiveAvgPool2d(1), _to_tensor)
