from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import ResNet50_Weights

from inspecto_ml.export import (
    LinearProbe,
    Preprocessing,
    apply_threshold,
    check_equivalence,
    choose_threshold,
    export_onnx,
    preprocess,
)
from tests.audit.helpers import make_picture

CLASSES = ["CRACK", "NO_DEFECT", "STAIN"]


@pytest.mark.parametrize("size", [(64, 48), (48, 64), (300, 300), (257, 513)])
def test_pillow_preprocessing_matches_torchvision(size: tuple[int, int]) -> None:
    image = make_picture(3).resize(size)
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()  # no download: preprocessing only

    ours = preprocess(image, Preprocessing(resize=232, crop=224))
    theirs = transform(image).numpy()

    assert ours.shape == theirs.shape == (3, 224, 224)
    assert np.max(np.abs(ours - theirs)) < 1e-5


def _probe() -> LinearProbe:
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1))
    rng = np.random.default_rng(0)
    return LinearProbe(
        backbone,
        mean=rng.normal(size=4),
        scale=rng.uniform(0.5, 2, size=4),
        weight=rng.normal(size=(3, 4)),
        bias=rng.normal(size=3),
    )


def test_exported_model_is_equivalent_to_pytorch(tmp_path: Path) -> None:
    model = _probe()
    path = tmp_path / "model.onnx"
    export_onnx(model, path, crop=16)
    batch = np.random.default_rng(1).normal(size=(5, 3, 16, 16)).astype(np.float32)

    result = check_equivalence(model, path, batch, batch)

    assert result.images == 5
    assert result.max_abs_difference < 1e-5
    assert result.top1_agreement == 1.0


def test_linear_probe_outputs_probabilities() -> None:
    output = _probe()(torch.zeros(2, 3, 16, 16))

    assert output.shape == (2, 3)
    assert torch.allclose(output.sum(1), torch.ones(2))


def test_low_confidence_becomes_no_defect() -> None:
    probabilities = np.array([[0.6, 0.1, 0.3], [0.4, 0.3, 0.3]])

    adjusted = apply_threshold(probabilities, CLASSES, threshold=0.5, no_defect="NO_DEFECT")

    assert adjusted.tolist() == [[0.6, 0.1, 0.3], [0.0, 1.0, 0.0]]


def test_choose_threshold_prefers_the_best_then_the_smallest() -> None:
    truth = ["CRACK", "NO_DEFECT", "STAIN"]
    probabilities = np.array([[0.9, 0.05, 0.05], [0.42, 0.4, 0.18], [0.1, 0.1, 0.8]])

    threshold, search = choose_threshold(probabilities, truth, CLASSES, "NO_DEFECT")

    assert search["0.0"] < search["0.45"] == 1.0
    assert threshold == 0.45
    assert len(search) == 20
