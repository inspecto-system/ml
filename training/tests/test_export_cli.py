import json
from pathlib import Path

import pytest
from torch import nn
from torchvision.models import ResNet50_Weights

from inspecto_ml import baseline_cli, export_cli
from inspecto_ml.features import Backbone
from inspecto_ml.split_cli import main as split_main
from tests.test_baseline import _archive, _mapping


def _small_backbone() -> Backbone:
    """A tiny network with the real ResNet-50 preprocessing (which needs no download)."""
    network = nn.Sequential(nn.Conv2d(3, 6, 5, stride=4), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
    return Backbone("small", network, ResNet50_Weights.IMAGENET1K_V2.transforms())


def test_command_line_exports_an_equivalent_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backbone = _small_backbone()  # one instance: the same weights for features and export
    monkeypatch.setattr(baseline_cli, "resnet50", lambda: backbone)
    monkeypatch.setattr(export_cli, "resnet50", lambda: backbone)
    bd3 = _archive(tmp_path)
    split = tmp_path / "split.json"
    split_main(["--bd3", str(bd3), "--mapping", str(_mapping(tmp_path)), "--out", str(split)])
    common = ["--bd3", str(bd3), "--split", str(split), "--features", str(tmp_path / "f")]
    baseline_cli.main([*common, "--out", str(tmp_path / "run.json")])

    export_cli.main([*common, "--run", str(tmp_path / "run.json"), "--out-dir", str(tmp_path / "m")])

    info = json.loads((tmp_path / "m" / "model.json").read_text(encoding="utf-8"))
    assert (tmp_path / "m" / "model.onnx").stat().st_size > 0
    assert info["classes"] == ["CRACK", "NO_DEFECT", "STAIN"]
    assert info["preprocessing"]["resize"] == 232
    assert info["inputSize"] == 224
    assert info["equivalence"]["max_abs_difference"] < 1e-3
    assert info["equivalence"]["top1_agreement"] == 1.0
    assert 0.0 <= info["threshold"] < 1.0
    assert info["modelVersion"].startswith("resnet50-logreg-")
