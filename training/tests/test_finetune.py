import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from inspecto_ml import finetune_cli
from inspecto_ml.datasets import DatasetPaths, dataset_sources
from inspecto_ml.finetune import (
    RECIPES,
    Normalisation,
    Recipe,
    Trainer,
    TrainingData,
    as_input,
    augment,
    predict,
    set_trainable,
    train_mode,
)
from inspecto_ml.imagecache import balanced_subset, cached_images, square
from inspecto_ml.manifest import load_manifest
from inspecto_ml.split import Entry, Part
from inspecto_ml.split_cli import main as split_main
from tests.test_baseline import _archive, _mapping

PLAIN = Normalisation((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_square_keeps_the_shorter_side_and_crops_the_centre() -> None:
    image = Image.new("RGB", (60, 30), (255, 0, 0))
    image.paste((0, 0, 255), (0, 0, 10, 30))  # a blue band on the left is cut off

    result = square(image, 20)

    assert result.shape == (20, 20, 3)
    assert result.dtype == np.uint8
    assert (result[..., 0] > 250).all()


def _entry(uid: int, target: str) -> Entry:
    return Entry("bd3", target, f"k{uid}", f"s{uid}", target, f"g{uid}", Part.TRAIN)


def test_balanced_subset_caps_every_class_and_keeps_order() -> None:
    entries = [_entry(i, "NO_DEFECT") for i in range(10)] + [
        _entry(i + 10, "MOLD") for i in range(3)
    ]

    subset = balanced_subset(entries, per_class=4, seed=1)

    assert sum(e.target == "NO_DEFECT" for e in subset) == 4
    assert sum(e.target == "MOLD" for e in subset) == 3
    assert subset == [e for e in entries if e in subset]
    assert subset == balanced_subset(entries, per_class=4, seed=1)


def test_augment_and_centre_crop_shapes() -> None:
    batch = np.full((2, 40, 40, 3), 128, dtype=np.uint8)

    augmented = augment(batch, np.random.default_rng(0), crop=32)
    centre = as_input(batch, crop=32)

    assert augmented.shape == centre.shape == (2, 3, 32, 32)
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0
    assert torch.allclose(centre, torch.full_like(centre, 128 / 255))


class _Tiny(nn.Module):
    """Mean colour → linear layer; ``classifier`` is the head."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(3, 3, 1)
        self.classifier = nn.Linear(3, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.classifier(self.body(images).mean(dim=(2, 3)))
        return out


def test_set_trainable_counts_only_matching_parameters() -> None:
    model = _Tiny(2)

    assert set_trainable(model, ("classifier",)) == 3 * 2 + 2
    assert not model.body.weight.requires_grad


def _coloured(rng: np.random.Generator, colour: int, count: int) -> np.ndarray:
    images = rng.integers(0, 40, size=(count, 40, 40, 3), dtype=np.uint8)
    images[..., colour] = 220
    return images


def test_trainer_learns_colours_and_keeps_the_best_epoch() -> None:
    rng = np.random.default_rng(0)
    images = np.concatenate([_coloured(rng, 0, 40), _coloured(rng, 2, 40)])
    data = TrainingData(
        images=images,
        labels=np.array([0] * 40 + [1] * 40),
        val_images=np.concatenate([_coloured(rng, 0, 10), _coloured(rng, 2, 10)]),
        val_targets=["RED"] * 10 + ["BLUE"] * 10,
        classes=["RED", "BLUE"],
    )
    recipe = Recipe(
        "tiny",
        ("classifier",),
        ("body", "classifier"),
        batch_size=8,
        head_lr=0.05,
        tune_lr=0.01,
        crop=32,
    )
    messages: list[str] = []
    model = _Tiny(2)

    history = Trainer(model, data, recipe, PLAIN, messages.append, seed=1).run()

    assert len(history.epochs) == 3
    assert history.best_val_macro_f1 == max(e["val_macro_f1"] for e in history.epochs)
    assert history.best_val_macro_f1 == pytest.approx(1.0)
    assert predict(model, data.val_images, PLAIN, crop=32).argmax(1).tolist() == [0] * 10 + [1] * 10
    assert messages[0].startswith("epoch 1:")


def test_cached_images_are_built_once_and_reused(tmp_path: Path) -> None:
    bd3 = _archive(tmp_path)
    split = tmp_path / "split.json"
    split_main(["--bd3", str(bd3), "--mapping", str(_mapping(tmp_path)), "--out", str(split)])
    entries = load_manifest(split).part(Part.TRAIN)[:5]
    sources = dataset_sources(DatasetPaths(bd3=bd3))
    path = tmp_path / "cache" / "train.npy"

    first = cached_images(entries, sources, 24, path)
    built = path.stat().st_mtime_ns
    again = cached_images(entries, sources, 24, path)

    assert first.shape == (5, 24, 24, 3)
    assert np.array_equal(first, again)
    assert path.stat().st_mtime_ns == built


def test_command_line_fine_tunes_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        RECIPES,
        "tiny",
        Recipe("tiny", ("classifier",), ("body", "classifier"), head_epochs=1, tune_epochs=1),
    )
    monkeypatch.setattr(
        finetune_cli, "create_model", lambda _name, classes: (_Tiny(classes), PLAIN)
    )
    monkeypatch.setattr(finetune_cli, "TRAIN_SIDE", 240)
    bd3 = _archive(tmp_path)
    split = tmp_path / "split.json"
    split_main(["--bd3", str(bd3), "--mapping", str(_mapping(tmp_path)), "--out", str(split)])
    out = tmp_path / "run.json"

    finetune_cli.main(
        [
            "--bd3", str(bd3), "--model", "tiny", "--split", str(split),
            "--cache", str(tmp_path / "cache"), "--per-class", "5",
            "--weights", str(tmp_path / "w.pt"), "--out", str(out),
        ]
    )  # fmt: skip

    run = json.loads(out.read_text(encoding="utf-8"))
    assert run["model"] == "tiny"
    assert run["train_images_per_class"] == {"CRACK": 5, "NO_DEFECT": 5, "STAIN": 5}
    assert len(run["epochs"]) == 2
    assert "macro_f1" in run["test"]
    assert (tmp_path / "w.pt").stat().st_size > 0


class _WithBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Conv2d(3, 4, 1)
        self.norm = nn.BatchNorm2d(4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.classifier(self.norm(self.body(images)).mean(dim=(2, 3)))
        return out


def test_train_mode_keeps_frozen_batch_norm_statistics() -> None:
    model = _WithBatchNorm()
    set_trainable(model, ("classifier",))
    before = model.norm.running_mean.clone()  # type: ignore[union-attr]  # a buffer of BatchNorm2d

    train_mode(model)
    model(torch.rand(4, 3, 5, 5))

    assert not model.norm.training
    assert model.classifier.training
    assert torch.equal(model.norm.running_mean, before)  # type: ignore[arg-type]
    set_trainable(model, ("norm", "classifier"))
    train_mode(model)
    assert model.norm.training
