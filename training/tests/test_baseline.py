import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inspecto_ml import baseline_cli
from inspecto_ml.baseline import PartData, linear_probe, majority_baseline, select_rows
from inspecto_ml.datasets import DatasetPaths, dataset_sources
from inspecto_ml.manifest import load_manifest, read_images
from inspecto_ml.split import Part
from inspecto_ml.split_cli import main as split_main
from tests.audit.helpers import jpeg_bytes, write_zip
from tests.fakes import tiny_backbone

CENTRES = {"CRACK": (0.0, 0.0), "STAIN": (5.0, 0.0), "NO_DEFECT": (0.0, 5.0)}


def _part(rng: np.random.Generator, per_class: int, classes: tuple[str, ...]) -> PartData:
    targets = [name for name in classes for _ in range(per_class)]
    features = np.array([rng.normal(CENTRES[name], 0.5) for name in targets])
    return PartData(features, targets)


def _parts() -> dict[Part, PartData]:
    rng = np.random.default_rng(0)
    every = ("CRACK", "STAIN", "NO_DEFECT")
    train = _part(rng, 30, every)
    train = PartData(
        np.vstack([train.features, _part(rng, 30, ("NO_DEFECT",)).features]),
        train.targets + ["NO_DEFECT"] * 30,
    )
    return {
        Part.TRAIN: train,
        Part.VAL: _part(rng, 10, every),
        Part.TEST: _part(rng, 10, every),
        Part.EXTERNAL: _part(rng, 10, ("CRACK", "STAIN")),
    }


def test_majority_baseline_predicts_the_most_frequent_training_class() -> None:
    result = majority_baseline(_parts())

    assert result["predicts"] == "NO_DEFECT"
    assert result["val"]["macro_f1"] == pytest.approx(1 / 6)  # F1 0.5 for NO_DEFECT, 0 for others
    assert result["external"]["per_class"].keys() == {"CRACK", "STAIN"}


def test_linear_probe_separates_clear_clusters_and_reports_the_search() -> None:
    result = linear_probe(_parts(), seed=1)

    assert result["val"]["macro_f1"] == pytest.approx(1.0)
    assert result["test"]["macro_f1"] > 0.9
    assert result["external"]["macro_f1"] > 0.9
    assert set(result["val_macro_f1_by_C"]) == {"0.01", "0.1", "1.0"}


def test_select_rows_follows_the_wanted_order() -> None:
    features = np.array([[1.0], [2.0], [3.0]])

    assert select_rows(["a", "b", "c"], features, ["c", "a"]).tolist() == [[3.0], [1.0]]


def _colour(name: str, index: int) -> Image.Image:
    """Class colour plus a random texture: distinct pictures, separable by mean colour."""
    base = {"stain": (200, 40, 40), "crack": (40, 40, 40), "plain": (120, 200, 120)}[name]
    seed = 1000 * ("stain", "crack", "plain").index(name) + index
    texture = np.random.default_rng(seed).integers(-30, 31, size=(4, 4, 3))
    pixels = np.clip(np.asarray(base) + texture, 0, 255).astype(np.uint8)
    return Image.fromarray(np.kron(pixels, np.ones((8, 8, 1), dtype=np.uint8)))


def _archive(tmp_path: Path) -> Path:
    members = {
        f"t/{name}/{i}.jpg": jpeg_bytes(_colour(name, i))
        for name in ("stain", "crack", "plain")
        for i in range(20)
    }
    return write_zip(tmp_path / "bd3.zip", members)


def _mapping(tmp_path: Path) -> Path:
    path = tmp_path / "mapping.toml"
    path.write_text(
        '[bd3]\nrole = "train"\nexcluded = []\n'
        '[bd3.classes]\nstain = "STAIN"\ncrack = "CRACK"\nplain = "NO_DEFECT"\n',
        encoding="utf-8",
    )
    return path


def test_read_images_fails_when_the_archive_lacks_manifest_images(tmp_path: Path) -> None:
    bd3 = _archive(tmp_path)
    split = tmp_path / "split.json"
    split_main(["--bd3", str(bd3), "--mapping", str(_mapping(tmp_path)), "--out", str(split)])
    manifest = load_manifest(split)
    other = write_zip(tmp_path / "other.zip", {"t/stain/new.jpg": jpeg_bytes(_colour("stain", 99))})

    with pytest.raises(KeyError, match="missing in the archives"):
        list(read_images(manifest.entries, dataset_sources(DatasetPaths(bd3=other))))


def test_command_line_runs_baselines_and_reuses_cached_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline_cli, "resnet50", tiny_backbone)
    bd3 = _archive(tmp_path)
    split = tmp_path / "split.json"
    split_main(["--bd3", str(bd3), "--mapping", str(_mapping(tmp_path)), "--out", str(split)])
    args = ["--bd3", str(bd3), "--split", str(split), "--features", str(tmp_path / "f")]

    baseline_cli.main([*args, "--out", str(tmp_path / "run1.json")])
    baseline_cli.main([*args, "--out", str(tmp_path / "run2.json")])

    first = json.loads((tmp_path / "run1.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "run2.json").read_text(encoding="utf-8"))
    assert first["linear_probe"]["test"]["macro_f1"] == pytest.approx(1.0)
    assert first["split_version"] == load_manifest(split).version
    assert second["feature_seconds"] == first["feature_seconds"]  # read from the cache
    assert len(list((tmp_path / "f").iterdir())) == 1
