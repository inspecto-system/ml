"""Command line: ``python -m inspecto_ml.export_cli --split ... --run baseline.json --out-dir ...``.

Exports the reference model of ml-service — the ML-03 linear probe on ResNet-50 — to
``model.onnx`` with ``model.json`` (version, classes, preprocessing, threshold, metrics). The probe
is refitted on the cached features with the ``C`` of the baseline run and the same seed; the
threshold is chosen on the validation part. The export fails unless ONNX Runtime on Pillow
preprocessing agrees with PyTorch on torchvision preprocessing for the first test images:
probabilities within 1e-3 and the same top-1 class.
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.pipeline import Pipeline

from inspecto_ml.baseline import PartData, probe, select_rows
from inspecto_ml.baseline_cli import load_features
from inspecto_ml.class_mapping import NO_DEFECT
from inspecto_ml.datasets import add_dataset_arguments, dataset_sources, paths_from
from inspecto_ml.export import (
    Equivalence,
    LinearProbe,
    Preprocessing,
    apply_threshold,
    check_equivalence,
    choose_threshold,
    export_onnx,
    preprocess,
)
from inspecto_ml.features import Backbone, resnet50
from inspecto_ml.manifest import Manifest, load_manifest, read_images
from inspecto_ml.metrics import evaluate
from inspecto_ml.split import Part

MAX_DIFFERENCE = 1e-3
log = logging.getLogger("inspecto_ml.export")


class EquivalenceError(RuntimeError):
    """ONNX Runtime does not reproduce PyTorch closely enough."""


def _parts(args: argparse.Namespace, manifest: Manifest) -> dict[Part, PartData]:
    order, features, _ = load_features(args, manifest)
    parts = {}
    for part in (Part.TRAIN, Part.VAL, Part.TEST):
        entries = manifest.part(part)
        rows = select_rows(order, features, [entry.uid for entry in entries])
        parts[part] = PartData(rows, [entry.target for entry in entries])
    return parts


def _module(backbone: Backbone, fitted: Pipeline) -> LinearProbe:
    scaler, classifier = fitted[0], fitted[-1]
    return LinearProbe(
        backbone.network, scaler.mean_, scaler.scale_, classifier.coef_, classifier.intercept_
    )


def _equivalence(
    args: argparse.Namespace,
    manifest: Manifest,
    module: LinearProbe,
    backbone: Backbone,
    path: Path,
) -> tuple[Equivalence, Preprocessing]:
    spec = Preprocessing(resize=backbone_resize(backbone), crop=backbone_crop(backbone))
    entries = manifest.part(Part.TEST)[: args.check_images]
    sources = dataset_sources(paths_from(args))
    torch_inputs, pillow_inputs = [], []
    for _, image in read_images(entries, sources):
        torch_inputs.append(backbone.transform(image))
        pillow_inputs.append(preprocess(image, spec))
    result = check_equivalence(
        module, path, torch.stack(torch_inputs).numpy(), np.stack(pillow_inputs)
    )
    if result.max_abs_difference >= MAX_DIFFERENCE or result.top1_agreement < 1.0:
        msg = f"ONNX export is not equivalent: {result}"
        raise EquivalenceError(msg)
    return result, spec


def backbone_resize(backbone: Backbone) -> int:
    """Shorter-side size of the torchvision preprocessing of the backbone."""
    return int(getattr(backbone.transform, "resize_size", [256])[0])


def backbone_crop(backbone: Backbone) -> int:
    """Crop size of the torchvision preprocessing of the backbone."""
    return int(getattr(backbone.transform, "crop_size", [224])[0])


def build_model(args: argparse.Namespace) -> dict[str, Any]:
    """Fit, export and check the reference model; return the content of ``model.json``."""
    manifest = load_manifest(args.split)
    run = json.loads(args.run.read_text(encoding="utf-8"))
    parts = _parts(args, manifest)
    fitted = probe(run["linear_probe"]["C"], run["seed"]).fit(
        parts[Part.TRAIN].features, parts[Part.TRAIN].targets
    )
    classes = [str(name) for name in fitted.classes_]
    val_probabilities = fitted.predict_proba(parts[Part.VAL].features)
    threshold, search = choose_threshold(
        val_probabilities, parts[Part.VAL].targets, classes, NO_DEFECT
    )
    test = apply_threshold(
        fitted.predict_proba(parts[Part.TEST].features), classes, threshold, NO_DEFECT
    )
    backbone = resnet50()
    module = _module(backbone, fitted)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "model.onnx"
    export_onnx(module, path, backbone_crop(backbone))
    equivalence, spec = _equivalence(args, manifest, module, backbone, path)
    return {
        "modelVersion": f"resnet50-logreg-{manifest.version[:8]}",
        "classes": classes,
        "noDefectClass": NO_DEFECT,
        "inputSize": spec.crop,
        "preprocessing": asdict(spec),
        "threshold": threshold,
        "splitVersion": manifest.version,
        "C": run["linear_probe"]["C"],
        "seed": run["seed"],
        "thresholdSearchValMacroF1": search,
        "testWithThreshold": evaluate(parts[Part.TEST].targets, test, classes),
        "equivalence": asdict(equivalence),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inspecto_ml.export_cli", description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--split", type=Path, required=True, help="split manifest")
    parser.add_argument("--features", type=Path, required=True, help="feature cache directory")
    parser.add_argument("--run", type=Path, required=True, help="baseline run report (C, seed)")
    parser.add_argument("--check-images", type=int, default=256, help="test images to compare")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory of the model")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Export the reference model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    info = build_model(args)
    path = args.out_dir / "model.json"
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "model %s: threshold %s, %s", info["modelVersion"], info["threshold"], info["equivalence"]
    )


if __name__ == "__main__":
    main()
