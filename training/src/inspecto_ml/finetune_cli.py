"""Command line: ``python -m inspecto_ml.finetune_cli --model deit_small --split ... --out ...``.

Fine-tunes one pretrained network (``RECIPES``) on a class-balanced subset of the training part,
chooses the best epoch on the validation part, then evaluates once on the test and external parts.
Writes a run report and the weights; pretrained weights are downloaded once into ``HF_HOME``.
"""

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from inspecto_ml.datasets import add_dataset_arguments, dataset_sources, paths_from
from inspecto_ml.finetune import RECIPES, Normalisation, Trainer, TrainingData, predict
from inspecto_ml.imagecache import balanced_subset, cached_images
from inspecto_ml.manifest import load_manifest
from inspecto_ml.metrics import evaluate
from inspecto_ml.split import Part

TRAIN_SIDE = 256  # random 224 crops are taken from it
EVAL_SIDE = 256  # the centre 224 crop is used, as ml-service does after resizing to 256

log = logging.getLogger("inspecto_ml.finetune")


def create_model(timm_name: str, classes: int) -> tuple[torch.nn.Module, Normalisation]:
    """Pretrained network with a new classifier and its input normalisation."""
    import timm  # noqa: PLC0415 — heavy import, only for real runs

    model = timm.create_model(timm_name, pretrained=True, num_classes=classes)
    config = cast("dict[str, Any]", model.pretrained_cfg)
    return model, Normalisation(tuple(config["mean"]), tuple(config["std"]))


def build_run(args: argparse.Namespace) -> dict[str, Any]:
    """Cache images, fine-tune and evaluate one network."""
    torch.set_num_threads(os.cpu_count() or 1)
    recipe = RECIPES[args.model]
    manifest = load_manifest(args.split)
    sources = dataset_sources(paths_from(args))
    prefix = args.cache / manifest.version[:12]
    train = balanced_subset(manifest.part(Part.TRAIN), args.per_class, args.seed)
    classes = sorted({entry.target for entry in manifest.entries if entry.part is Part.TRAIN})
    index = {name: position for position, name in enumerate(classes)}

    def images(part: Part, entries: list[Any], side: int) -> np.ndarray:
        name = f"{prefix}-{part.value}-{len(entries)}-{side}.npy"
        return cached_images(entries, sources, side, Path(name))

    started = time.monotonic()
    data = TrainingData(
        images=images(Part.TRAIN, train, TRAIN_SIDE),
        labels=np.array([index[entry.target] for entry in train]),
        val_images=images(Part.VAL, manifest.part(Part.VAL), EVAL_SIDE),
        val_targets=[entry.target for entry in manifest.part(Part.VAL)],
        classes=classes,
    )
    cache_seconds = time.monotonic() - started
    model, normalisation = create_model(recipe.timm_name, len(classes))
    started = time.monotonic()
    history = Trainer(model, data, recipe, normalisation, log.info, args.seed).run()
    train_seconds = time.monotonic() - started
    results = {}
    for part in (Part.TEST, Part.EXTERNAL):
        entries = manifest.part(part)
        if not entries:
            continue
        probabilities = predict(model, images(part, entries, EVAL_SIDE), normalisation)
        targets = [entry.target for entry in entries]
        labels = sorted(set(targets)) if part is Part.EXTERNAL else None
        results[part.value] = evaluate(targets, probabilities, classes, labels)
    started = time.monotonic()
    predict(model, data.val_images[:64], normalisation, batch_size=1)
    per_image_ms = (time.monotonic() - started) / 64 * 1000
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.weights)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "timm_name": recipe.timm_name,
        "split_version": manifest.version,
        "seed": args.seed,
        "train_images_per_class": dict(
            zip(*np.unique([entry.target for entry in train], return_counts=True), strict=True)
        ),
        "classes": classes,
        "cpu_threads": torch.get_num_threads(),
        "cache_seconds": round(cache_seconds, 1),
        "train_seconds": round(train_seconds, 1),
        "cpu_ms_per_image": round(per_image_ms, 1),
        "epochs": history.epochs,
        "best_epoch": history.best_epoch,
        "val_macro_f1": history.best_val_macro_f1,
        **results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inspecto_ml.finetune_cli", description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--model", choices=sorted(RECIPES), required=True)
    parser.add_argument("--split", type=Path, required=True, help="split manifest")
    parser.add_argument("--cache", type=Path, required=True, help="directory of image caches")
    parser.add_argument("--per-class", type=int, default=1500, help="training images per class")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weights", type=Path, required=True, help="where to save the weights")
    parser.add_argument("--out", type=Path, required=True, help="JSON run report")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Fine-tune and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    run = build_run(args)
    args.out.write_text(
        json.dumps(run, ensure_ascii=False, indent=2, default=int), encoding="utf-8"
    )
    log.info(
        "%s: val %.3f (epoch %d), test %s, external %s",
        args.model,
        run["val_macro_f1"],
        run["best_epoch"],
        run.get("test", {}).get("macro_f1"),
        run.get("external", {}).get("macro_f1"),
    )


if __name__ == "__main__":
    main()
