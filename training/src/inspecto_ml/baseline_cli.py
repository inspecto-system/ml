"""Command line: ``python -m inspecto_ml.baseline_cli --split split.json --bd3 ... --out run.json``.

Computes ResNet-50 features of every image of the split (cached in ``--features`` by split
version), fits the baselines and writes a run report: split version, model, seed, timings and
the metrics on the validation, test and external parts.
"""

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inspecto_ml.baseline import PartData, linear_probe, majority_baseline, select_rows
from inspecto_ml.datasets import add_dataset_arguments, dataset_sources, paths_from
from inspecto_ml.features import extract, resnet50
from inspecto_ml.manifest import Manifest, load_manifest, read_images
from inspecto_ml.split import Part

log = logging.getLogger("inspecto_ml.baseline")


def load_features(
    args: argparse.Namespace, manifest: Manifest
) -> tuple[list[str], np.ndarray, float]:
    """ResNet-50 features of the split: uids, matrix and seconds spent; cached by split version."""
    cache = args.features / f"resnet50-{manifest.version[:12]}.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        return [str(uid) for uid in stored["uids"]], stored["features"], float(stored["seconds"])
    torch.set_num_threads(os.cpu_count() or 1)
    sources = dataset_sources(paths_from(args))
    uids: list[str] = []

    def images() -> Any:  # noqa: ANN401 — generator of PIL images
        for count, (entry, image) in enumerate(read_images(manifest.entries, sources), start=1):
            uids.append(entry.uid)
            if count % 1000 == 0:
                log.info("features: %d of %d", count, len(manifest.entries))
            yield image

    started = time.monotonic()
    features = extract(resnet50(), images())
    seconds = time.monotonic() - started
    args.features.mkdir(parents=True, exist_ok=True)
    np.savez(cache, uids=np.asarray(uids), features=features, seconds=seconds)
    return uids, features, seconds


def build_run(args: argparse.Namespace) -> dict[str, Any]:
    """Features, both baselines and their metrics."""
    manifest = load_manifest(args.split)
    order, features, seconds = load_features(args, manifest)
    parts = {}
    for part in Part:
        entries = manifest.part(part)
        if entries:
            rows = select_rows(order, features, [entry.uid for entry in entries])
            parts[part] = PartData(rows, [entry.target for entry in entries])
    started = time.monotonic()
    probe = linear_probe(parts, args.seed)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split_version": manifest.version,
        "seed": args.seed,
        "backbone": "resnet50/IMAGENET1K_V2 (torchvision), frozen",
        "cpu_threads": torch.get_num_threads(),
        "feature_seconds": round(seconds, 1),
        "probe_fit_seconds": round(time.monotonic() - started, 1),
        "majority": majority_baseline(parts),
        "linear_probe": probe,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inspecto_ml.baseline_cli", description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--split", type=Path, required=True, help="split manifest")
    parser.add_argument("--features", type=Path, required=True, help="feature cache directory")
    parser.add_argument("--seed", type=int, default=2026, help="seed of the classifier")
    parser.add_argument("--out", type=Path, required=True, help="JSON run report to write")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the baselines and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    run = build_run(args)
    args.out.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "majority val macro-F1 %.3f; probe C=%s val %.3f test %.3f",
        run["majority"]["val"]["macro_f1"],
        run["linear_probe"]["C"],
        run["linear_probe"]["val"]["macro_f1"],
        run["linear_probe"]["test"]["macro_f1"],
    )


if __name__ == "__main__":
    main()
