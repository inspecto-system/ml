"""Command line: ``python -m inspecto_ml.audit --bd3 ... --sdnet ... --fbd ... --out report.json``.

Reads the archives as downloaded (FBD — the arrays extracted from its two-part archive with 7-Zip)
and writes a JSON report; nothing is unpacked or copied.
"""

import argparse
import hashlib
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspecto_ml.audit.duplicates import exact_groups, near_groups
from inspecto_ml.audit.records import ImageRecord, inspect_sample
from inspecto_ml.audit.report import dataset_summary, duplicate_summary
from inspecto_ml.audit.samples import (
    Sample,
    last_two_folders,
    parent_folder,
    read_arrays,
    read_nested_zip,
    read_zip,
)

SDNET_INNER_ARCHIVE = "DATA_Maguire_20180517_ALL/SDNET2018.zip"
# FBD stores labels as integers 0..3. The order is not documented; it follows from the class sizes
# in the FBD README (spalling 436/109, crack 232/58, stain 422/106, vegetation 182/46 in the
# training/validation arrays) — each count is unique, so the mapping is unambiguous.
FBD_CLASSES = ("spalling", "crack", "stain", "vegetation")
FBD_PARTS = ("training", "validation")
PROGRESS_EVERY = 5000

log = logging.getLogger("inspecto_ml.audit")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fbd_samples(directory: Path) -> Iterator[Sample]:
    for part in FBD_PARTS:
        images = directory / f"{part}_images.npy"
        labels = directory / f"{part}_labels.npy"
        yield from read_arrays(images, labels, "fbd", FBD_CLASSES)


def _sources(args: argparse.Namespace) -> dict[str, tuple[list[Path], Iterator[Sample]]]:
    sources: dict[str, tuple[list[Path], Iterator[Sample]]] = {}
    if args.bd3:
        sources["bd3"] = ([args.bd3], read_zip(args.bd3, "bd3", parent_folder))
    if args.sdnet:
        samples = read_nested_zip(args.sdnet, SDNET_INNER_ARCHIVE, "sdnet2018", last_two_folders)
        sources["sdnet2018"] = ([args.sdnet], samples)
    if args.fbd:
        files = [
            args.fbd / f"{part}_{kind}.npy" for part in FBD_PARTS for kind in ("images", "labels")
        ]
        sources["fbd"] = (files, _fbd_samples(args.fbd))
    return sources


def _inspect_all(name: str, samples: Iterator[Sample]) -> list[ImageRecord]:
    records = []
    for count, sample in enumerate(samples, start=1):
        records.append(inspect_sample(sample))
        if count % PROGRESS_EVERY == 0:
            log.info("%s: %d images", name, count)
    return records


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect every image of the given datasets and summarise them and their duplicates."""
    datasets: dict[str, Any] = {}
    every: list[ImageRecord] = []
    for name, (files, samples) in _sources(args).items():
        records = _inspect_all(name, samples)
        every.extend(records)
        summary = dataset_summary(records)
        summary["files"] = {path.name: _file_sha256(path) for path in files}
        datasets[name] = summary
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "near_duplicate_max_distance": args.max_distance,
        "datasets": datasets,
        "exact_duplicates": duplicate_summary(exact_groups(every)),
        "near_duplicates": duplicate_summary(near_groups(every, args.max_distance)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inspecto_ml.audit", description=__doc__)
    parser.add_argument("--bd3", type=Path, help="BD3 archive from Kaggle (archive.zip)")
    parser.add_argument("--sdnet", type=Path, help="SDNET2018 archive (DATA_Maguire_...zip)")
    parser.add_argument("--fbd", type=Path, help="directory with the extracted FBD .npy files")
    parser.add_argument(
        "--max-distance", type=int, default=4, help="near-duplicate distance, bits of 128"
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON report to write")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the audit and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    report = build_report(args)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("report written to %s", args.out)


if __name__ == "__main__":
    main()
