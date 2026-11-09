"""Command line: ``python -m inspecto_ml.audit --bd3 ... --sdnet ... --fbd ... --out report.json``.

Reads the archives as downloaded (FBD — the arrays extracted from its two-part archive with 7-Zip)
and writes a JSON report; nothing is unpacked or copied.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspecto_ml.audit.duplicates import exact_groups, near_groups
from inspecto_ml.audit.records import ImageRecord, inspect_all
from inspecto_ml.audit.report import dataset_summary, duplicate_summary
from inspecto_ml.datasets import add_dataset_arguments, dataset_sources, paths_from
from inspecto_ml.fingerprint import file_sha256

log = logging.getLogger("inspecto_ml.audit")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect every image of the given datasets and summarise them and their duplicates."""
    datasets: dict[str, Any] = {}
    every: list[ImageRecord] = []
    for name, source in dataset_sources(paths_from(args)).items():
        records = inspect_all(name, source.samples())
        every.extend(records)
        summary = dataset_summary(records)
        summary["files"] = {path.name: file_sha256(path) for path in source.files}
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
    add_dataset_arguments(parser)
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
