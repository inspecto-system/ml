"""Command line: ``python -m inspecto_ml.split_cli --bd3 ... --fbd ... --out split.json``.

Writes the split manifest: every used image with its category and part, the counts, what was
dropped, checksums of the source files and of the mapping, and a version — the SHA-256 of the
entries, by which results refer to the exact data they were computed on.
"""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspecto_ml.audit.records import ImageRecord, inspect_all
from inspecto_ml.class_mapping import load_mapping
from inspecto_ml.datasets import add_dataset_arguments, dataset_sources, paths_from
from inspecto_ml.fingerprint import file_sha256, json_sha256
from inspecto_ml.split import build_split, part_counts

log = logging.getLogger("inspecto_ml.split")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect the datasets and split them."""
    mapping = load_mapping(args.mapping)
    records: list[ImageRecord] = []
    files: dict[str, str] = {}
    for name, source in dataset_sources(paths_from(args)).items():
        records.extend(inspect_all(name, source.samples()))
        files |= {path.name: file_sha256(path) for path in source.files}
    entries, stats = build_split(records, mapping, args.seed, args.max_distance)
    rows = [asdict(entry) for entry in entries]
    return {
        "version": json_sha256(rows),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "near_duplicate_max_distance": args.max_distance,
        "mapping_sha256": json_sha256(
            {
                name: {**asdict(dataset), "excluded": sorted(dataset.excluded)}
                for name, dataset in mapping.items()
            }
        ),
        "source_files": files,
        "dropped": asdict(stats),
        "counts": part_counts(entries),
        "entries": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m inspecto_ml.split_cli", description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--mapping", type=Path, help="class mapping (default: the approved one)")
    parser.add_argument("--seed", type=int, default=2026, help="seed of the group order")
    parser.add_argument("--max-distance", type=int, default=4, help="near-duplicate distance")
    parser.add_argument("--out", type=Path, required=True, help="JSON manifest to write")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Build the split and write the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    manifest = build_manifest(args)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("split %s written to %s: %s", manifest["version"][:12], args.out, manifest["counts"])


if __name__ == "__main__":
    main()
