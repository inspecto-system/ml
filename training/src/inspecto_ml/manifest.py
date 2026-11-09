"""Reading a split manifest and the images it lists."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from inspecto_ml.datasets import Source
from inspecto_ml.split import Entry, Part


@dataclass(frozen=True)
class Manifest:
    """Entries of a split and its version (SHA-256 of the entries)."""

    version: str
    entries: list[Entry]

    def part(self, part: Part) -> list[Entry]:
        """Entries of one part, in manifest order."""
        return [entry for entry in self.entries if entry.part is part]


def load_manifest(path: Path) -> Manifest:
    """Read a manifest written by ``inspecto_ml.split_cli``."""
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = [Entry(**{**row, "part": Part(row["part"])}) for row in document["entries"]]
    return Manifest(document["version"], entries)


def read_images(
    entries: list[Entry], sources: Mapping[str, Source]
) -> Iterator[tuple[Entry, Image.Image]]:
    """Decoded RGB images of the entries, read from the dataset archives in one pass each.

    Raises:
        KeyError: a dataset of the entries has no source, or an entry was not found in it.
    """
    wanted = {entry.uid: entry for entry in entries}
    found = 0
    for dataset in sorted({entry.dataset for entry in entries}):
        for sample in sources[dataset].samples():
            entry = wanted.get(f"{sample.dataset}:{sample.key}")
            if entry is not None:
                found += 1
                with sample.decode() as image:
                    yield entry, image.convert("RGB")
    if found != len(wanted):
        msg = f"{len(wanted) - found} images of the manifest are missing in the archives"
        raise KeyError(msg)
