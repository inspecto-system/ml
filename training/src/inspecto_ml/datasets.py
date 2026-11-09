"""Where the open datasets lie on disk and how to read them (audit, split, training)."""

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class DatasetPaths:
    """Downloaded archives; a dataset whose path is ``None`` is skipped."""

    bd3: Path | None = None
    sdnet: Path | None = None
    fbd: Path | None = None


@dataclass(frozen=True)
class Source:
    """A dataset: its files (for checksums) and a way to read its samples, as often as needed."""

    files: list[Path]
    samples: Callable[[], Iterator[Sample]]


def _fbd_samples(directory: Path) -> Iterator[Sample]:
    for part in FBD_PARTS:
        images = directory / f"{part}_images.npy"
        labels = directory / f"{part}_labels.npy"
        yield from read_arrays(images, labels, "fbd", FBD_CLASSES)


def dataset_sources(paths: DatasetPaths) -> dict[str, Source]:
    """Sources of the datasets given in ``paths``, by dataset name."""
    sources: dict[str, Source] = {}
    if (bd3 := paths.bd3) is not None:
        sources["bd3"] = Source([bd3], lambda: read_zip(bd3, "bd3", parent_folder))
    if (sdnet := paths.sdnet) is not None:
        sources["sdnet2018"] = Source(
            [sdnet],
            lambda: read_nested_zip(sdnet, SDNET_INNER_ARCHIVE, "sdnet2018", last_two_folders),
        )
    if (fbd := paths.fbd) is not None:
        files = [fbd / f"{part}_{kind}.npy" for part in FBD_PARTS for kind in ("images", "labels")]
        sources["fbd"] = Source(files, lambda: _fbd_samples(fbd))
    return sources


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Command-line options ``--bd3``, ``--sdnet``, ``--fbd``."""
    parser.add_argument("--bd3", type=Path, help="BD3 archive from Kaggle (archive.zip)")
    parser.add_argument("--sdnet", type=Path, help="SDNET2018 archive (DATA_Maguire_...zip)")
    parser.add_argument("--fbd", type=Path, help="directory with the extracted FBD .npy files")


def paths_from(args: argparse.Namespace) -> DatasetPaths:
    """Dataset paths from parsed ``add_dataset_arguments`` options."""
    return DatasetPaths(args.bd3, args.sdnet, args.fbd)
