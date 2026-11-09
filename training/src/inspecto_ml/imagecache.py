"""Images of a split decoded once into a uint8 array on disk (ML-04/ML-05).

Fine-tuning reads every training image in every epoch; decoding JPEGs from the archives each time
would cost more than the training on CPU. Images are stored square: the shorter side resized to
``side`` (bilinear, as ``inspecto_ml.export.preprocess``) and the centre cropped.
"""

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from inspecto_ml.datasets import Source
from inspecto_ml.manifest import read_images
from inspecto_ml.split import Entry


def square(image: Image.Image, side: int) -> np.ndarray:
    """``(side, side, 3)`` uint8: shorter side resized to ``side``, centre crop."""
    rgb = image.convert("RGB")
    short, long = sorted(rgb.size)
    other = int(side * long / short)
    size = (side, other) if rgb.width <= rgb.height else (other, side)
    resized = rgb.resize(size, Image.Resampling.BILINEAR)
    left = round((resized.width - side) / 2)
    top = round((resized.height - side) / 2)
    return np.asarray(resized.crop((left, top, left + side, top + side)), dtype=np.uint8)


def balanced_subset(entries: Sequence[Entry], per_class: int, seed: int) -> list[Entry]:
    """At most ``per_class`` entries of every target, chosen with the seed, in manifest order."""
    by_target: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_target[entry.target].append(entry)
    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    for target in sorted(by_target):
        group = sorted(by_target[target], key=lambda entry: entry.uid)
        picks = rng.permutation(len(group))[:per_class]
        chosen.update(group[int(index)].uid for index in picks)
    return [entry for entry in entries if entry.uid in chosen]


def cached_images(
    entries: Sequence[Entry], sources: Mapping[str, Source], side: int, path: Path
) -> np.ndarray:
    """``(N, side, side, 3)`` images of the entries in their order, built once and memory-mapped."""
    order_file = path.with_suffix(".json")
    wanted = [entry.uid for entry in entries]
    if path.exists() and order_file.exists():
        stored = json.loads(order_file.read_text(encoding="utf-8"))
        if stored == wanted:
            reused: np.ndarray = np.load(path, mmap_mode="r")
            return reused
    path.parent.mkdir(parents=True, exist_ok=True)
    position = {uid: index for index, uid in enumerate(wanted)}
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(len(wanted), side, side, 3)
    )
    for entry, image in read_images(list(entries), sources):
        array[position[entry.uid]] = square(image, side)
    array.flush()
    order_file.write_text(json.dumps(wanted), encoding="utf-8")
    built: np.ndarray = np.load(path, mmap_mode="r")
    return built
