"""Readers that turn a dataset archive into a stream of labelled samples without unpacking it."""

import io
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class Sample:
    """One image of a dataset: where it is, which class it belongs to and how to read it."""

    dataset: str
    label: str
    key: str
    content: bytes
    decode: Callable[[], Image.Image]


def _decode_bytes(content: bytes) -> Callable[[], Image.Image]:
    return lambda: Image.open(io.BytesIO(content))


def _is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES


def _zip_samples(
    archive: zipfile.ZipFile, dataset: str, label_of: Callable[[PurePosixPath], str]
) -> Iterator[Sample]:
    for info in archive.infolist():
        if info.is_dir() or not _is_image(info.filename):
            continue
        content = archive.read(info)
        path = PurePosixPath(info.filename)
        yield Sample(dataset, label_of(path), info.filename, content, _decode_bytes(content))


def parent_folder(path: PurePosixPath) -> str:
    """Class = the folder the image lies in (BD3: ``.../train/algae/x.jpg`` → ``algae``)."""
    return path.parent.name


def last_two_folders(path: PurePosixPath) -> str:
    """Class = surface and crack folders (SDNET2018: ``W/CW/x.jpg`` → ``W/CW``)."""
    return f"{path.parent.parent.name}/{path.parent.name}"


def read_zip(
    path: Path, dataset: str, label_of: Callable[[PurePosixPath], str]
) -> Iterator[Sample]:
    """Images of a zip archive, labelled by their path inside it."""
    with zipfile.ZipFile(path) as archive:
        yield from _zip_samples(archive, dataset, label_of)


def read_nested_zip(
    path: Path, inner: str, dataset: str, label_of: Callable[[PurePosixPath], str]
) -> Iterator[Sample]:
    """Images of a zip archive stored inside another one (SDNET2018 is shipped this way)."""
    with (
        zipfile.ZipFile(path) as outer,
        outer.open(inner) as handle,
        zipfile.ZipFile(handle) as archive,
    ):
        yield from _zip_samples(archive, dataset, label_of)


def _array_decoder(array: np.ndarray) -> Callable[[], Image.Image]:
    return lambda: Image.fromarray(array)


def read_arrays(
    images: Path, labels: Path, dataset: str, class_names: tuple[str, ...]
) -> Iterator[Sample]:
    """Images stored as a NumPy array ``(N, H, W, 3)`` with integer labels (FBD)."""
    pixels = np.load(images, mmap_mode="r", allow_pickle=False)
    classes = np.load(labels, allow_pickle=False)
    if len(pixels) != len(classes):
        msg = f"{images.name}: {len(pixels)} images but {len(classes)} labels"
        raise ValueError(msg)
    for index, label in enumerate(classes):
        array = np.ascontiguousarray(pixels[index])
        key = f"{images.name}[{index}]"
        yield Sample(dataset, class_names[int(label)], key, array.tobytes(), _array_decoder(array))
