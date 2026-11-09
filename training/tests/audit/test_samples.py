import io
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from inspecto_ml.audit.samples import (
    last_two_folders,
    parent_folder,
    read_arrays,
    read_nested_zip,
    read_zip,
)
from tests.audit.helpers import ImageFactory, jpeg_bytes, write_zip


def test_should_label_by_folder_and_skip_non_images(tmp_path: Path, picture: ImageFactory) -> None:
    archive = write_zip(
        tmp_path / "bd3.zip",
        {
            "BD3/train/algae/a.jpg": jpeg_bytes(picture(1)),
            "BD3/train/stain/b.JPG": jpeg_bytes(picture(2)),
            "BD3/readme.txt": b"not an image",
        },
    )

    samples = list(read_zip(archive, "bd3", parent_folder))

    assert [(s.dataset, s.label, s.key) for s in samples] == [
        ("bd3", "algae", "BD3/train/algae/a.jpg"),
        ("bd3", "stain", "BD3/train/stain/b.JPG"),
    ]
    assert samples[0].decode().size == (64, 48)


def test_should_read_images_of_an_archive_inside_an_archive(
    tmp_path: Path, picture: ImageFactory
) -> None:
    inner = write_zip(tmp_path / "inner.zip", {"SDNET/W/CW/1.jpg": jpeg_bytes(picture(3))})
    outer = write_zip(tmp_path / "outer.zip", {"ALL/SDNET2018.zip": inner.read_bytes()})

    samples = list(read_nested_zip(outer, "ALL/SDNET2018.zip", "sdnet2018", last_two_folders))

    assert [(s.label, s.key) for s in samples] == [("W/CW", "SDNET/W/CW/1.jpg")]


def test_folder_labels() -> None:
    path = PurePosixPath("SDNET2018/W/UW/7001-1.jpg")

    assert parent_folder(path) == "UW"
    assert last_two_folders(path) == "W/UW"


def _save(path: Path, array: np.ndarray) -> Path:
    buffer = io.BytesIO()
    np.save(buffer, array)
    path.write_bytes(buffer.getvalue())
    return path


def test_should_read_labelled_arrays(tmp_path: Path) -> None:
    pixels = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    pixels[1] = 255
    images = _save(tmp_path / "training_images.npy", pixels)
    labels = _save(tmp_path / "training_labels.npy", np.array([0, 1, 1]))

    samples = list(read_arrays(images, labels, "fbd", ("spalling", "crack")))

    assert [(s.label, s.key) for s in samples] == [
        ("spalling", "training_images.npy[0]"),
        ("crack", "training_images.npy[1]"),
        ("crack", "training_images.npy[2]"),
    ]
    assert samples[0].content == samples[2].content != samples[1].content
    assert samples[1].decode().getpixel((0, 0)) == (255, 255, 255)


def test_should_reject_arrays_of_different_length(tmp_path: Path) -> None:
    images = _save(tmp_path / "images.npy", np.zeros((2, 4, 4, 3), dtype=np.uint8))
    labels = _save(tmp_path / "labels.npy", np.array([0]))

    with pytest.raises(ValueError, match="2 images but 1 labels"):
        list(read_arrays(images, labels, "fbd", ("spalling",)))
