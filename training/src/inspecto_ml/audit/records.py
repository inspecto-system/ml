"""Per-image facts used by the audit: content digest, perceptual hash, size, format, errors."""

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from inspecto_ml.audit.samples import Sample

HASH_SIDE = 8
PROGRESS_EVERY = 5000

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    """What the audit knows about one image. ``error`` is set when the file cannot be decoded."""

    dataset: str
    label: str
    key: str
    sha256: str
    dhash: int | None = None
    width: int = 0
    height: int = 0
    image_format: str = ""
    mode: str = ""
    error: str = ""


def _row_differences(gray: Image.Image) -> int:
    """64 bits: is each pixel brighter than its right neighbour on a 9x8 grid."""
    small = gray.resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())
    value = 0
    for row in range(HASH_SIDE):
        for column in range(HASH_SIDE):
            left = pixels[row * (HASH_SIDE + 1) + column]
            right = pixels[row * (HASH_SIDE + 1) + column + 1]
            value = (value << 1) | int(left > right)
    return value


def difference_hash(image: Image.Image) -> int:
    """128-bit difference hash: horizontal (64 bits) and vertical (64 bits) brightness steps.

    Near-identical pictures (resized, recompressed, slightly brightened) get hashes that differ
    in a few bits, so the Hamming distance finds near-duplicates. One direction alone is not
    enough: every featureless surface lit from one side (a smooth gradient) gets the same
    all-zero or all-one horizontal hash; the second direction tells such pictures apart.
    """
    gray = image.convert("L")
    horizontal = _row_differences(gray)
    vertical = _row_differences(gray.transpose(Image.Transpose.TRANSPOSE))
    return (horizontal << (HASH_SIDE * HASH_SIDE)) | vertical


def inspect_sample(sample: Sample) -> ImageRecord:
    """Decode the sample fully once and record its facts; a broken image is recorded, not raised."""
    digest = hashlib.sha256(sample.content).hexdigest()
    try:
        with sample.decode() as image:
            image.load()
            return ImageRecord(
                dataset=sample.dataset,
                label=sample.label,
                key=sample.key,
                sha256=digest,
                dhash=difference_hash(image),
                width=image.width,
                height=image.height,
                image_format=image.format or "ARRAY",
                mode=image.mode,
            )
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return ImageRecord(sample.dataset, sample.label, sample.key, digest, error=str(error))


def inspect_all(name: str, samples: Iterable[Sample]) -> list[ImageRecord]:
    """Inspect every sample of a dataset, logging progress."""
    records = []
    for count, sample in enumerate(samples, start=1):
        records.append(inspect_sample(sample))
        if count % PROGRESS_EVERY == 0:
            log.info("%s: %d images", name, count)
    return records
