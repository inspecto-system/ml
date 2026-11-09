import io

import numpy as np
from PIL import Image

from inspecto_ml.audit.records import difference_hash, inspect_sample
from inspecto_ml.audit.samples import Sample
from tests.audit.helpers import ImageFactory, jpeg_bytes


def _sample(content: bytes, label: str = "crack") -> Sample:
    return Sample("bd3", label, "x.jpg", content, lambda: Image.open(io.BytesIO(content)))


def test_resized_recompressed_copy_has_a_close_hash(picture: ImageFactory) -> None:
    original = picture(1)
    copy = Image.open(io.BytesIO(jpeg_bytes(original.resize((128, 96)), quality=70)))

    distance = (difference_hash(original) ^ difference_hash(copy)).bit_count()

    assert distance <= 4


def test_different_pictures_have_distant_hashes(picture: ImageFactory) -> None:
    distance = (difference_hash(picture(1)) ^ difference_hash(picture(2))).bit_count()

    assert distance > 20


def _side_lit_surface(seed: int) -> Image.Image:
    """A featureless surface brightening to the right, with faint random horizontal bands."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(40, 220, 64)[np.newaxis, :]
    bands = rng.normal(0, 12, size=(48, 1))
    return Image.fromarray(np.clip(gradient + bands, 0, 255).astype(np.uint8))


def test_side_lit_surfaces_are_not_taken_for_copies() -> None:
    first, second = difference_hash(_side_lit_surface(1)), difference_hash(_side_lit_surface(2))
    horizontal_mask = ((1 << 64) - 1) << 64

    assert first & horizontal_mask == second & horizontal_mask  # one direction alone: "copies"
    assert (first ^ second).bit_count() > 10


def test_should_record_facts_of_a_decodable_image(picture: ImageFactory) -> None:
    content = jpeg_bytes(picture(1))

    record = inspect_sample(_sample(content))

    assert (record.width, record.height, record.image_format, record.mode) == (
        64,
        48,
        "JPEG",
        "RGB",
    )
    assert record.dhash is not None
    assert len(record.sha256) == 64
    assert not record.error


def test_should_record_a_broken_image_instead_of_failing(picture: ImageFactory) -> None:
    truncated = jpeg_bytes(picture(1))[:200]

    record = inspect_sample(_sample(truncated))

    assert record.error
    assert record.dhash is None


def test_should_record_bytes_that_are_not_an_image() -> None:
    record = inspect_sample(_sample(b"plain text"))

    assert "cannot identify image file" in record.error
