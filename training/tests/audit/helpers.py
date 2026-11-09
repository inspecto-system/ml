import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

ImageFactory = Callable[[int], Image.Image]


def make_picture(seed: int) -> Image.Image:
    """A distinct, deterministic 64x48 RGB picture per seed."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    return Image.fromarray(np.kron(blocks, np.ones((8, 8, 1), dtype=np.uint8)))


def jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path
