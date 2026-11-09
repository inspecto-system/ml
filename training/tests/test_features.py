import pytest
from PIL import Image

from inspecto_ml.features import extract
from tests.fakes import tiny_backbone


def test_should_extract_one_feature_row_per_image_across_batches() -> None:
    images = [Image.new("RGB", (10, 10), (value, 0, 255)) for value in (0, 51, 102, 153, 204)]

    features = extract(tiny_backbone(), images, batch_size=2)

    assert features.shape == (5, 3)
    assert features[:, 0].tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert features[:, 2].tolist() == [1.0] * 5


def test_no_images_give_an_empty_matrix() -> None:
    assert extract(tiny_backbone(), []).shape == (0, 0)
