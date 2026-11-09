from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml_service.classifier import (
    Classifier,
    InvalidImageError,
    ModelInfo,
    Prediction,
    preprocess,
    suggestions,
)
from tests.support import jpeg, write_model

INFO = ModelInfo(
    version="v",
    classes=("CRACK", "MOLD", "NO_DEFECT", "STAIN", "PEELING"),
    no_defect="NO_DEFECT",
    input_size=8,
    resize=10,
    mean=(0, 0, 0),
    std=(1, 1, 1),
    threshold=0.5,
)


def test_suggestions_are_top_three_categories_without_no_defect() -> None:
    probabilities = np.array([0.6, 0.15, 0.1, 0.1, 0.05])

    assert suggestions(probabilities, INFO) == [
        Prediction("CRACK", 0.6),
        Prediction("MOLD", 0.15),
        Prediction("STAIN", 0.1),
    ]


@pytest.mark.parametrize(
    "probabilities",
    [[0.2, 0.1, 0.6, 0.05, 0.05], [0.45, 0.2, 0.15, 0.1, 0.1]],
    ids=["no defect wins", "below threshold"],
)
def test_no_suggestion(probabilities: list[float]) -> None:
    assert suggestions(np.array(probabilities), INFO) == []


def test_classifier_answers_by_the_model(tmp_path: Path) -> None:
    classifier = Classifier(write_model(tmp_path))

    red = classifier.classify(jpeg((250, 10, 10)))
    green = classifier.classify(jpeg((10, 250, 10)))

    assert [p.code for p in red] == ["CRACK", "STAIN"]
    assert red[0].probability > 0.9
    assert green == []
    assert classifier.info.version == "colour-test"


def test_classifier_rejects_bytes_that_are_not_an_image(tmp_path: Path) -> None:
    classifier = Classifier(write_model(tmp_path))

    with pytest.raises(InvalidImageError):
        classifier.classify(b"not an image")


def test_preprocessing_keeps_the_shorter_side_and_crops_the_centre() -> None:

    image = Image.new("RGB", (40, 20), (255, 0, 0))
    array = preprocess(image, INFO)

    assert array.shape == (3, 8, 8)
    assert np.allclose(array[0], 1.0, atol=0.01)
    assert np.allclose(array[1:], 0.0, atol=0.01)
