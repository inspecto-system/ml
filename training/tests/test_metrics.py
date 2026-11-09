import numpy as np
import pytest

from inspecto_ml.metrics import evaluate

CLASSES = ["CRACK", "NO_DEFECT", "STAIN", "MOLD"]


def test_should_compute_macro_f1_accuracy_top3_and_confusion() -> None:
    truth = ["CRACK", "CRACK", "STAIN", "NO_DEFECT"]
    probabilities = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],  # CRACK, right
            [0.1, 0.6, 0.2, 0.1],  # NO_DEFECT, wrong; CRACK only 3rd? no: 0.1 is 3rd/4th
            [0.1, 0.1, 0.5, 0.3],  # STAIN, right
            [0.2, 0.5, 0.2, 0.1],  # NO_DEFECT, right
        ]
    )

    result = evaluate(truth, probabilities, CLASSES)

    assert result["images"] == 4
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["per_class"]["CRACK"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert result["per_class"]["MOLD"]["support"] == 0
    assert result["confusion"]["matrix"][0] == [1, 1, 0, 0]
    assert result["top3_accuracy"] == pytest.approx(1.0)


def test_should_restrict_macro_f1_to_given_labels_but_count_other_predictions_as_errors() -> None:
    truth = ["CRACK", "STAIN"]
    probabilities = np.array([[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05]])

    result = evaluate(truth, probabilities, CLASSES, labels=["CRACK", "STAIN"])

    assert result["per_class"].keys() == {"CRACK", "STAIN"}
    assert result["macro_f1"] == pytest.approx(0.5)  # CRACK 1.0, STAIN 0.0
    assert result["accuracy"] == pytest.approx(0.5)


def test_top3_misses_a_class_ranked_fourth() -> None:
    result = evaluate(["MOLD"], np.array([[0.4, 0.3, 0.2, 0.1]]), CLASSES)

    assert result["top3_accuracy"] == 0.0
