"""Quality metrics of ML_DOCUMENTATION.md §5 for a probabilistic classifier."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

TOP_K = 3


def evaluate(
    truth: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Macro-F1, accuracy, top-3 accuracy, per-class precision/recall/F1 and confusion matrix.

    ``labels`` restricts macro-F1 and the per-class table to the classes present in the data
    (FBD has no "no defect" images); predictions of other classes still count as errors.
    """
    names = np.asarray(classes)
    predicted = names[np.argmax(probabilities, axis=1)]
    scored = list(labels if labels is not None else classes)
    top = np.argsort(-probabilities, axis=1)[:, :TOP_K]
    index = {name: position for position, name in enumerate(classes)}
    in_top = [index[label] in row for label, row in zip(truth, top, strict=True)]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=scored, zero_division=0
    )
    return {
        "images": len(truth),
        "macro_f1": float(
            f1_score(truth, predicted, labels=scored, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(truth, predicted)),
        "top3_accuracy": float(np.mean(in_top)),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(scored, precision, recall, f1, support, strict=True)
        },
        "confusion": {
            "labels": list(classes),
            "matrix": confusion_matrix(truth, predicted, labels=list(classes)).tolist(),
        },
    }
