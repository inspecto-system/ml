"""Baselines of ML-03: the majority class and a linear probe on frozen ResNet-50 features.

The probe is a logistic regression on standardised features with balanced class weights; its
regularisation ``C`` is chosen by macro-F1 on the validation part. The test and external parts
are evaluated once, with the chosen model.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from inspecto_ml.metrics import evaluate
from inspecto_ml.split import Part

C_GRID = (0.01, 0.1, 1.0)
MAX_ITERATIONS = 3000


@dataclass(frozen=True)
class PartData:
    """Features and targets of one part of the split."""

    features: np.ndarray
    targets: list[str]


def select_rows(order: Sequence[str], features: np.ndarray, wanted: Sequence[str]) -> np.ndarray:
    """Rows of ``features`` (whose rows follow ``order``) for the ``wanted`` uids, in that order."""
    position = {uid: row for row, uid in enumerate(order)}
    return features[[position[uid] for uid in wanted]]


def _scores(model: Any, data: PartData, labels: Sequence[str] | None = None) -> dict[str, Any]:  # noqa: ANN401 — any fitted sklearn classifier
    probabilities = model.predict_proba(data.features)
    return evaluate(data.targets, probabilities, list(model.classes_), labels)


def _external_labels(parts: dict[Part, PartData]) -> list[str] | None:
    external = parts.get(Part.EXTERNAL)
    return sorted(set(external.targets)) if external else None


def majority_baseline(parts: dict[Part, PartData]) -> dict[str, Any]:
    """Always the most frequent training category."""
    train = parts[Part.TRAIN]
    model = DummyClassifier(strategy="most_frequent").fit(train.features, train.targets)
    return _evaluate_parts(model, parts) | {
        "predicts": str(model.classes_[np.argmax(model.class_prior_)])
    }


def probe(c: float, seed: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, class_weight="balanced", max_iter=MAX_ITERATIONS, random_state=seed
        ),
    )


def linear_probe(parts: dict[Part, PartData], seed: int) -> dict[str, Any]:
    """Logistic regression with ``C`` chosen on validation macro-F1."""
    train, val = parts[Part.TRAIN], parts[Part.VAL]
    search: dict[str, float] = {}
    best: tuple[float, float, Pipeline] | None = None
    for c in C_GRID:
        model = probe(c, seed).fit(train.features, train.targets)
        score = float(_scores(model, val)["macro_f1"])
        search[str(c)] = score
        if best is None or score > best[0]:
            best = (score, c, model)
    if best is None:
        msg = "C_GRID is empty"
        raise ValueError(msg)
    _, chosen, model = best
    return _evaluate_parts(model, parts) | {"C": chosen, "val_macro_f1_by_C": search}


def _evaluate_parts(model: Any, parts: dict[Part, PartData]) -> dict[str, Any]:  # noqa: ANN401 — any fitted sklearn classifier
    results = {part.value: _scores(model, parts[part]) for part in (Part.VAL, Part.TEST)}
    if Part.EXTERNAL in parts:
        results[Part.EXTERNAL.value] = _scores(model, parts[Part.EXTERNAL], _external_labels(parts))
    return results
