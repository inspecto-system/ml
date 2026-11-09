"""Approved mapping of dataset classes to Inspecto categories (ML-02), read from TOML."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any

from inspecto_ml.taxonomy import target_categories, validate_class_mapping

NO_DEFECT = "NO_DEFECT"


class Role(StrEnum):
    """What a dataset is used for."""

    TRAIN = "train"
    EVALUATION = "evaluation"


class InvalidMappingFileError(ValueError):
    """The mapping file does not follow the format of ``class_mapping.toml``."""


class UnknownClassError(KeyError):
    """A dataset class is neither mapped nor excluded."""


@dataclass(frozen=True)
class DatasetMapping:
    """How the classes of one dataset are used; ``None`` in ``classes`` means "no defect"."""

    role: Role
    classes: Mapping[str, str | None]
    excluded: frozenset[str]

    def target_of(self, label: str) -> str | None:
        """Category code of a dataset class, ``None`` for "no defect".

        Raises:
            UnknownClassError: the class is excluded or not described in the mapping.
        """
        if label in self.classes:
            return self.classes[label]
        reason = "excluded" if label in self.excluded else "not in the mapping"
        msg = f"class {label!r} is {reason}"
        raise UnknownClassError(msg)

    def is_used(self, label: str) -> bool:
        """Whether images of the class take part in training or evaluation."""
        return label in self.classes


def _parse_dataset(name: str, section: Any) -> DatasetMapping:  # noqa: ANN401 — raw TOML value
    try:
        role = Role(section["role"])
        raw_classes = section["classes"]
        excluded = frozenset(section["excluded"])
    except (KeyError, TypeError, ValueError) as error:
        msg = f"[{name}]: needs role (train/evaluation), classes and excluded"
        raise InvalidMappingFileError(msg) from error
    classes = {label: None if code == NO_DEFECT else code for label, code in raw_classes.items()}
    validate_class_mapping(classes)
    if overlap := sorted(excluded & classes.keys()):
        msg = f"[{name}]: classes both mapped and excluded: {', '.join(overlap)}"
        raise InvalidMappingFileError(msg)
    return DatasetMapping(role, classes, excluded)


def parse_mapping(text: str) -> dict[str, DatasetMapping]:
    """Parse and validate a mapping in the format of ``class_mapping.toml``."""
    document = tomllib.loads(text)
    return {name: _parse_dataset(name, section) for name, section in document.items()}


def load_mapping(path: Path | None = None) -> dict[str, DatasetMapping]:
    """The approved mapping shipped with the package, or the one in ``path``."""
    if path is None:
        text = files("inspecto_ml").joinpath("class_mapping.toml").read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return parse_mapping(text)


def predicted_categories(mapping: Mapping[str, DatasetMapping]) -> list[str]:
    """Category codes the model can suggest: those of the training datasets."""
    merged = {
        f"{name}:{label}": code
        for name, dataset in mapping.items()
        if dataset.role is Role.TRAIN
        for label, code in dataset.classes.items()
    }
    return target_categories(merged)
