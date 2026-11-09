"""System defect categories of Inspecto and validation of dataset class mappings (ML-02).

The codes are the 14 system categories of core-service (migration V4__dictionaries.sql).
A dataset class maps either to one of them or to ``None`` — an image without a defect,
for which no suggestion is shown.
"""

from collections.abc import Mapping
from typing import Final

SYSTEM_CATEGORY_CODES: Final[frozenset[str]] = frozenset(
    {
        "CRACK",
        "PEELING",
        "STAIN",
        "MOLD",
        "SPALLING",
        "UNEVEN_SURFACE",
        "WALLPAPER",
        "TILE",
        "FLOOR_COVERING",
        "WINDOW",
        "DOOR",
        "PLUMBING_LEAK",
        "ELECTRICAL_FAULT",
        "OTHER",
    }
)


class InvalidClassMappingError(ValueError):
    """A dataset class mapping refers to categories that Inspecto does not have."""


def validate_class_mapping(mapping: Mapping[str, str | None]) -> None:
    """Check that every dataset class maps to a system category code or to ``None``.

    Raises:
        InvalidClassMappingError: the mapping is empty or names unknown category codes.
    """
    if not mapping:
        msg = "class mapping is empty"
        raise InvalidClassMappingError(msg)
    unknown = sorted(
        f"{dataset_class} -> {code}"
        for dataset_class, code in mapping.items()
        if code is not None and code not in SYSTEM_CATEGORY_CODES
    )
    if unknown:
        msg = f"unknown category codes: {', '.join(unknown)}"
        raise InvalidClassMappingError(msg)


def target_categories(mapping: Mapping[str, str | None]) -> list[str]:
    """Return the sorted category codes a validated mapping can predict (without ``None``)."""
    validate_class_mapping(mapping)
    return sorted({code for code in mapping.values() if code is not None})
