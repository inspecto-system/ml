import pytest

from inspecto_ml.taxonomy import (
    SYSTEM_CATEGORY_CODES,
    InvalidClassMappingError,
    target_categories,
    validate_class_mapping,
)


def test_system_categories_match_core_dictionary() -> None:
    # The 14 codes of core-service V4__dictionaries.sql.
    expected = {
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

    assert expected == SYSTEM_CATEGORY_CODES


def test_should_accept_mapping_to_system_codes_and_no_defect() -> None:
    validate_class_mapping({"major crack": "CRACK", "minor crack": "CRACK", "normal": None})


def test_should_reject_empty_mapping() -> None:
    with pytest.raises(InvalidClassMappingError, match="empty"):
        validate_class_mapping({})


def test_should_name_every_unknown_code() -> None:
    with pytest.raises(InvalidClassMappingError, match=r"algae -> ALGAE, rust -> RUST"):
        validate_class_mapping({"rust": "RUST", "algae": "ALGAE", "stain": "STAIN"})


def test_should_return_distinct_sorted_targets_without_no_defect() -> None:
    mapping = {"stain": "STAIN", "major crack": "CRACK", "minor crack": "CRACK", "normal": None}

    assert target_categories(mapping) == ["CRACK", "STAIN"]
