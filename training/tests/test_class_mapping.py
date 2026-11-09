from pathlib import Path

import pytest

from inspecto_ml.class_mapping import (
    InvalidMappingFileError,
    Role,
    UnknownClassError,
    load_mapping,
    parse_mapping,
    predicted_categories,
)
from inspecto_ml.taxonomy import InvalidClassMappingError

# Every class found by the ML-01 audit (platform/docs/ML_DOCUMENTATION.md §3.1-3.3).
AUDITED_CLASSES = {
    "bd3": {"algae", "major_crack", "minor_crack", "peeling", "plain", "spalling", "stain"},
    "sdnet2018": {"D/CD", "D/UD", "P/CP", "P/UP", "W/CW", "W/UW"},
    "fbd": {"crack", "spalling", "stain", "vegetation"},
}


def test_approved_mapping_describes_every_audited_class_once() -> None:
    mapping = load_mapping()

    assert mapping.keys() == AUDITED_CLASSES.keys()
    for name, classes in AUDITED_CLASSES.items():
        assert mapping[name].classes.keys() | mapping[name].excluded == classes


def test_approved_mapping_targets() -> None:
    mapping = load_mapping()

    assert mapping["bd3"].target_of("algae") == "MOLD"
    assert mapping["bd3"].target_of("minor_crack") == "CRACK"
    assert mapping["bd3"].target_of("plain") is None
    assert mapping["sdnet2018"].target_of("W/UW") is None
    assert not mapping["sdnet2018"].is_used("P/CP")
    assert mapping["fbd"].role is Role.EVALUATION


def test_model_suggests_five_categories() -> None:
    assert predicted_categories(load_mapping()) == ["CRACK", "MOLD", "PEELING", "SPALLING", "STAIN"]


def test_target_of_excluded_or_unknown_class_fails() -> None:
    fbd = load_mapping()["fbd"]

    with pytest.raises(UnknownClassError, match="excluded"):
        fbd.target_of("vegetation")
    with pytest.raises(UnknownClassError, match="not in the mapping"):
        fbd.target_of("rust")


def test_should_load_mapping_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "mapping.toml"
    path.write_text(
        '[x]\nrole = "train"\nexcluded = []\n[x.classes]\ncrack = "CRACK"\n', encoding="utf-8"
    )

    assert load_mapping(path)["x"].classes == {"crack": "CRACK"}


@pytest.mark.parametrize(
    "text",
    [
        '[x]\nrole = "test"\nexcluded = []\n[x.classes]\na = "CRACK"\n',
        '[x]\nrole = "train"\n[x.classes]\na = "CRACK"\n',
        '[x]\nrole = "train"\nexcluded = []\n',
    ],
    ids=["unknown role", "no excluded", "no classes"],
)
def test_should_reject_incomplete_sections(text: str) -> None:
    with pytest.raises(InvalidMappingFileError, match=r"\[x\]: needs role"):
        parse_mapping(text)


def test_should_reject_class_both_mapped_and_excluded() -> None:
    text = '[x]\nrole = "train"\nexcluded = ["a"]\n[x.classes]\na = "CRACK"\n'

    with pytest.raises(InvalidMappingFileError, match="both mapped and excluded: a"):
        parse_mapping(text)


def test_should_reject_unknown_category_code() -> None:
    text = '[x]\nrole = "train"\nexcluded = []\n[x.classes]\na = "RUST"\n'

    with pytest.raises(InvalidClassMappingError, match="a -> RUST"):
        parse_mapping(text)
