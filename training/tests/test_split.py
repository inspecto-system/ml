import json
from collections import Counter
from pathlib import Path

import pytest

from inspecto_ml.audit.records import ImageRecord
from inspecto_ml.class_mapping import parse_mapping
from inspecto_ml.split import Entry, LeakError, Part, build_split, check_no_leaks, part_counts
from inspecto_ml.split_cli import main
from tests.audit.helpers import ImageFactory, jpeg_bytes, write_zip

MAPPING = parse_mapping(
    """
[bd3]
role = "train"
excluded = ["junk"]
[bd3.classes]
major_crack = "CRACK"
minor_crack = "CRACK"
stain = "STAIN"
plain = "NO_DEFECT"

[fbd]
role = "evaluation"
excluded = []
[fbd.classes]
crack = "CRACK"
"""
)


def _record(
    key: str, label: str, sha: str = "", dhash: int | None = None, **kw: str
) -> ImageRecord:
    unique_hash = dhash if dhash is not None else hash(key) & ((1 << 128) - 1)
    return ImageRecord(
        kw.get("dataset", "bd3"),
        label,
        key,
        sha or f"sha-{key}",
        unique_hash,
        error=kw.get("error", ""),
    )


def test_should_skip_unused_drop_copies_and_contradictions() -> None:
    records = [
        _record("a", "stain"),
        _record("a-copy", "stain", sha="sha-a"),  # duplicate in the same category: one kept
        _record("b", "major_crack", sha="same"),
        _record("b2", "minor_crack", sha="same"),  # both CRACK: a copy, not a contradiction
        _record("c", "stain", sha="conflict"),
        _record("c2", "plain", sha="conflict"),  # STAIN vs NO_DEFECT: both dropped
        _record("d", "junk"),  # excluded class
        _record("e", "stain", dataset="unknown"),  # dataset not in the mapping
        _record("f", "stain", error="broken"),
    ]

    entries, stats = build_split(records, MAPPING, seed=1, max_distance=4)

    assert sorted(entry.key for entry in entries) == ["a", "b"]
    assert (stats.unused, stats.duplicate_copies, stats.contradictory) == (3, 2, 2)


def test_should_keep_near_duplicates_in_one_part() -> None:
    records = [_record(f"n{i}", "stain", dhash=0b1 << i) for i in range(3)]  # chain within 2 bits
    records += [_record(f"s{i}", "stain") for i in range(40)]

    entries, _ = build_split(records, MAPPING, seed=3, max_distance=4)

    near = [entry for entry in entries if entry.key.startswith("n")]
    assert len({entry.group for entry in near}) == 1
    assert len({entry.part for entry in near}) == 1


def test_should_split_each_category_70_15_15_and_send_evaluation_data_to_external() -> None:
    records = [_record(f"c{i}", "major_crack") for i in range(100)]
    records += [_record(f"p{i}", "plain") for i in range(20)]
    records += [_record(f"x{i}", "crack", dataset="fbd") for i in range(5)]

    entries, _ = build_split(records, MAPPING, seed=7, max_distance=4)

    counts = part_counts(entries)
    assert counts["train"] == {"CRACK": 70, "NO_DEFECT": 14}
    assert counts["val"] == {"CRACK": 15, "NO_DEFECT": 3}
    assert counts["test"] == {"CRACK": 15, "NO_DEFECT": 3}
    assert counts["external"] == {"CRACK": 5}


def test_split_depends_on_seed_only() -> None:
    records = [_record(f"c{i}", "stain") for i in range(50)]

    first, _ = build_split(records, MAPPING, seed=5, max_distance=4)
    again, _ = build_split(list(reversed(records)), MAPPING, seed=5, max_distance=4)
    other, _ = build_split(records, MAPPING, seed=6, max_distance=4)

    assert first == again
    assert first != other


def _entry(key: str, sha: str, group: str, part: Part) -> Entry:
    return Entry("bd3", "stain", key, sha, "STAIN", group, part)


def test_check_no_leaks_finds_content_or_group_in_two_parts() -> None:
    with pytest.raises(LeakError, match="sha256 s is in train and test"):
        check_no_leaks([_entry("a", "s", "a", Part.TRAIN), _entry("b", "s", "b", Part.TEST)])
    with pytest.raises(LeakError, match="group g is in val and train"):
        check_no_leaks([_entry("a", "s1", "g", Part.VAL), _entry("b", "s2", "g", Part.TRAIN)])


def test_command_line_writes_manifest(tmp_path: Path, picture: ImageFactory) -> None:
    members = {f"t/stain/{i}.jpg": jpeg_bytes(picture(i)) for i in range(20)}
    members["t/plain/x.jpg"] = members["t/stain/0.jpg"]  # contradiction: dropped
    bd3 = write_zip(tmp_path / "bd3.zip", members)
    mapping = tmp_path / "mapping.toml"
    mapping.write_text(
        '[bd3]\nrole = "train"\nexcluded = []\n'
        '[bd3.classes]\nstain = "STAIN"\nplain = "NO_DEFECT"\n',
        encoding="utf-8",
    )
    out = tmp_path / "split.json"

    main(["--bd3", str(bd3), "--mapping", str(mapping), "--seed", "1", "--out", str(out)])

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["dropped"] == {"unused": 0, "duplicate_copies": 0, "contradictory": 2}
    assert Counter(row["part"] for row in manifest["entries"]) == {"train": 13, "val": 3, "test": 3}
    assert manifest["source_files"].keys() == {"bd3.zip"}
    assert len(manifest["version"]) == 64
