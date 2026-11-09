import io
import json
from pathlib import Path

import numpy as np

from inspecto_ml.audit.__main__ import main
from inspecto_ml.audit.duplicates import DuplicateGroup, Scope
from inspecto_ml.audit.records import ImageRecord
from inspecto_ml.audit.report import dataset_summary, duplicate_summary
from inspecto_ml.datasets import SDNET_INNER_ARCHIVE
from tests.audit.helpers import ImageFactory, jpeg_bytes, write_zip


def _record(label: str, width: int, error: str = "") -> ImageRecord:
    return ImageRecord(
        "bd3", label, f"{label}{width}", "sha", 1, width, width // 2, "JPEG", "RGB", error
    )


def test_dataset_summary_counts_classes_sizes_and_broken_files() -> None:
    records = [
        _record("stain", 100),
        _record("algae", 300),
        _record("algae", 200),
        _record("algae", 0, "bad"),
    ]

    summary = dataset_summary(records)

    assert summary["images"] == len(records)
    assert summary["classes"] == {"algae": 3, "stain": 1}
    assert summary["formats"] == {"JPEG": 3}
    assert summary["width"] == {"min": 100, "median": 200, "max": 300}
    assert summary["broken"] == [{"key": "algae0", "error": "bad"}]


def test_dataset_summary_of_only_broken_files_has_no_sizes() -> None:
    assert dataset_summary([_record("algae", 0, "bad")])["width"] == {}


def test_duplicate_summary_counts_extra_copies_per_scope() -> None:
    three = DuplicateGroup(Scope.SAME_CLASS, (_record("a", 1), _record("a", 2), _record("a", 3)))
    two = DuplicateGroup(Scope.SAME_CLASS, (_record("b", 1), _record("b", 2)))

    summary = duplicate_summary([two, three])

    assert summary["same_class"]["groups"] == 2
    assert summary["same_class"]["extra_copies"] == 3
    assert summary["same_class"]["examples"][0] == ["bd3:a:a1", "bd3:a:a2", "bd3:a:a3"]
    assert summary["same_class"]["by_dataset"] == {"bd3": {"groups": 2, "extra_copies": 3}}
    assert summary["cross_class"] == {
        "groups": 0,
        "extra_copies": 0,
        "by_dataset": {},
        "examples": [],
    }


def _npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array)
    path.write_bytes(buffer.getvalue())


def test_command_line_writes_report_of_all_datasets(tmp_path: Path, picture: ImageFactory) -> None:
    same = jpeg_bytes(picture(1))
    bd3 = write_zip(
        tmp_path / "bd3.zip",
        {"t/algae/1.jpg": same, "t/stain/2.jpg": same, "t/stain/3.jpg": jpeg_bytes(picture(2))},
    )
    inner = write_zip(tmp_path / "inner.zip", {"S/W/CW/1.jpg": jpeg_bytes(picture(3))})
    sdnet = write_zip(tmp_path / "sdnet.zip", {SDNET_INNER_ARCHIVE: inner.read_bytes()})
    fbd = tmp_path / "fbd"
    fbd.mkdir()
    for part in ("training", "validation"):
        _npy(fbd / f"{part}_images.npy", np.asarray(picture(4).resize((8, 8)))[np.newaxis])
        _npy(fbd / f"{part}_labels.npy", np.array([3]))
    out = tmp_path / "report.json"

    main(["--bd3", str(bd3), "--sdnet", str(sdnet), "--fbd", str(fbd), "--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["datasets"]["bd3"]["classes"] == {"algae": 1, "stain": 2}
    assert report["datasets"]["sdnet2018"]["classes"] == {"W/CW": 1}
    assert report["datasets"]["fbd"]["classes"] == {"vegetation": 2}
    assert set(report["datasets"]["fbd"]["files"]) == {
        "training_images.npy",
        "training_labels.npy",
        "validation_images.npy",
        "validation_labels.npy",
    }
    assert report["exact_duplicates"]["cross_class"]["groups"] == 1
    assert report["exact_duplicates"]["same_class"]["groups"] == 1  # the two FBD arrays
    assert report["near_duplicate_max_distance"] == 4
