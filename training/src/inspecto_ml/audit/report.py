"""Summary of an audit: what goes into the data card of each dataset."""

from collections import Counter
from collections.abc import Sequence
from statistics import median
from typing import Any

from inspecto_ml.audit.duplicates import DuplicateGroup, Scope
from inspecto_ml.audit.records import ImageRecord

EXAMPLE_GROUPS = 10


def _size_stats(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    return {"min": min(values), "median": median(values), "max": max(values)}


def dataset_summary(records: Sequence[ImageRecord]) -> dict[str, Any]:
    """Class balance, image properties and broken files of one dataset."""
    decoded = [record for record in records if not record.error]
    return {
        "images": len(records),
        "classes": dict(sorted(Counter(record.label for record in records).items())),
        "formats": dict(Counter(record.image_format for record in decoded).most_common()),
        "modes": dict(Counter(record.mode for record in decoded).most_common()),
        "width": _size_stats([record.width for record in decoded]),
        "height": _size_stats([record.height for record in decoded]),
        "broken": [
            {"key": record.key, "error": record.error} for record in records if record.error
        ],
    }


def _member(record: ImageRecord) -> str:
    return f"{record.dataset}:{record.label}:{record.key}"


def _counts(groups: Sequence[DuplicateGroup]) -> dict[str, int]:
    return {
        "groups": len(groups),
        "extra_copies": sum(len(group.members) - 1 for group in groups),
    }


def _by_dataset(groups: Sequence[DuplicateGroup]) -> dict[str, dict[str, int]]:
    datasets = sorted({group.members[0].dataset for group in groups})
    return {
        dataset: _counts([group for group in groups if group.members[0].dataset == dataset])
        for dataset in datasets
    }


def duplicate_summary(groups: Sequence[DuplicateGroup]) -> dict[str, Any]:
    """Per scope: groups and extra copies (also per dataset) and the largest example groups."""
    summary: dict[str, Any] = {}
    for scope in Scope:
        in_scope = [group for group in groups if group.scope is scope]
        largest = sorted(in_scope, key=lambda group: -len(group.members))[:EXAMPLE_GROUPS]
        summary[scope.value] = {
            **_counts(in_scope),
            "by_dataset": {} if scope is Scope.CROSS_DATASET else _by_dataset(in_scope),
            "examples": [[_member(record) for record in group.members] for group in largest],
        }
    return summary
