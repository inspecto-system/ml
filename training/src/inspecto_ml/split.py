"""Train/validation/test split without leaks (ML-03, rules of ML_DOCUMENTATION.md §3.5).

1. Images of excluded classes, broken files and datasets absent from the mapping are skipped.
2. Exact copies (same SHA-256): one is kept; if the copies map to different categories, all
   of them are dropped as contradictory labelling.
3. Near-duplicates (difference hash within ``max_distance``) form a group that goes to one part.
4. Groups of training datasets are spread over train/val/test (70/15/15) per category, in a
   seeded random order; evaluation-only datasets (FBD) go to the ``external`` part as a whole.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from inspecto_ml.audit.duplicates import near_groups
from inspecto_ml.audit.records import ImageRecord
from inspecto_ml.class_mapping import NO_DEFECT, DatasetMapping, Role


class Part(StrEnum):
    """Part of the split an image belongs to."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    EXTERNAL = "external"  # evaluation-only datasets: generalisation to another domain


FRACTIONS: Mapping[Part, float] = {Part.TRAIN: 0.70, Part.VAL: 0.15, Part.TEST: 0.15}


@dataclass(frozen=True)
class Entry:
    """One image of the split. ``target`` is a category code or ``NO_DEFECT``."""

    dataset: str
    label: str
    key: str
    sha256: str
    target: str
    group: str
    part: Part

    @property
    def uid(self) -> str:
        """Unique name of the image across datasets."""
        return f"{self.dataset}:{self.key}"


@dataclass(frozen=True)
class SplitStats:
    """What was dropped while building the split."""

    unused: int
    duplicate_copies: int
    contradictory: int


class LeakError(ValueError):
    """An image or a group of near-duplicates lies in more than one part."""


def _uid(record: ImageRecord) -> str:
    return f"{record.dataset}:{record.key}"


def _targets(
    records: Iterable[ImageRecord], mapping: Mapping[str, DatasetMapping]
) -> tuple[dict[str, tuple[ImageRecord, str]], int]:
    used: dict[str, tuple[ImageRecord, str]] = {}
    unused = 0
    for record in records:
        dataset = mapping.get(record.dataset)
        if record.error or dataset is None or not dataset.is_used(record.label):
            unused += 1
            continue
        used[_uid(record)] = (record, dataset.target_of(record.label) or NO_DEFECT)
    return used, unused


def _drop_exact_copies(
    items: dict[str, tuple[ImageRecord, str]],
) -> tuple[dict[str, tuple[ImageRecord, str]], int, int]:
    by_digest: dict[str, list[str]] = defaultdict(list)
    for uid, (record, _) in items.items():
        by_digest[record.sha256].append(uid)
    kept: dict[str, tuple[ImageRecord, str]] = {}
    copies = contradictory = 0
    for uids in by_digest.values():
        if len({items[uid][1] for uid in uids}) > 1:
            contradictory += len(uids)
            continue
        first = min(uids)
        kept[first] = items[first]
        copies += len(uids) - 1
    return kept, copies, contradictory


def _group_of(items: dict[str, tuple[ImageRecord, str]], max_distance: int) -> dict[str, str]:
    group = {uid: uid for uid in items}
    for near in near_groups([record for record, _ in items.values()], max_distance):
        members = [_uid(record) for record in near.members]
        name = min(members)
        for uid in members:
            group[uid] = name
    return group


def _spread(groups: Sequence[tuple[str, int]], seed: int) -> dict[str, Part]:
    """Assign groups (name, size) of one category to parts, each time to the most lacking part."""
    order = np.random.default_rng(seed).permutation(len(groups))
    total = sum(size for _, size in groups)
    filled = dict.fromkeys(FRACTIONS, 0)
    parts: dict[str, Part] = {}
    for index in order:
        name, size = groups[int(index)]
        part = max(FRACTIONS, key=lambda p: FRACTIONS[p] * total - filled[p])
        parts[name] = part
        filled[part] += size
    return parts


def _parts(
    items: dict[str, tuple[ImageRecord, str]],
    group: Mapping[str, str],
    mapping: Mapping[str, DatasetMapping],
    seed: int,
) -> dict[str, Part]:
    members: dict[str, list[str]] = defaultdict(list)
    for uid in sorted(items):
        members[group[uid]].append(uid)
    by_target: dict[str, list[tuple[str, int]]] = defaultdict(list)
    parts: dict[str, Part] = {}
    for name, uids in members.items():
        if mapping[items[uids[0]][0].dataset].role is Role.EVALUATION:
            parts[name] = Part.EXTERNAL
            continue
        target = Counter(items[uid][1] for uid in uids).most_common(1)[0][0]
        by_target[target].append((name, len(uids)))
    for target in sorted(by_target):
        parts |= _spread(by_target[target], seed)
    return parts


def build_split(
    records: Iterable[ImageRecord],
    mapping: Mapping[str, DatasetMapping],
    seed: int,
    max_distance: int,
) -> tuple[list[Entry], SplitStats]:
    """Split audited images by the rules of the module docstring."""
    used, unused = _targets(records, mapping)
    kept, copies, contradictory = _drop_exact_copies(used)
    group = _group_of(kept, max_distance)
    parts = _parts(kept, group, mapping, seed)
    entries = [
        Entry(
            record.dataset,
            record.label,
            record.key,
            record.sha256,
            target,
            group[uid],
            parts[group[uid]],
        )
        for uid, (record, target) in sorted(kept.items())
    ]
    check_no_leaks(entries)
    return entries, SplitStats(unused, copies, contradictory)


def check_no_leaks(entries: Iterable[Entry]) -> None:
    """Fail if the same content or the same near-duplicate group is in two parts."""
    for attribute in ("sha256", "group"):
        seen: dict[str, Part] = {}
        for entry in entries:
            value = getattr(entry, attribute)
            if seen.setdefault(value, entry.part) is not entry.part:
                msg = f"{attribute} {value} is in {seen[value]} and {entry.part}"
                raise LeakError(msg)


def part_counts(entries: Iterable[Entry]) -> dict[str, dict[str, int]]:
    """Number of images per part and target."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        counts[entry.part.value][entry.target] += 1
    return {part: dict(sorted(counter.items())) for part, counter in sorted(counts.items())}
