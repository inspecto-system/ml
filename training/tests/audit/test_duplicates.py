from inspecto_ml.audit.duplicates import (
    Scope,
    exact_groups,
    near_groups,
    similar_hash_pairs,
)
from inspecto_ml.audit.records import ImageRecord


def _record(
    key: str, sha: str, dhash: int | None, label: str = "crack", dataset: str = "bd3"
) -> ImageRecord:
    return ImageRecord(dataset, label, key, sha, dhash)


def test_exact_groups_by_scope() -> None:
    records = [
        _record("a", "s1", 1),
        _record("b", "s1", 1),
        _record("c", "s2", 2, label="stain"),
        _record("d", "s2", 2, label="peeling"),
        _record("e", "s3", 3),
        _record("f", "s3", 3, dataset="fbd"),
        _record("g", "s4", 4),
    ]

    groups = {group.scope: [r.key for r in group.members] for group in exact_groups(records)}

    assert groups == {
        Scope.SAME_CLASS: ["a", "b"],
        Scope.CROSS_CLASS: ["c", "d"],
        Scope.CROSS_DATASET: ["e", "f"],
    }


def test_similar_hash_pairs_finds_pairs_within_distance_only() -> None:
    base = 0b1011_0000
    hashes = [base, base ^ 0b1, base ^ 0b11, base ^ (0b111 << 40), 1 << 63]

    pairs = similar_hash_pairs(hashes, max_distance=2)

    assert pairs == {
        (base, base ^ 0b1),
        (base, base ^ 0b11),
        (base ^ 0b1, base ^ 0b11),
    }


def test_similar_hash_pairs_finds_bits_spread_over_chunks() -> None:
    first = 0
    second = (1 << 0) | (1 << 30) | (1 << 127)

    assert similar_hash_pairs([first, second], max_distance=3) == {(first, second)}
    assert similar_hash_pairs([first, second], max_distance=2) == set()


def test_near_groups_join_chains_and_skip_pure_exact_copies() -> None:
    records = [
        _record("a", "s1", 0b000),
        _record("b", "s2", 0b001),
        _record("c", "s3", 0b011, label="stain"),
        _record("d", "s4", 0b111 << 50),
        _record("e", "s4", 0b111 << 50),
        _record("broken", "s5", None),
    ]

    groups = near_groups(records, max_distance=1)

    assert len(groups) == 1
    assert groups[0].scope is Scope.CROSS_CLASS
    assert sorted(r.key for r in groups[0].members) == ["a", "b", "c"]
