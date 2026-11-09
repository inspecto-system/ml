"""Exact and near-duplicate search over audit records.

Exact duplicates share the SHA-256 of their content. Near-duplicates have difference hashes within
a small Hamming distance. To avoid comparing every pair, hashes are split into ``max_distance + 1``
chunks: two hashes within the distance must agree on at least one chunk (pigeonhole principle),
so only hashes sharing a chunk are compared.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from inspecto_ml.audit.records import ImageRecord

HASH_BITS = 128  # records.difference_hash


class Scope(StrEnum):
    """Where the copies of a duplicate lie — what it means for training."""

    SAME_CLASS = "same_class"  # inflates the class; leaks between splits
    CROSS_CLASS = "cross_class"  # the same picture with different labels: label noise
    CROSS_DATASET = "cross_dataset"  # one picture in two datasets


@dataclass(frozen=True)
class DuplicateGroup:
    """Images considered copies of each other."""

    scope: Scope
    members: tuple[ImageRecord, ...]


def scope_of(members: Sequence[ImageRecord]) -> Scope:
    """Classify a group of copies by the datasets and classes of its members."""
    if len({member.dataset for member in members}) > 1:
        return Scope.CROSS_DATASET
    if len({member.label for member in members}) > 1:
        return Scope.CROSS_CLASS
    return Scope.SAME_CLASS


def exact_groups(records: Iterable[ImageRecord]) -> list[DuplicateGroup]:
    """Groups of two or more images with identical content."""
    by_digest: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_digest[record.sha256].append(record)
    groups = [members for members in by_digest.values() if len(members) > 1]
    return [DuplicateGroup(scope_of(members), tuple(members)) for members in groups]


def _chunks(value: int, parts: int) -> list[tuple[int, int]]:
    bounds = [part * HASH_BITS // parts for part in range(parts + 1)]
    return [
        (part, (value >> bounds[part]) & ((1 << (bounds[part + 1] - bounds[part])) - 1))
        for part in range(parts)
    ]


def similar_hash_pairs(hashes: Iterable[int], max_distance: int) -> set[tuple[int, int]]:
    """Pairs of distinct hashes whose Hamming distance is at most ``max_distance``."""
    parts = max_distance + 1
    buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for value in set(hashes):
        for chunk in _chunks(value, parts):
            buckets[chunk].add(value)
    pairs: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        for first, second in combinations(sorted(bucket), 2):
            if (first ^ second).bit_count() <= max_distance:
                pairs.add((first, second))
    return pairs


def _union_find_groups(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    parent = {node: node for node in nodes}

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in edges:
        parent[root(first)] = root(second)
    groups: dict[int, set[int]] = defaultdict(set)
    for node in parent:
        groups[root(node)].add(node)
    return list(groups.values())


def near_groups(records: Sequence[ImageRecord], max_distance: int) -> list[DuplicateGroup]:
    """Groups of images whose hashes are equal or connected by pairs within ``max_distance``.

    Exact copies are excluded: a group counts only if it has at least two different contents.
    """
    by_hash: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.dhash is not None:
            by_hash[record.dhash].append(record)
    edges = similar_hash_pairs(by_hash, max_distance)
    result = []
    for hash_group in _union_find_groups(by_hash, edges):
        members = [record for value in sorted(hash_group) for record in by_hash[value]]
        if len({member.sha256 for member in members}) > 1:
            result.append(DuplicateGroup(scope_of(members), tuple(members)))
    return result
