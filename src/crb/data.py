"""Load, validate and split the dataset files."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import ValidationError

from crb.schema import Block, Collection, CollectionsFile

VALIDATION_SEED = 42
VALIDATION_GROUP_FRACTION = 0.10


def load_collections(path: str | Path = "data/collections.yaml") -> list[Collection]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CollectionsFile.model_validate(raw).collections


def iter_blocks(path: str | Path) -> Iterator[Block]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Block.model_validate(json.loads(line))


def load_blocks(path: str | Path) -> list[Block]:
    return list(iter_blocks(path))


def load_split(train_path: str | Path, test_path: str | Path) -> dict[str, list[Block]]:
    """Return {'train': [...], 'validation': [...], 'test': [...]}.

    data/train.jsonl carries both train and validation rows; the split field
    separates them.
    """
    out: dict[str, list[Block]] = {"train": [], "validation": [], "test": []}
    for block in iter_blocks(train_path):
        out[block.split].append(block)
    out["test"] = load_blocks(test_path)
    return out


def validation_group_ids(group_ids: list[str], seed: int = VALIDATION_SEED) -> set[str]:
    """The deterministic 10% of train groups that become the validation split."""
    ordered = sorted(set(group_ids))
    k = round(len(ordered) * VALIDATION_GROUP_FRACTION)
    rng = random.Random(seed)
    return set(rng.sample(ordered, k))


def blocks_by_collection(blocks: list[Block]) -> dict[str, list[Block]]:
    out: dict[str, list[Block]] = {}
    for b in blocks:
        for label in b.labels:
            out.setdefault(label, []).append(b)
    return out


__all__ = [
    "VALIDATION_SEED",
    "VALIDATION_GROUP_FRACTION",
    "load_collections",
    "iter_blocks",
    "load_blocks",
    "load_split",
    "validation_group_ids",
    "blocks_by_collection",
    "ValidationError",
]
