from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from crb.data import load_collections, load_split  # noqa: E402


@pytest.fixture(scope="session")
def collections():
    return load_collections(REPO / "data" / "collections.yaml")


@pytest.fixture(scope="session")
def splits():
    return load_split(REPO / "data" / "train.jsonl", REPO / "data" / "test.jsonl")


@pytest.fixture(scope="session")
def all_items(splits):
    return splits["train"] + splits["validation"] + splits["test"]
