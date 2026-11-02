"""The dataset follows every Block 1 rule (mirrors scripts/validate_data.py)."""

from __future__ import annotations

from collections import Counter

from crb.schema import COLLECTION_IDS
from validate_data import validate  # scripts/ on sys.path via conftest

REPO_DATA = "data/train.jsonl", "data/test.jsonl"


def _run_validator():
    from pathlib import Path

    return validate(Path(REPO_DATA[0]), Path(REPO_DATA[1]))


def test_collection_ids_match_prd(collections):
    assert [c.id for c in collections] == list(COLLECTION_IDS)


def test_items_match_schema(all_items):
    assert len(all_items) == 800
    for item in all_items:
        assert item.kind in ("note", "link")
        assert item.content_status in ("note", "read", "not_read")
        assert item.lang in ("pt", "en")
        assert item.difficulty in ("easy", "hard")
        assert 0 <= len(item.labels) <= 2
        assert all(lab in COLLECTION_IDS for lab in item.labels)


def test_split_counts(splits):
    assert len(splits["test"]) == 360
    assert len(splits["train"]) + len(splits["validation"]) == 440
    assert 0.08 <= len(splits["validation"]) / 440 <= 0.13


def test_label_cardinality(splits):
    for name in ("train", "validation", "test"):
        counts = Counter(len(b.labels) for b in splits[name])
        assert counts.get(2, 0) > 0, f"{name} needs two-label items"
        assert all(len(b.labels) <= 2 for b in splits[name])


def test_groups_disjoint(splits):
    test_groups = {b.group_id for b in splits["test"]}
    other_groups = {b.group_id for b in splits["train"]} | {
        b.group_id for b in splits["validation"]
    }
    assert not (test_groups & other_groups)


def test_no_near_duplicates(splits):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    from crb.prompts import state_text

    test = [state_text(b) for b in splits["test"]]
    train = [state_text(b) for b in splits["train"] + splits["validation"]]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    vec.fit(test + train)
    m_test = vec.transform(test).toarray()
    m_train = vec.transform(train).toarray()
    worst = float(np.max(m_test @ m_train.T))
    assert worst < 0.8


def _train_file(splits):
    """The train.jsonl file = train + validation rows (plan counts the file)."""
    return splits["train"] + splits["validation"]


def test_hard_ratio(splits):
    for name, items in (("train", _train_file(splits)), ("test", splits["test"])):
        hard = sum(1 for b in items if b.difficulty == "hard")
        assert hard >= 0.25 * len(items), f"{name}"


def test_not_read_ratio(splits):
    for name, items in (("train", _train_file(splits)), ("test", splits["test"])):
        links = [b for b in items if b.kind == "link"]
        not_read = [b for b in links if b.content_status == "not_read"]
        assert len(not_read) >= 0.10 * len(links), f"{name}"


def test_kind_and_lang_ratio(splits):
    for name, items in (("train", _train_file(splits)), ("test", splits["test"])):
        notes = sum(1 for b in items if b.kind == "note")
        pt = sum(1 for b in items if b.lang == "pt")
        assert 0.40 <= notes / len(items) <= 0.60, f"{name}"
        assert 0.60 <= pt / len(items) <= 0.80, f"{name}"


def test_full_validator_passes():
    errors, stats = _run_validator()
    assert errors == []
    assert stats["test_total"] == 360
    assert stats["train_total"] + stats["validation_total"] == 440
