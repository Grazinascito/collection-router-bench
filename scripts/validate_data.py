"""Validate data/test.jsonl and data/train.jsonl against the Block 1 rules.

Exit code 0 = all rules pass. --stats writes data/stats.json for the README
DATA block (used by scripts/update_readme.py).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from crb.data import load_collections, validation_group_ids
from crb.prompts import state_text
from crb.schema import COLLECTION_IDS, Block

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SIMILARITY_LIMIT = 0.8

SPLIT_TOTALS = {"test": 360, "train": 440}
SINGLE_PER_COLLECTION = {"test": 24, "train": 30}
MULTI_TOTAL = {"test": 48, "train": 55}
UNC_TOTAL = {"test": 48, "train": 55}


def _normalized(it: Block) -> str:
    preview_desc = it.preview.description if it.preview else ""
    return " ".join(filter(None, [it.user_title, preview_desc, it.content or ""])).lower()


def check(errors: list[str], ok: bool, msg: str) -> None:
    if not ok:
        errors.append(msg)


def validate(train_path: Path, test_path: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    collections = load_collections(DATA_DIR / "collections.yaml")
    ids = [c.id for c in collections]
    check(errors, ids == list(COLLECTION_IDS), f"collection order mismatch: {ids}")

    test = [Block.model_validate(json.loads(x)) for x in _read(test_path)]
    train = [Block.model_validate(json.loads(x)) for x in _read(train_path)]
    val = [it for it in train if it.split == "validation"]
    tr = [it for it in train if it.split == "train"]

    for split, items in (("test", test), ("train", train)):
        check(
            errors,
            len(items) == SPLIT_TOTALS[split],
            f"{split}: {len(items)} != {SPLIT_TOTALS[split]}",
        )

        singles = [it for it in items if len(it.labels) == 1]
        check(
            errors,
            len(singles) == 11 * SINGLE_PER_COLLECTION[split],
            f"{split} singles: {len(singles)}",
        )
        per_coll = Counter(it.labels[0] for it in singles)
        for cid in COLLECTION_IDS:
            check(
                errors,
                per_coll[cid] == SINGLE_PER_COLLECTION[split],
                f"{split} singles {cid}: {per_coll[cid]} != {SINGLE_PER_COLLECTION[split]}",
            )

        multi = [it for it in items if len(it.labels) == 2]
        check(
            errors,
            len(multi) == MULTI_TOTAL[split],
            f"{split} two-label: {len(multi)} != {MULTI_TOTAL[split]}",
        )
        coverage = Counter(lab for it in multi for lab in it.labels)
        minimum = 6 if split == "test" else 7
        for cid in COLLECTION_IDS:
            check(
                errors,
                coverage[cid] >= minimum,
                f"{split} two-label coverage {cid}: {coverage[cid]} < {minimum}",
            )

        uncollected = [it for it in items if not it.labels]
        check(
            errors, len(uncollected) == UNC_TOTAL[split], f"{split} uncollected: {len(uncollected)}"
        )

        hard = sum(1 for it in items if it.difficulty == "hard")
        check(errors, hard >= 0.25 * len(items), f"{split} hard ratio: {hard}/{len(items)} < 25%")

        notes = sum(1 for it in items if it.kind == "note")
        check(
            errors, 0.40 <= notes / len(items) <= 0.60, f"{split} notes ratio: {notes}/{len(items)}"
        )

        pt = sum(1 for it in items if it.lang == "pt")
        check(errors, 0.60 <= pt / len(items) <= 0.80, f"{split} pt ratio: {pt}/{len(items)}")

        links = [it for it in items if it.kind == "link"]
        not_read = sum(1 for it in links if it.content_status == "not_read")
        check(errors, not_read >= 0.10 * len(links), f"{split} not_read: {not_read}/{len(links)}")

        seen_norm: set[str] = set()
        seen_ids: set[str] = set()
        for it in items:
            check(errors, it.id not in seen_ids, f"{split} duplicate id {it.id}")
            seen_ids.add(it.id)
            norm = _normalized(it)
            check(errors, norm not in seen_norm, f"{split} exact duplicate: {it.id}")
            seen_norm.add(norm)
            valid = all(lab in COLLECTION_IDS for lab in it.labels)
            check(errors, valid, f"{it.id}: unknown label")
            check(errors, bool(it.rationale.strip()), f"{it.id}: empty rationale")
            if it.kind == "link":
                check(
                    errors, "synthetic" in (it.url or ""), f"{it.id}: url missing synthetic marker"
                )
            if it.content_status == "note":
                wc = len((it.content or "").split())
                check(errors, 40 <= wc <= 150, f"{it.id}: note words {wc}")
            if it.content_status == "read":
                wc = len((it.content or "").split())
                check(errors, 80 <= wc <= 250, f"{it.id}: link words {wc}")

    # groups disjoint across splits
    test_groups = {it.group_id for it in test}
    train_groups = {it.group_id for it in train}
    check(errors, not (test_groups & train_groups), "group_id appears in test and train")

    # validation = deterministic 10% of train groups
    expected_val = validation_group_ids(sorted(train_groups))
    actual_val = {it.group_id for it in val}
    check(
        errors, actual_val == expected_val, "validation split does not match seed-42 group sampling"
    )

    # TF-IDF char 3-5gram similarity between each test item and each train item
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    vec.fit([state_text(it) for it in test] + [state_text(it) for it in train])
    m_test = vec.transform([state_text(it) for it in test]).toarray()
    m_train = vec.transform([state_text(it) for it in train]).toarray()
    sims = np.max(m_test @ m_train.T, axis=1)
    worst = float(np.max(sims))
    check(
        errors,
        worst < SIMILARITY_LIMIT,
        f"test/train TF-IDF similarity {worst:.3f} >= {SIMILARITY_LIMIT}",
    )

    stats = {
        "test_total": len(test),
        "train_total": len(tr),
        "validation_total": len(val),
        "test_hard": sum(1 for it in test if it.difficulty == "hard"),
        "train_hard": sum(1 for it in train if it.difficulty == "hard"),
        "test_notes": sum(1 for it in test if it.kind == "note"),
        "test_pt": sum(1 for it in test if it.lang == "pt"),
        "test_not_read_links": sum(1 for it in test if it.content_status == "not_read"),
        "test_links": sum(1 for it in test if it.kind == "link"),
        "max_test_train_similarity": round(worst, 4),
        "test_two_label": sum(1 for it in test if len(it.labels) == 2),
        "test_uncollected": sum(1 for it in test if not it.labels),
    }
    return errors, stats


def _read(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    train_path = DATA_DIR / "train.jsonl"
    test_path = DATA_DIR / "test.jsonl"
    errors, stats = validate(train_path, test_path)
    if "--stats" in sys.argv:
        (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats, indent=2))
    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for e in errors[:30]:
            print(f"  - {e}")
        return 1
    print("OK: all Block 1 data rules pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
