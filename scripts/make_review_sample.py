"""Generate data/review_sample.csv for the human review (Grazi, ~10 min).

Stratified by collection (every collection appears), at least --min-hard hard
items, deterministic under --seed. Columns: id,state_text,labels,rationale,ok,
fixed_labels,note. ok/fixed_labels/note start empty for the reviewer to fill.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from crb.data import load_collections, load_split
from crb.prompts import state_text

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_sample(items, n: int, min_hard: int, seed: int) -> list:
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)

    chosen: list = []
    chosen_ids: set[str] = set()

    def covered(coll: str) -> int:
        return sum(1 for it in chosen if coll in it.labels)

    # pass 1: one item per collection (prefer hard for variety)
    for coll in [c.id for c in load_collections(DATA_DIR / "collections.yaml")]:
        cand = next(
            (
                it
                for it in pool
                if it.id not in chosen_ids and coll in it.labels and it.difficulty == "hard"
            ),
            None,
        )
        if cand is None:
            cand = next((it for it in pool if it.id not in chosen_ids and coll in it.labels), None)
        if cand is not None:
            chosen.append(cand)
            chosen_ids.add(cand.id)

    # pass 2: top up hard quota
    hard = sum(1 for it in chosen if it.difficulty == "hard")
    for it in pool:
        if hard >= min_hard or len(chosen) >= n:
            break
        if it.id not in chosen_ids and it.difficulty == "hard":
            chosen.append(it)
            chosen_ids.add(it.id)
            hard += 1

    # pass 3: fill to n, balancing collection coverage (least covered first)
    while len(chosen) < n:
        counts = {c.id: covered(c.id) for c in load_collections(DATA_DIR / "collections.yaml")}
        least = min(counts.values())
        prefer = [cid for cid, k in counts.items() if k == least]
        cand = next(
            (it for it in pool if it.id not in chosen_ids and any(c in prefer for c in it.labels)),
            None,
        )
        if cand is None:
            cand = next((it for it in pool if it.id not in chosen_ids), None)
        if cand is None:
            break
        chosen.append(cand)
        chosen_ids.add(cand.id)

    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-hard", type=int, default=15)
    args = ap.parse_args()

    splits = load_split(DATA_DIR / "train.jsonl", DATA_DIR / "test.jsonl")
    items = splits["train"] + splits["validation"] + splits["test"]
    chosen = build_sample(items, args.n, args.min_hard, args.seed)

    out = DATA_DIR / "review_sample.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "state_text", "labels", "rationale", "ok", "fixed_labels", "note"])
        for it in chosen:
            writer.writerow(
                [
                    it.id,
                    state_text(it),
                    ";".join(it.labels),
                    it.rationale,
                    "",
                    "",
                    "",
                ]
            )
    hard = sum(1 for it in chosen if it.difficulty == "hard")
    print(f"wrote {out}: {len(chosen)} items, {hard} hard")


if __name__ == "__main__":
    main()
