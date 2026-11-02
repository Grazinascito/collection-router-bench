"""Apply the human review from data/review_sample.csv.

ok=true keeps the labels; ok=false applies fixed_labels (semicolon-separated
ids, empty = uncollected). Re-runs validation and writes data/review_stats.json
with n_reviewed, n_ok, n_fixed. Rows with an empty ok are not counted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from crb.schema import COLLECTION_IDS, Block
from validate_data import validate

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRUE_WORDS = {"true", "1", "yes", "y", "sim", "s", "ok"}
FALSE_WORDS = {"false", "0", "no", "n", "nao", "não"}


def _parse_ok(value: str) -> bool | None:
    v = value.strip().lower()
    if v in TRUE_WORDS:
        return True
    if v in FALSE_WORDS:
        return False
    return None


def _parse_fixed(value: str) -> list[str]:
    v = value.strip()
    if not v:
        return []
    labels = [x.strip() for x in v.replace(",", ";").split(";") if x.strip()]
    unknown = [x for x in labels if x not in COLLECTION_IDS]
    if unknown:
        raise SystemExit(f"unknown label(s) in fixed_labels: {unknown}")
    if len(labels) > 2:
        raise SystemExit(f"at most 2 labels allowed: {labels}")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default=str(DATA_DIR / "review_sample.csv"))
    args = ap.parse_args()

    with open(args.csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {r["id"]: r for r in rows}

    n_reviewed = n_ok = n_fixed = 0
    for path_name in ("train.jsonl", "test.jsonl"):
        path = DATA_DIR / path_name
        blocks = [
            Block.model_validate(json.loads(x))
            for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        changed = False
        for i, block in enumerate(blocks):
            row = by_id.get(block.id)
            if row is None:
                continue
            ok = _parse_ok(row["ok"])
            if ok is None:
                continue
            n_reviewed += 1
            if ok:
                n_ok += 1
                continue
            n_fixed += 1
            fixed = _parse_fixed(row["fixed_labels"])
            updates = {"labels": fixed}
            if fixed and block.rationale:
                updates["rationale"] = (
                    f"{block.rationale} [human review: labels set to {', '.join(fixed)}]"
                )
            elif not fixed:
                updates["rationale"] = f"{block.rationale} [human review: moved to uncollected]"
            blocks[i] = block.model_copy(update=updates)
            changed = True
        if changed:
            path.write_text(
                "".join(json.dumps(b.model_dump(), ensure_ascii=False) + "\n" for b in blocks),
                encoding="utf-8",
            )

    errors, _stats = validate(DATA_DIR / "train.jsonl", DATA_DIR / "test.jsonl")
    if errors:
        print(f"FAIL after applying review: {len(errors)} problem(s)")
        for e in errors[:20]:
            print(f"  - {e}")
        raise SystemExit(1)

    (DATA_DIR / "review_stats.json").write_text(
        json.dumps({"n_reviewed": n_reviewed, "n_ok": n_ok, "n_fixed": n_fixed}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK: reviewed={n_reviewed} kept={n_ok} fixed={n_fixed}; data still valid")


if __name__ == "__main__":
    main()
