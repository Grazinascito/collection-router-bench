"""Deterministic generator for data/test.jsonl and data/train.jsonl.

Quotas (from the plan, Block 1):
- test:  360 = 264 single-label (24 per collection) + 48 two-label + 48 uncollected
- train: 440 = 330 single-label (30 per collection) + 55 two-label + 55 uncollected
- 10% of train groups become split=validation (seed 42, via crb.data)
- hard (boundary + multi-label) >= 25%, notes 40-60%, pt 60-80%, links not_read >= 10%

Seed banks live in banks_a/b/c/d.py; see docs/decisions.md for why a committed
seeded generator replaces one-off handwritten JSONL.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from banks_a import CLOSERS, EASY_A
from banks_b import EASY_B
from banks_c import BOUNDARY
from banks_d import MULTI, UNC
from crb.data import validation_group_ids
from crb.schema import COLLECTION_IDS

EASY = {**EASY_A, **EASY_B}
SEED = 42
CONTENT_NOTE_WORDS = (40, 150)
CONTENT_LINK_WORDS = (80, 250)

KIND_PATTERNS = {
    "easy": ["link", "link", "note", "note"],
    "boundary": ["link", "link", "note"],
    "multi3": ["link", "link", "note"],
    "multi2": ["link", "note"],
    "unc4": ["link", "note", "note", "note"],
    "unc3": ["link", "note", "note"],
}


def _words(text: str) -> int:
    return len(text.split())


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return "-".join(slug.split("-")[:6]) or "item"


class Generator:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.items: list[dict] = []
        self.group_counter = 0
        self.link_counter = 0
        self.not_read_counter = 0
        self.url_seq = 0
        self.closer_counter = 0
        self.split = "test"

    # ------------------------------------------------------------------
    def _is_not_read(self) -> bool:
        """Mark every 9th link as not_read (~11% of links, above the 10% floor)."""
        self.link_counter += 1
        if self.link_counter % 9 == 0:
            self.not_read_counter += 1
            return True
        return False

    def _next_group(self) -> str:
        self.group_counter += 1
        return f"g-{self.split}{self.group_counter:04d}"

    def _next_url(self, seed: dict, title: str) -> str:
        self.url_seq += 1
        return f"https://{seed['dom']}/{_slugify(title)}-synthetic-{self.url_seq:04d}"

    def _pick_body(self, seed: dict, kind: str, used: set[str]) -> list[str]:
        sentences = list(seed["b"])
        if kind == "link":
            return sentences
        sizes = [s for s in (5, 6) if s <= len(sentences)] or [len(sentences)]
        for _ in range(80):
            size = self.rng.choice(sizes)
            start = self.rng.randrange(0, len(sentences) - size + 1)
            text = " ".join(sentences[start : start + size])
            if not (CONTENT_NOTE_WORDS[0] <= _words(text) <= CONTENT_NOTE_WORDS[1]):
                continue
            if text not in used:
                return sentences[start : start + size]
        # deterministic fallback: any fitting, unused window in order
        for size in range(4, len(sentences) + 1):
            for start in range(0, len(sentences) - size + 1):
                text = " ".join(sentences[start : start + size])
                if (
                    CONTENT_NOTE_WORDS[0] <= _words(text) <= CONTENT_NOTE_WORDS[1]
                    and text not in used
                ):
                    return sentences[start : start + size]
        return sentences[-6:]

    def _link_content(self, seed: dict, occ: int) -> str:
        """Full body plus split-pool closing sentences until the 80-word floor.

        Links beyond the first of a seed always get one extra closer so two
        links from the same seed never produce identical content.
        """
        parts = list(seed["b"])
        pool = CLOSERS[self.split][seed["l"]]
        if occ >= 1:
            parts.append(pool[(self.closer_counter + occ) % len(pool)])
            self.closer_counter += 1
        words = _words(" ".join(parts))
        for _ in range(3):
            if words >= CONTENT_LINK_WORDS[0]:
                break
            closer = pool[self.closer_counter % len(pool)]
            self.closer_counter += 1
            parts.append(closer)
            words = _words(" ".join(parts))
        return " ".join(parts)

    def make_item(
        self,
        seed: dict,
        labels: list[str],
        difficulty: str,
        kind: str,
        occ: int = 0,
        used: set[str] | None = None,
    ) -> dict:
        titles = list(seed["t"])
        if kind == "link":
            title = titles[occ % len(titles)]
            preview_title = titles[(occ + 1) % len(titles)]
            not_read = self._is_not_read()
            content = None if not_read else self._link_content(seed, occ)
            if content is not None:
                wc = _words(content)
                if not (CONTENT_LINK_WORDS[0] <= wc <= CONTENT_LINK_WORDS[1]):
                    raise AssertionError(
                        f"seed body out of link word range ({wc}): {seed['t'][0][:50]}"
                    )
            return {
                "kind": "link",
                "user_title": title,
                "user_description": None,
                "url": self._next_url(seed, title),
                "preview": {
                    "site_name": seed["site"],
                    "title": preview_title,
                    "description": seed["d"],
                },
                "content": content,
                "content_status": "not_read" if not_read else "read",
                "lang": seed["l"],
                "labels": labels,
                "difficulty": difficulty,
                "rationale": seed["why"],
            }
        # note
        used = used if used is not None else set()
        body = self._pick_body(seed, "note", used)
        content = " ".join(body)
        wc = _words(content)
        if not (CONTENT_NOTE_WORDS[0] <= wc <= CONTENT_NOTE_WORDS[1]):
            raise AssertionError(f"note body out of word range ({wc}): {seed['t'][0][:50]}")
        return {
            "kind": "note",
            "user_title": titles[occ % len(titles)],
            "user_description": None,
            "url": None,
            "preview": None,
            "content": content,
            "content_status": "note",
            "lang": seed["l"],
            "labels": labels,
            "difficulty": difficulty,
            "rationale": seed["why"],
        }

    def consume_seed(
        self, seed: dict, labels: list[str], difficulty: str, n: int, pattern: list[str]
    ) -> None:
        group = self._next_group()
        used: set[str] = set()
        for i in range(n):
            kind = pattern[i] if i < len(pattern) else "note"
            item = self.make_item(seed, labels, difficulty, kind, occ=i, used=used)
            used.add(item["content"] or "")
            item["group_id"] = group
            self.items.append(item)


def _take(
    seeds: list[dict], needed: int, pattern: list[str], gen: Generator, labels_of, difficulty: str
) -> int:
    """Consume seeds in bank order until `needed` items exist; returns count added."""
    added = 0
    for seed in seeds:
        if added >= needed:
            break
        n = min(seed.get("y", 4), needed - added)
        gen.consume_seed(seed, labels_of(seed), difficulty, n, pattern)
        added += n
    return added


def build_split(split: str) -> list[dict]:
    gen = Generator()
    gen.split = split
    easy_target = 18 if split == "test" else 24
    hard_target = 6
    multi_total = 48 if split == "test" else 55
    unc_total = 48 if split == "test" else 55

    # 1) easy singles per collection
    for coll in COLLECTION_IDS:
        seeds = EASY[coll][split]
        added = _take(seeds, easy_target, KIND_PATTERNS["easy"], gen, lambda s, c=coll: [c], "easy")
        assert added == easy_target, f"{split} easy {coll}: {added} != {easy_target}"

    # 2) hard boundary singles: 6 per collection from seeds leaning to it
    for coll in COLLECTION_IDS:
        leaning = [s for s in BOUNDARY[split] if s["lean"] == coll]
        added = _take(
            leaning, hard_target, KIND_PATTERNS["boundary"], gen, lambda s: [s["lean"]], "hard"
        )
        assert added == hard_target, f"{split} hard {coll}: {added} != {hard_target}"

    # 3) multi-label
    added = 0
    for seed in MULTI[split]:
        if added >= multi_total:
            break
        n = min(seed["y"], multi_total - added)
        pattern = KIND_PATTERNS["multi3"] if seed["y"] >= 3 else KIND_PATTERNS["multi2"]
        gen.consume_seed(seed, list(seed["colls"]), "hard", n, pattern)
        added += n
    assert added == multi_total, f"{split} multi: {added} != {multi_total}"

    # 4) uncollected
    added = 0
    for seed in UNC[split]:
        if added >= unc_total:
            break
        n = min(seed["y"], unc_total - added)
        pattern = KIND_PATTERNS["unc4"] if seed["y"] >= 4 else KIND_PATTERNS["unc3"]
        gen.consume_seed(seed, [], "easy", n, pattern)
        added += n
    assert added == unc_total, f"{split} uncollected: {added} != {unc_total}"

    items = gen.items
    prefix = split
    for i, item in enumerate(items, start=1):
        item["id"] = f"{prefix}-{i:04d}"
        item["split"] = split

    if split == "train":
        groups = sorted({it["group_id"] for it in items})
        val_groups = validation_group_ids(groups, SEED)
        for it in items:
            if it["group_id"] in val_groups:
                it["split"] = "validation"
    return items


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data"
    test_items = build_split("test")
    train_items = build_split("train")

    (out_dir / "test.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in test_items),
        encoding="utf-8",
    )
    (out_dir / "train.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in train_items),
        encoding="utf-8",
    )

    def stats(items: list[dict]) -> str:
        n = len(items)
        notes = sum(1 for it in items if it["kind"] == "note")
        pt = sum(1 for it in items if it["lang"] == "pt")
        hard = sum(1 for it in items if it["difficulty"] == "hard")
        links = n - notes
        not_read = sum(1 for it in items if it["content_status"] == "not_read")
        multi = sum(1 for it in items if len(it["labels"]) == 2)
        zero = sum(1 for it in items if not it["labels"])
        return (
            f"n={n} notes={notes} ({notes / n:.0%}) pt={pt} ({pt / n:.0%}) "
            f"hard={hard} ({hard / n:.0%}) links={links} not_read={not_read} "
            f"({not_read / max(links, 1):.0%} of links) two-label={multi} zero-label={zero}"
        )

    print("test :", stats(test_items))
    print("train:", stats(train_items))
    print(
        "validation:",
        sum(1 for it in train_items if it["split"] == "validation"),
        "items in",
        len({it["group_id"] for it in train_items if it["split"] == "validation"}),
        "groups",
    )


if __name__ == "__main__":
    main()
