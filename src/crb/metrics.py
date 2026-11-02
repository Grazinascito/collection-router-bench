"""Metrics: 12 labels (11 collections + uncollected), F1s, exact match, paired
bootstrap by group, latency percentiles, stability across reps, slices."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from crb.schema import COLLECTION_IDS, Block

UNCOLLECTED = "uncollected"
ALL_LABELS = list(COLLECTION_IDS) + [UNCOLLECTED]


def to_vectors(labels: list[str]) -> list[int]:
    """One-hot over the 12 labels; an empty set becomes uncollected=1."""
    effective = labels if labels else [UNCOLLECTED]
    return [1 if lab in effective else 0 for lab in ALL_LABELS]


def gold_of(blocks_by_id: dict[str, Block], item_id: str) -> list[str]:
    return list(blocks_by_id[item_id].labels)


def _f1(y_true: list[list[int]], y_pred: list[list[int]], average: str) -> float:
    if not y_true:
        return 0.0
    return float(f1_score(np.array(y_true), np.array(y_pred), average=average, zero_division=0))


def exact_match(truths: list[list[str]], preds: list[list[str]]) -> float:
    if not truths:
        return 0.0
    hits = sum(1 for t, p in zip(truths, preds, strict=True) if set(t) == set(p))
    return hits / len(truths)


def arm_quality(records: list[dict], blocks_by_id: dict[str, Block], rep: int = 1) -> dict:
    """F1s and error rates on one rep (records of a single arm)."""
    by_item = {r["item_id"]: r for r in records if r["rep"] == rep}
    truths: list[list[str]] = []
    preds: list[list[str]] = []
    for item_id, block in blocks_by_id.items():
        rec = by_item.get(item_id)
        if rec is None:
            continue
        truths.append(list(block.labels))
        preds.append(list(rec["pred"]))
    y_true = [to_vectors(t) for t in truths]
    y_pred = [to_vectors(p) for p in preds]
    return {
        "n": len(truths),
        "micro_f1": round(_f1(y_true, y_pred, "micro"), 4),
        "macro_f1": round(_f1(y_true, y_pred, "macro"), 4),
        "exact_match": round(exact_match(truths, preds), 4),
        "format_error_rate": _rate(records, "format_error"),
        "error_rate": _rate(records, "error"),
        "mean_label_mass": _mean_label_mass(records),
    }


def _rate(records: list[dict], field: str, rep: int = 1) -> float:
    rows = [r for r in records if r["rep"] == rep]
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r.get(field)) / len(rows), 4)


def _mean_label_mass(records: list[dict], rep: int = 1) -> float | None:
    values = [
        r["label_mass"] for r in records if r["rep"] == rep and r.get("label_mass") is not None
    ]
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def latency_stats(records: list[dict], blocks_by_id: dict[str, Block]) -> dict:
    """Per item the median latency across reps; p50/p95 across items."""
    by_item: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if not r.get("error"):
            by_item[r["item_id"]].append(r["latency_ms"])
    medians = [float(np.median(v)) for v in by_item.values()]
    if not medians:
        return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0}
    medians.sort()
    n = len(medians)
    return {
        "n": n,
        "p50_ms": round(medians[n // 2], 3),
        "p95_ms": round(medians[min(n - 1, int(n * 0.95))], 3),
    }


def stability(records: list[dict], reps: int = 3) -> dict:
    """Share of items with the same label set across all reps; for arm B also
    the mean largest P(yes) swing between reps."""
    by_item: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in records:
        by_item[r["item_id"]][r["rep"]] = r
    full = [d for d in by_item.values() if len(d) >= reps]
    if not full:
        return {"same_set_rate": None, "mean_max_p_yes_swing": None}
    same = sum(
        1 for d in full if len({tuple(sorted(d[r]["pred"])) for r in range(1, reps + 1)}) == 1
    )
    swings = []
    for d in full:
        per_rep = [d[r].get("probs") for r in range(1, reps + 1)]
        if any(p is None for p in per_rep):
            continue
        keys = per_rep[0].keys()
        for k in keys:
            values = [p[k] for p in per_rep]
            swings.append(max(values) - min(values))
    return {
        "same_set_rate": round(same / len(full), 4),
        "mean_max_p_yes_swing": round(float(np.mean(swings)), 4) if swings else None,
    }


def token_means(records: list[dict]) -> dict:
    prompt_keys = ("prompt_tokens", "input_tokens")
    completion_keys = ("completion_tokens", "output_tokens")

    def _mean(keys: tuple[str, ...]) -> float | None:
        values = []
        for r in records:
            usage = r.get("usage") or {}
            for k in keys:
                if k in usage:
                    values.append(usage[k])
                    break
        return round(float(np.mean(values)), 1) if values else None

    return {
        "prompt_tokens_mean": _mean(prompt_keys),
        "completion_tokens_mean": _mean(completion_keys),
    }


def slices(records: list[dict], blocks_by_id: dict[str, Block], rep: int = 1) -> dict:
    """F1 by collection, difficulty, kind, content_status and lang."""
    by_item = {r["item_id"]: r for r in records if r["rep"] == rep}
    groups: dict[str, list[tuple[Block, dict]]] = defaultdict(list)
    for item_id, rec in by_item.items():
        block = blocks_by_id[item_id]
        for lab in block.labels:
            groups[f"collection:{lab}"].append((block, rec))
        groups[f"difficulty:{block.difficulty}"].append((block, rec))
        groups[f"kind:{block.kind}"].append((block, rec))
        groups[f"content_status:{block.content_status}"].append((block, rec))
        groups[f"lang:{block.lang}"].append((block, rec))
        if not block.labels:
            groups["collection:uncollected"].append((block, rec))
    out = {}
    for key, pairs in sorted(groups.items()):
        y_true = [to_vectors(list(b.labels)) for b, _ in pairs]
        y_pred = [to_vectors(list(r["pred"])) for _, r in pairs]
        out[key] = {
            "n": len(pairs),
            "micro_f1": round(_f1(y_true, y_pred, "micro"), 4),
            "macro_f1": round(_f1(y_true, y_pred, "macro"), 4),
        }
    return out


def paired_bootstrap(
    records_a: list[dict],
    records_b: list[dict],
    blocks_by_id: dict[str, Block],
    n_resamples: int = 1000,
    seed: int = 42,
    rep: int = 1,
) -> dict:
    """Paired bootstrap by group_id: resample groups, recompute the difference.

    Reports 95% CI for d_micro_f1 (B-A), d_macro_f1, d_exact_match and the
    p50 ratio (A/B).
    """
    a_by_item = {r["item_id"]: r for r in records_a if r["rep"] == rep}
    b_by_item = {r["item_id"]: r for r in records_b if r["rep"] == rep}
    item_ids = [i for i in blocks_by_id if i in a_by_item and i in b_by_item]
    groups: dict[str, list[str]] = defaultdict(list)
    for item_id in item_ids:
        groups[blocks_by_id[item_id].group_id].append(item_id)
    group_names = sorted(groups)

    rng = random.Random(seed)

    def _metrics(sample_item_ids: list[str]) -> tuple[float, float, float, float]:
        truths = [list(blocks_by_id[i].labels) for i in sample_item_ids]
        preds_a = [list(a_by_item[i]["pred"]) for i in sample_item_ids]
        preds_b = [list(b_by_item[i]["pred"]) for i in sample_item_ids]
        y_t = [to_vectors(t) for t in truths]
        micro_d = _f1(y_t, [to_vectors(p) for p in preds_b], "micro") - _f1(
            y_t, [to_vectors(p) for p in preds_a], "micro"
        )
        macro_d = _f1(y_t, [to_vectors(p) for p in preds_b], "macro") - _f1(
            y_t, [to_vectors(p) for p in preds_a], "macro"
        )
        em_d = exact_match(truths, preds_b) - exact_match(truths, preds_a)
        lat_a = float(np.median([a_by_item[i]["latency_ms"] for i in sample_item_ids]))
        lat_b = float(np.median([b_by_item[i]["latency_ms"] for i in sample_item_ids]))
        ratio = lat_a / lat_b if lat_b else 0.0
        return micro_d, macro_d, em_d, ratio

    base = _metrics(item_ids)
    diffs = {"micro": [], "macro": [], "em": [], "ratio": []}
    for _ in range(n_resamples):
        chosen = [rng.choice(group_names) for _ in group_names]
        sample = [i for g in chosen for i in groups[g]]
        m, ma, em, ra = _metrics(sample)
        diffs["micro"].append(m)
        diffs["macro"].append(ma)
        diffs["em"].append(em)
        diffs["ratio"].append(ra)

    def _ci(values: list[float]) -> tuple[float, float]:
        ordered = sorted(values)
        n = len(ordered)
        return round(ordered[int(0.025 * n)], 4), round(ordered[min(n - 1, int(0.975 * n))], 4)

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "d_micro_f1": round(base[0], 4),
        "d_micro_f1_ci95": _ci(diffs["micro"]),
        "d_macro_f1": round(base[1], 4),
        "d_macro_f1_ci95": _ci(diffs["macro"]),
        "d_exact_match": round(base[2], 4),
        "d_exact_match_ci95": _ci(diffs["em"]),
        "p50_ratio_a_over_b": round(base[3], 4),
        "p50_ratio_a_over_b_ci95": _ci(diffs["ratio"]),
    }
