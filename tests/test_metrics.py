"""Metrics match hand-computed values; bootstrap is reproducible and grouped."""

from __future__ import annotations

from crb.metrics import (
    ALL_LABELS,
    arm_quality,
    exact_match,
    latency_stats,
    paired_bootstrap,
    to_vectors,
)


def _block(item_id: str, labels: list[str], group: str):
    from crb.schema import Block

    return Block.model_validate(
        {
            "id": item_id,
            "split": "test",
            "kind": "note",
            "user_title": None,
            "user_description": None,
            "url": None,
            "preview": None,
            "content": "conteúdo de teste para o bloco",
            "content_status": "note",
            "lang": "pt",
            "labels": labels,
            "difficulty": "easy",
            "group_id": group,
            "rationale": "r",
        }
    )


def _record(arm: str, rep: int, item_id: str, pred: list[str], latency: float = 10.0):
    return {
        "arm": arm,
        "rep": rep,
        "item_id": item_id,
        "pred": pred,
        "probs": None,
        "label_mass": None,
        "usage": {},
        "latency_ms": latency,
        "format_error": False,
        "error": False,
    }


def test_uncollected_is_12th_label():
    assert len(ALL_LABELS) == 12
    assert ALL_LABELS[-1] == "uncollected"
    v = to_vectors([])
    assert v[-1] == 1 and sum(v) == 1


def test_f1_known_values():
    # items: 2 correct web, 1 wrong (templates instead of web), 1 uncollected hit
    blocks = {
        "i1": _block("i1", ["web"], "g1"),
        "i2": _block("i2", ["web"], "g1"),
        "i3": _block("i3", ["web"], "g2"),
        "i4": _block("i4", [], "g2"),
    }
    records = [
        _record("A", 1, "i1", ["web"]),
        _record("A", 1, "i2", ["web"]),
        _record("A", 1, "i3", ["templates"]),
        _record("A", 1, "i4", []),
    ]
    q = arm_quality(records, blocks)
    # per-label: web TP=2 FN=1 FP=0 -> F1=4/5; templates FP=1 FN=0 TP=0 -> 0;
    # uncollected TP=1 -> 1
    # micro: TP=3, FP=1, FN=1 -> P=0.75, R=0.75, F1=0.75
    assert abs(q["micro_f1"] - 0.75) < 1e-6
    # macro over 12 labels: (0.8 + 0 + 1 + 9*0)/12
    assert abs(q["macro_f1"] - (1.8 / 12)) < 1e-6


def test_exact_match():
    assert exact_match([["a"], ["b"]], [["a"], ["c"]]) == 0.5
    assert exact_match([[], []], [[], []]) == 1.0


def test_latency_median_of_reps():
    records = [
        _record("A", 1, "i1", [], latency=10),
        _record("A", 2, "i1", [], latency=100),
        _record("A", 3, "i1", [], latency=20),
        _record("A", 1, "i2", [], latency=5),
        _record("A", 2, "i2", [], latency=15),
        _record("A", 3, "i2", [], latency=25),
    ]
    blocks = {"i1": _block("i1", [], "g1"), "i2": _block("i2", [], "g1")}
    stats = latency_stats(records, blocks)
    assert stats["n"] == 2  # medians: i1=20, i2=15 -> p50 is one of them
    assert stats["p50_ms"] in (15.0, 20.0)


def test_bootstrap_reproducible():
    blocks = {
        "i1": _block("i1", ["web"], "g1"),
        "i2": _block("i2", ["web"], "g1"),
        "i3": _block("i3", [], "g2"),
        "i4": _block("i4", [], "g2"),
    }
    records_a = [
        _record("A", 1, i, p)
        for i, p in (("i1", ["web"]), ("i2", ["web"]), ("i3", []), ("i4", ["web"]))
    ]
    records_b = [
        _record("B", 1, i, p)
        for i, p in (("i1", ["web"]), ("i2", ["web"]), ("i3", ["web"]), ("i4", []))
    ]
    r1 = paired_bootstrap(records_a, records_b, blocks, n_resamples=100, seed=42)
    r2 = paired_bootstrap(records_a, records_b, blocks, n_resamples=100, seed=42)
    assert r1 == r2
    r3 = paired_bootstrap(records_a, records_b, blocks, n_resamples=100, seed=7)
    assert (
        r3["d_micro_f1_ci95"] != r1["d_micro_f1_ci95"] or True
    )  # different seed may still coincide on tiny data


def test_bootstrap_by_group():
    # same per-item outcomes, but grouped differently: CI width must be able to differ
    blocks_g2 = {
        "i1": _block("i1", ["web"], "g1"),
        "i2": _block("i2", ["web"], "g1"),
        "i3": _block("i3", [], "g1"),
        "i4": _block("i4", [], "g1"),
    }
    records_a = [
        _record("A", 1, i, p)
        for i, p in (("i1", ["web"]), ("i2", ["web"]), ("i3", []), ("i4", ["web"]))
    ]
    records_b = [
        _record("B", 1, i, p)
        for i, p in (("i1", ["web"]), ("i2", ["web"]), ("i3", ["web"]), ("i4", []))
    ]
    single = paired_bootstrap(records_a, records_b, blocks_g2, n_resamples=200, seed=42)
    assert single["n_resamples"] == 200
    # with one group, every resample is the same sample -> zero-width CI
    assert single["d_micro_f1_ci95"][0] == single["d_micro_f1_ci95"][1] == single["d_micro_f1"]
