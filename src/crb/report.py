"""Build summary.json, report.md, facts.md, predictions.parquet and figures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from crb.data import load_split
from crb.metrics import (
    ALL_LABELS,
    arm_quality,
    latency_stats,
    paired_bootstrap,
    slices,
    stability,
    token_means,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def load_run(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    raw = run_dir / "raw"
    records_a = (
        [
            json.loads(x)
            for x in (raw / "A.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        if (raw / "A.jsonl").exists()
        else []
    )
    records_b = (
        [
            json.loads(x)
            for x in (raw / "B.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        if (raw / "B.jsonl").exists()
        else []
    )
    run_meta = {}
    if (run_dir / "run_meta.json").exists():
        run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    return records_a, records_b, run_meta


def compute(run_dir: Path, data_dir: Path = DATA_DIR) -> dict:
    records_a, records_b, run_meta = load_run(run_dir)
    splits = load_split(data_dir / "train.jsonl", data_dir / "test.jsonl")
    test_blocks = {b.id: b for b in splits["test"]}
    reps = max([r["rep"] for r in records_a + records_b], default=0)

    summary: dict = {
        "run_id": run_meta.get("run_id", run_dir.name),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_id": run_meta.get("model_id"),
        "model_sha": run_meta.get("model_sha"),
        "arm_b_impl": run_meta.get("arm_b_impl"),
        "server_id": run_meta.get("server_id"),
        "reps": reps,
        "n_test": len(test_blocks),
        "environment": run_meta.get("environment"),
        "arms": {},
    }

    for name, records in (("A", records_a), ("B", records_b)):
        if not records:
            continue
        summary["arms"][name] = {
            "quality": arm_quality(records, test_blocks, rep=1),
            "latency": latency_stats(records, test_blocks),
            "stability": stability(records, reps=max(2, min(reps, 3))),
            "tokens": token_means(records),
            "slices": slices(records, test_blocks, rep=1),
        }

    if records_a and records_b:
        summary["comparison"] = paired_bootstrap(records_a, records_b, test_blocks)

    if run_meta.get("throughput"):
        summary["throughput"] = run_meta["throughput"]

    classifier_path = run_dir / "classifier_metrics.json"
    if classifier_path.exists():
        summary["classifier"] = json.loads(classifier_path.read_text(encoding="utf-8"))

    return summary


def predictions_frame(run_dir: Path, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    records_a, records_b, _ = load_run(run_dir)
    splits = load_split(data_dir / "train.jsonl", data_dir / "test.jsonl")
    test_blocks = {b.id: b for b in splits["test"]}
    rows = []
    for records in (records_a, records_b):
        for r in records:
            block = test_blocks.get(r["item_id"])
            rows.append(
                {
                    "item_id": r["item_id"],
                    "arm": r["arm"],
                    "arm_b_impl": r.get("arm_b_impl"),
                    "rep": r["rep"],
                    "gold": ";".join(block.labels) if block else None,
                    "pred": ";".join(r["pred"]),
                    "correct": (block is not None and set(block.labels) == set(r["pred"])),
                    "latency_ms": r["latency_ms"],
                    "format_error": r.get("format_error", False),
                    "error": r.get("error", False),
                    "label_mass": r.get("label_mass"),
                    "probs": json.dumps(r.get("probs"), ensure_ascii=False)
                    if r.get("probs")
                    else None,
                    "difficulty": block.difficulty if block else None,
                    "kind": block.kind if block else None,
                    "lang": block.lang if block else None,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown outputs (every number comes from the summary dict)


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report_md(summary: dict) -> str:
    lines: list[str] = [f"# Benchmark report — {summary['run_id']}", ""]
    lines += [
        f"- model: `{summary.get('model_id')}` (sha `{summary.get('model_sha')}`)",
        f"- arm B implementation: `{summary.get('arm_b_impl')}`",
        f"- reps: {summary.get('reps')}, test items: {summary.get('n_test')}",
        f"- server_id: `{summary.get('server_id')}`",
        "",
        "## Quality and latency (rep 1)",
        "",
        "| arm | micro-F1 | macro-F1 | exact match | fmt err | err | label mass | p50/p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("A", "B"):
        arm = summary["arms"].get(name)
        if not arm:
            continue
        q, lat = arm["quality"], arm["latency"]
        lines.append(
            f"| {name} | {q['micro_f1']:.4f} | {q['macro_f1']:.4f} | {q['exact_match']:.4f} "
            f"| {_fmt_pct(q['format_error_rate'])} | {_fmt_pct(q['error_rate'])} "
            f"| {q['mean_label_mass'] if q['mean_label_mass'] is not None else '—'} "
            f"| {lat['p50_ms']:.1f} / {lat['p95_ms']:.1f} |"
        )
    lines += ["", "## Paired bootstrap (1,000 resamples by group, 95% CI)", ""]
    comp = summary.get("comparison")
    if comp:
        micro_ci = comp["d_micro_f1_ci95"]
        macro_ci = comp["d_macro_f1_ci95"]
        em_ci = comp["d_exact_match_ci95"]
        ratio_ci = comp["p50_ratio_a_over_b_ci95"]
        lines += [
            "| metric | value | CI 95% |",
            "| --- | --- | --- |",
            f"| Δ micro-F1 (B − A) | {comp['d_micro_f1']:+.4f} "
            f"| [{micro_ci[0]:+.4f}, {micro_ci[1]:+.4f}] |",
            f"| Δ macro-F1 (B − A) | {comp['d_macro_f1']:+.4f} "
            f"| [{macro_ci[0]:+.4f}, {macro_ci[1]:+.4f}] |",
            f"| Δ exact match (B − A) | {comp['d_exact_match']:+.4f} "
            f"| [{em_ci[0]:+.4f}, {em_ci[1]:+.4f}] |",
            f"| p50 ratio (A / B) | {comp['p50_ratio_a_over_b']:.3f} "
            f"| [{ratio_ci[0]:.3f}, {ratio_ci[1]:.3f}] |",
        ]
    lines += [
        "",
        "## Stability across reps",
        "",
        "| arm | same set | max P(yes) swing |",
        "| --- | --- | --- |",
    ]
    for name in ("A", "B"):
        arm = summary["arms"].get(name)
        if not arm:
            continue
        st = arm["stability"]
        swing = st.get("mean_max_p_yes_swing")
        same = _fmt_pct(st["same_set_rate"]) if st["same_set_rate"] is not None else "—"
        lines.append(f"| {name} | {same} | {swing if swing is not None else '—'} |")
    if summary.get("throughput"):
        lines += [
            "",
            "## Throughput",
            "",
            "| arm | conc. | decisions/hour | p50 ms | p95 ms |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in summary["throughput"]:
            lines.append(
                f"| {row['arm']} | {row['concurrency']} | {row['decisions_per_hour']:.0f} "
                f"| {row['p50_ms']:.1f} | {row['p95_ms']:.1f} |"
            )
    if summary.get("classifier"):
        clf = summary["classifier"]
        lines += [
            "",
            "## Arm C (classifier)",
            "",
            f"- micro-F1: {clf['micro_f1']:.4f}; macro-F1: {clf['macro_f1']:.4f}; "
            f"exact match: {clf['exact_match']:.4f}",
            "- latency (in-process, batch 1): "
            f"p50 {clf['latency']['p50_ms']:.1f} ms; p95 {clf['latency']['p95_ms']:.1f} ms",
        ]
    lines += [
        "",
        "## Per-collection micro-F1 (rep 1)",
        "",
        "| collection | A | B |",
        "| --- | --- | --- |",
    ]
    keys = [
        k for k in (summary["arms"].get("A", {}).get("slices", {})) if k.startswith("collection:")
    ]
    for key in keys:
        coll = key.split(":", 1)[1]
        a = summary["arms"].get("A", {}).get("slices", {}).get(key, {}).get("micro_f1", "—")
        b = summary["arms"].get("B", {}).get("slices", {}).get(key, {}).get("micro_f1", "—")
        lines.append(f"| {coll} | {a} | {b} |")
    return "\n".join(lines) + "\n"


def facts_md(summary: dict) -> str:
    comp = summary.get("comparison") or {}
    arms = summary.get("arms", {})
    a = arms.get("A", {})
    b = arms.get("B", {})
    lines = [
        f"# Facts — {summary['run_id']}",
        f"model: {summary.get('model_id')}",
        f"arm_b_impl: {summary.get('arm_b_impl')}",
        f"test items: {summary.get('n_test')}; reps: {summary.get('reps')}",
    ]
    if a:
        lines += [
            f"arm A micro-F1: {a['quality']['micro_f1']}",
            f"arm A macro-F1: {a['quality']['macro_f1']}",
            f"arm A exact match: {a['quality']['exact_match']}",
            f"arm A p50 latency: {a['latency']['p50_ms']} ms",
            f"arm A p95 latency: {a['latency']['p95_ms']} ms",
        ]
    if b:
        lines += [
            f"arm B micro-F1: {b['quality']['micro_f1']}",
            f"arm B macro-F1: {b['quality']['macro_f1']}",
            f"arm B exact match: {b['quality']['exact_match']}",
            f"arm B p50 latency: {b['latency']['p50_ms']} ms",
            f"arm B p95 latency: {b['latency']['p95_ms']} ms",
        ]
    if comp:
        lines += [
            f"delta micro-F1 (B minus A): {comp['d_micro_f1']} (95% CI {comp['d_micro_f1_ci95']})",
            f"delta macro-F1 (B minus A): {comp['d_macro_f1']} (95% CI {comp['d_macro_f1_ci95']})",
            "delta exact match (B minus A): "
            f"{comp['d_exact_match']} (CI {comp['d_exact_match_ci95']})",
            "p50 ratio (A over B): "
            f"{comp['p50_ratio_a_over_b']} (CI {comp['p50_ratio_a_over_b_ci95']})",
        ]
    return "\n".join(lines) + "\n"


def make_figures(summary: dict, run_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    # 1) main figure: p50 vs micro-F1
    fig, ax = plt.subplots(figsize=(6, 4))
    comp = summary.get("comparison") or {}
    for name, marker in (("A", "o"), ("B", "s")):
        arm = summary["arms"].get(name)
        if not arm:
            continue
        ax.scatter(
            arm["latency"]["p50_ms"],
            arm["quality"]["micro_f1"],
            marker=marker,
            s=90,
            label=f"arm {name}",
        )
        ax.annotate(
            f"arm {name}",
            (arm["latency"]["p50_ms"], arm["quality"]["micro_f1"]),
            textcoords="offset points",
            xytext=(8, -4),
        )
    if comp:
        ci = comp["d_micro_f1_ci95"]
        ax.set_title(
            f"p50 vs micro-F1 (Δmicro-F1 B−A {comp['d_micro_f1']:+.3f},"
            f" CI {ci[0]:+.3f}..{ci[1]:+.3f})",
            fontsize=9,
        )
    else:
        ax.set_title("p50 vs micro-F1", fontsize=10)
    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("micro-F1")
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "f1_vs_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # 2) latency box (per-item medians)
    records_a, records_b, _ = load_run(run_dir)
    import numpy as np

    data = []
    labels = []
    for name, records in (("A", records_a), ("B", records_b)):
        if records:
            per_item: dict[str, list[float]] = {}
            for r in records:
                if not r.get("error"):
                    per_item.setdefault(r["item_id"], []).append(r["latency_ms"])
            data.append([float(np.median(v)) for v in per_item.values()])
            labels.append(name)
    if data:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel("median latency per item (ms)")
        ax.set_title("Latency distribution")
        fig.tight_layout()
        path = fig_dir / "latency_box.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    # 3) per-collection F1
    slices_a = summary["arms"].get("A", {}).get("slices", {})
    slices_b = summary["arms"].get("B", {}).get("slices", {})
    keys = [k.split(":", 1)[1] for k in slices_a if k.startswith("collection:")]
    if keys:
        fig, ax = plt.subplots(figsize=(9, 4))
        x = range(len(keys))
        va = [slices_a.get(f"collection:{k}", {}).get("micro_f1", 0) for k in keys]
        vb = [slices_b.get(f"collection:{k}", {}).get("micro_f1", 0) for k in keys]
        ax.bar([i - 0.2 for i in x], va, width=0.4, label="A")
        ax.bar([i + 0.2 for i in x], vb, width=0.4, label="B")
        ax.set_xticks(list(x))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("micro-F1")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / "per_collection_f1.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    # 4) throughput
    if summary.get("throughput"):
        fig, ax = plt.subplots(figsize=(6, 4))
        arms = sorted({(row["arm"], row["concurrency"]) for row in summary["throughput"]})
        names = [f"{a}@{c}" for a, c in arms]
        values = [
            next(
                r["decisions_per_hour"]
                for r in summary["throughput"]
                if r["arm"] == a and r["concurrency"] == c
            )
            for a, c in arms
        ]
        ax.bar(names, values)
        ax.set_ylabel("decisions / GPU-hour")
        ax.set_title("Throughput under load")
        fig.tight_layout()
        path = fig_dir / "throughput.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    return made


def write_all(run_dir: Path, data_dir: Path = DATA_DIR) -> dict:
    summary = compute(run_dir, data_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(report_md(summary), encoding="utf-8")
    (run_dir / "facts.md").write_text(facts_md(summary), encoding="utf-8")
    frame = predictions_frame(run_dir, data_dir)
    frame.to_parquet(run_dir / "predictions.parquet", index=False)
    made = make_figures(summary, run_dir)
    summary["figures"] = [str(p.name) for p in made]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "ALL_LABELS",
    "load_run",
    "compute",
    "predictions_frame",
    "report_md",
    "facts_md",
    "write_all",
]
