"""Gradio app for the collection-router-bench Space (CPU, no secrets).

Tabs: Results (table + figures), Explore (per-item predictions of each arm),
Try (arm C model), About (method + links). Files are colocated in the Space
repo by scripts/publish_hf.py: results/latest/*, data/*, runtime.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).parent
RUNTIME = (
    json.loads((HERE / "runtime.json").read_text()) if (HERE / "runtime.json").exists() else {}
)
MODEL_ID = RUNTIME.get("model_id", "collection-router-e5-small")


def _load_summary() -> dict | None:
    path = HERE / "results" / "latest" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def _load_predictions() -> pd.DataFrame:
    path = HERE / "results" / "latest" / "predictions.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_test() -> pd.DataFrame:
    path = HERE / "data" / "test.jsonl"
    rows = (
        [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if path.exists()
        else []
    )
    return pd.DataFrame(rows)


def results_table(summary: dict | None) -> str:
    if not summary:
        return "No results yet — run the notebook first."
    lines = [
        "| arm | micro-F1 | macro-F1 | exact match | p50 ms | p95 ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("A", "B", "C"):
        if name == "C":
            clf = summary.get("classifier")
            if clf:
                lines.append(
                    f"| C | {clf['micro_f1']:.4f} | {clf['macro_f1']:.4f} "
                    f"| {clf['exact_match']:.4f} | {clf['latency']['p50_ms']:.1f} "
                    f"| {clf['latency']['p95_ms']:.1f} |"
                )
            continue
        arm = summary.get("arms", {}).get(name)
        if not arm:
            continue
        q, lat = arm["quality"], arm["latency"]
        lines.append(
            f"| {name} | {q['micro_f1']:.4f} | {q['macro_f1']:.4f} | {q['exact_match']:.4f} "
            f"| {lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} |"
        )
    comp = summary.get("comparison")
    if comp:
        ci = comp["d_micro_f1_ci95"]
        ratio_ci = comp["p50_ratio_a_over_b_ci95"]
        lines += [
            "",
            f"Δ micro-F1 (B − A): **{comp['d_micro_f1']:+.4f}** "
            f"(95% CI {ci[0]:+.4f} .. {ci[1]:+.4f})",
            f"p50 ratio (A / B): **{comp['p50_ratio_a_over_b']:.2f}** "
            f"(95% CI {ratio_ci[0]:.2f} .. {ratio_ci[1]:.2f})",
        ]
    return "\n".join(lines)


def explore(collection: str, difficulty: str, arm: str):
    preds = _load_predictions()
    if preds.empty:
        return "No predictions yet.", ""
    blocks = _load_test()
    if blocks.empty:
        return "No test data in the Space.", ""
    rep1 = preds[preds["rep"] == 1]
    rows = rep1.merge(
        blocks[["id", "user_title", "content_status", "lang"]], left_on="item_id", right_on="id"
    )
    if collection != "all":
        rows = rows[
            rows["gold"].fillna("").str.contains(collection)
            | rows["pred"].fillna("").str.contains(collection)
        ]
    if difficulty != "all":
        rows = rows[rows["difficulty"] == difficulty]
    if arm != "all":
        rows = rows[rows["arm"] == arm]
    if rows.empty:
        return "No matching items.", ""
    sample = rows.iloc[0]
    probs = (
        json.loads(sample["probs"]) if isinstance(sample.get("probs"), str) else sample.get("probs")
    )
    probs_text = ""
    if probs:
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
        probs_text = "\n".join(f"- {k}: {v:.3f}" for k, v in ranked)
    block = blocks[blocks["id"] == sample["item_id"]].iloc[0]
    content = (block.get("content") or "")[:400]
    text = (
        f"**{block.get('user_title') or '(no title)'}**\n\n"
        f"gold: `{sample['gold'] or 'uncollected'}` | pred: `{sample['pred'] or 'uncollected'}` "
        f"| correct: {sample['correct']} | arm {sample['arm']} "
        f"| {sample['difficulty']} | {block.get('lang')}\n\n"
        f"{content}"
    )
    return text, probs_text


def try_classify(text: str):
    if not text.strip():
        return "Paste a block of text (a note or an article excerpt)."
    try:
        from transformers import pipeline

        pipe = pipeline("text-classification", model=MODEL_ID, top_k=None)
        results = pipe(f"query: {text}"[:2000])[0]
        ranked = sorted(results, key=lambda r: -r["score"])
        keep = [r for r in ranked if r["score"] >= 0.5]
        if not keep:
            return "Uncollected (no label reached 0.5)\n\n" + "\n".join(
                f"- {r['label']}: {r['score']:.3f}" for r in ranked[:4]
            )
        return "\n".join(f"- {r['label']}: {r['score']:.3f}" for r in keep)
    except Exception as exc:  # noqa: BLE001
        return f"Model not available in this Space yet: {exc}"


ABOUT = """## About

This Space shows the results of **collection-router-bench**: a benchmark that
compares three ways to route a saved block into 0 or more of 11 collections.

- **Arm A — generation**: `/v1/chat/completions` with a strict JSON schema.
- **Arm B — typed decision**: `/v1/systemone` `noul` questions, reading P(yes)
  from label logprobs (no tokens generated).
- **Arm C — classifier**: fine-tuned `multilingual-e5-small`.

Same model and server for A and B; only the decision mechanism differs.
Code: [GitHub](https://github.com/Grazinascito/collection-router-bench) ·
Data: [dataset](https://huggingface.co/datasets/HF_USER/collection-router-bench)
"""


def build_demo() -> gr.Blocks:
    summary = _load_summary()
    with gr.Blocks(title="collection-router-bench") as demo:
        with gr.Tab("Results"):
            gr.Markdown(results_table(summary))
            fig_dir = HERE / "results" / "latest" / "figures"
            for name, caption in (
                ("f1_vs_latency.png", "micro-F1 vs p50 latency"),
                ("per_collection_f1.png", "micro-F1 per collection"),
                ("latency_box.png", "latency distribution"),
                ("throughput.png", "throughput under load"),
            ):
                path = fig_dir / name
                if path.exists():
                    gr.Image(value=str(path), caption=caption)
        with gr.Tab("Explore"):
            with gr.Row():
                collection = gr.Dropdown(
                    [
                        "all",
                        "cognitive_psychology",
                        "dev_engineering",
                        "design_ui",
                        "ui_inspiration",
                        "app_patterns",
                        "app_ideas",
                        "templates",
                        "web",
                        "wellbeing",
                        "career_work",
                        "learning_study",
                    ],
                    value="all",
                    label="collection (gold or pred)",
                )
                difficulty = gr.Dropdown(["all", "easy", "hard"], value="all", label="difficulty")
                arm = gr.Dropdown(["all", "A", "B"], value="all", label="arm")
            out_text = gr.Markdown("Press Apply.")
            out_probs = gr.Markdown("")
            btn = gr.Button("Apply")
            btn.click(explore, inputs=[collection, difficulty, arm], outputs=[out_text, out_probs])
        with gr.Tab("Try"):
            gr.Markdown(
                "Arm C on CPU. Paste text; the classifier returns labels with sigmoid >= 0.5."
            )
            inp = gr.Textbox(
                lines=8, label="Block text", placeholder="Paste a note or article excerpt..."
            )
            out = gr.Markdown()
            btn = gr.Button("Classify")
            btn.click(try_classify, inputs=inp, outputs=out)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build_demo().launch()
