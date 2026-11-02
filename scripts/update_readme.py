"""Rewrite the README RESULTS and DATA blocks (numbers come only from
summary.json / stats.json). Also replaces the HF_USER placeholder with the
real user when a token is available. Idempotent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_summary(run_id: str | None) -> dict | None:
    local = REPO / "results" / (run_id or "latest") / "summary.json"
    if not local.exists() and run_id is None:
        candidates = sorted((REPO / "results").glob("*/summary.json"))
        local = candidates[-1] if candidates else None
    if local and local.exists():
        return json.loads(local.read_text())
    try:  # fall back to the published dataset
        from huggingface_hub import HfApi, hf_hub_download

        user = HfApi().whoami()["name"]
        path = hf_hub_download(
            f"{user}/collection-router-bench",
            "results/latest/summary.json",
            repo_type="dataset",
        )
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001 - offline / not published yet
        return None


def results_block(summary: dict) -> str:
    lines = [
        f"Run `{summary.get('run_id')}` — model `{summary.get('model_id')}` "
        f"(sha `{str(summary.get('model_sha'))[:8]}`), arm B impl `{summary.get('arm_b_impl')}`, "
        f"{summary.get('reps')} reps, {summary.get('n_test')} test items.",
        "",
        "| arm | micro-F1 | macro-F1 | exact match | p50 ms | p95 ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("A", "B"):
        arm = summary.get("arms", {}).get(name)
        if not arm:
            continue
        q, lat = arm["quality"], arm["latency"]
        lines.append(
            f"| {name} | {q['micro_f1']:.4f} | {q['macro_f1']:.4f} | {q['exact_match']:.4f} "
            f"| {lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} |"
        )
    clf = summary.get("classifier")
    if clf:
        lines.append(
            f"| C | {clf['micro_f1']:.4f} | {clf['macro_f1']:.4f} | {clf['exact_match']:.4f} "
            f"| {clf['latency']['p50_ms']:.1f} | {clf['latency']['p95_ms']:.1f} |"
        )
    comp = summary.get("comparison")
    if comp:
        lines += [
            "",
            f"Δ micro-F1 (B − A) **{comp['d_micro_f1']:+.4f}** "
            f"(95% CI {comp['d_micro_f1_ci95'][0]:+.4f}..{comp['d_micro_f1_ci95'][1]:+.4f}, "
            "paired bootstrap by group, 1,000 resamples)",
            f"Δ macro-F1 (B − A) **{comp['d_macro_f1']:+.4f}** (95% CI "
            f"{comp['d_macro_f1_ci95'][0]:+.4f}..{comp['d_macro_f1_ci95'][1]:+.4f})",
            f"Δ exact match (B − A) **{comp['d_exact_match']:+.4f}** (95% CI "
            f"{comp['d_exact_match_ci95'][0]:+.4f}..{comp['d_exact_match_ci95'][1]:+.4f})",
            f"p50 ratio (A / B) **{comp['p50_ratio_a_over_b']:.2f}** (95% CI "
            f"{comp['p50_ratio_a_over_b_ci95'][0]:.2f}..{comp['p50_ratio_a_over_b_ci95'][1]:.2f})",
        ]
    st = summary.get("arms", {}).get("B", {}).get("stability", {})
    if st.get("same_set_rate") is not None:
        lines += [
            "",
            f"Stability across reps: {st['same_set_rate']:.1%} of items keep the same set "
            f"(arm B mean max P(yes) swing {st.get('mean_max_p_yes_swing')})",
        ]
    return "\n".join(lines)


def data_block(stats: dict) -> str:
    hard_pct = stats["test_hard"] / stats["test_total"]
    notes_pct = stats["test_notes"] / stats["test_total"]
    pt_pct = stats["test_pt"] / stats["test_total"]
    return "\n".join(
        [
            f"- {stats['test_total']} test / {stats['train_total']} train / "
            f"{stats['validation_total']} validation items",
            f"- {stats['test_hard']} hard test items ({hard_pct:.0%}); "
            f"{stats['test_two_label']} two-label; {stats['test_uncollected']} uncollected",
            f"- {stats['test_notes']} notes ({notes_pct:.0%}); "
            f"{stats['test_pt']} pt ({pt_pct:.0%}); "
            f"{stats['test_not_read_links']}/{stats['test_links']} links not_read",
            "- max test/train char-3-5gram TF-IDF similarity: "
            f"{stats['max_test_train_similarity']}",
        ]
    )


def replace_between(text: str, start_marker: str, end_marker: str, block: str) -> str:
    pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
    return pattern.sub(start_marker + "\n" + block + "\n" + end_marker, text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--readme", default=str(REPO / "README.md"))
    ap.add_argument(
        "--hf-user", default=None, help="replace the HF_USER placeholder (no local token)"
    )
    args = ap.parse_args()

    readme_path = Path(args.readme)
    text = readme_path.read_text(encoding="utf-8")

    user = args.hf_user
    if not user:
        try:
            from huggingface_hub import HfApi

            user = HfApi().whoami()["name"]
        except Exception:  # noqa: BLE001 - token may be absent on the Mac
            user = None

    summary = load_summary(args.run_id)
    if summary:
        text = replace_between(
            text, "<!-- RESULTS:START -->", "<!-- RESULTS:END -->", results_block(summary)
        )
    stats_path = REPO / "data" / "stats.json"
    if stats_path.exists():
        text = replace_between(
            text,
            "<!-- DATA:START -->",
            "<!-- DATA:END -->",
            data_block(json.loads(stats_path.read_text())),
        )
    if user:
        text = text.replace("HF_USER", user).replace("HF_PLACEHOLDER", user)
    readme_path.write_text(text, encoding="utf-8")
    print(f"README updated (summary={'yes' if summary else 'no'}, user={user or 'unchanged'})")


if __name__ == "__main__":
    main()
