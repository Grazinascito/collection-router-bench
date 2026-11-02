"""Publish dataset, model, Space and Collection to the Hugging Face Hub.

Idempotent (exist_ok=True on every repo). --dry-run writes the cards to
cards_out/ locally and touches nothing on the Hub.

Order: dataset, model, Space, Collection (a Collection only accepts items
that already exist).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crb.data import load_collections  # noqa: E402

DATASET_NAME = "collection-router-bench"
MODEL_NAME = "collection-router-e5-small"
SPACE_NAME = "collection-router-bench"
COLLECTION_TITLE = "Collection Router: typed decisions vs JSON generation"
GEN_MODEL_NOTE = (
    "Data written by GLM (builtin:zai-coding-plan/GLM-5.3) via a committed seeded "
    "generator (scripts/gen_data.py + banks)."
)


def hf_user(api) -> str:
    return api.whoami()["name"]


def stage_split_files(stage: Path) -> None:
    """Dataset wants separate train/validation/test JSONL files with configs."""
    rows = [
        json.loads(x)
        for x in (REPO / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    (stage / "data").mkdir(parents=True, exist_ok=True)
    for name, batch in (("train", train), ("validation", validation)):
        (stage / "data" / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch), encoding="utf-8"
        )
    shutil.copy(REPO / "data" / "test.jsonl", stage / "data" / "test.jsonl")
    shutil.copy(REPO / "data" / "collections.yaml", stage / "data" / "collections.yaml")


def dataset_card(user: str, stats: dict | None) -> str:
    review = ""
    stats_path = REPO / "data" / "review_stats.json"
    if stats_path.exists():
        review_data = json.loads(stats_path.read_text())
        review = (
            f"Human review: {review_data['n_reviewed']} items reviewed, "
            f"{review_data['n_ok']} kept as-is, {review_data['n_fixed']} corrected "
            "(`review_stats.json`)."
        )
    else:
        review = "Human review: pending (50-item sample shipped in `review_sample.csv`)."
    counts = ""
    if stats:
        counts = (
            f"- test: {stats['test_total']} items; train: {stats['train_total']}; "
            f"validation: {stats['validation_total']}\n"
            f"- max test/train char 3-5gram TF-IDF similarity: "
            f"{stats['max_test_train_similarity']}\n"
        )
    return f"""---
license: cc-by-4.0
language:
  - pt
  - en
task_categories:
  - text-classification
task_ids:
  - multi-label-classification
size_categories:
  - n<1K
tags:
  - synthetic
  - routing
  - vllm
  - structured-decisions
  - benchmark
configs:
  - config_name: train
    data_files:
      - split: train
        path: data/train.jsonl
  - config_name: validation
    data_files:
      - split: validation
        path: data/validation.jsonl
  - config_name: test
    data_files:
      - split: test
        path: data/test.jsonl
---

# collection-router-bench (dataset)

Multi-label routing benchmark for a second-brain app: given a saved block
(note or link, with optional preview and content), choose 0 or more of 11
collections. Empty set = Uncollected. Used to compare JSON generation (A),
typed logprob decisions via vLLM `/v1/systemone` (B) and a fine-tuned
multilingual-e5-small classifier (C).

## Schema

One JSON object per line: `id, split, kind (note|link), user_title,
user_description, url, preview {{site_name, title, description}}, content,
content_status (note|read|not_read), lang (pt|en), labels (0-2 of 11),
difficulty (easy|hard), group_id, rationale`.

## Collections

{chr(10).join(f"- {c.id}: {c.name}" for c in load_collections(REPO / "data" / "collections.yaml"))}

Definitions (`includes`/`excludes`) ship in `data/collections.yaml`.

## How the data was made

Synthetic data, written by an LLM. {GEN_MODEL_NOTE} Deterministic under seed 42;
regenerate with `python scripts/gen_data.py` and check with
`python scripts/validate_data.py`. {counts}
{review}

## Limitations

Synthetic items with realistic-but-invented sources; URLs contain `synthetic`
in the path and do not open. ~70% pt-BR / ~30% en. Boundary difficulty is
template-driven: hard items sit on documented confusable pairs.

## License

cc-by-4.0 for the data. Code is MIT in the GitHub repository.
Results under `results/` follow the same license.
"""


def model_card(user: str, summary: dict | None) -> str:
    mi = ""
    if summary and summary.get("classifier"):
        clf = summary["classifier"]
        mi = f"""
model-index:
  - name: collection-router-e5-small
    results:
      - task:
          type: text-classification
          name: Multi-label Text Classification
        dataset:
          type: {user}/collection-router-bench
          name: collection-router-bench (test)
        metrics:
          - type: f1
            name: micro F1
            value: {clf["micro_f1"]}
          - type: f1
            name: macro F1
            value: {clf["macro_f1"]}
          - type: accuracy
            name: exact match
            value: {clf["exact_match"]}
"""
    metrics = ""
    if summary and summary.get("classifier"):
        clf = summary["classifier"]
        metrics = (
            f"Test micro-F1 {clf['micro_f1']}, "
            f"macro-F1 {clf['macro_f1']}, exact match {clf['exact_match']} "
            f"(in-process batch-1 latency p50 {clf['latency']['p50_ms']} ms)."
        )
    return f"""---
license: mit
base_model: intfloat/multilingual-e5-small
pipeline_tag: text-classification
language:
  - pt
  - en
datasets:
  - {user}/collection-router-bench
tags:
  - multi-label-classification
  - routing
  - second-brain{mi}
---

# collection-router-e5-small

Arm C of the [collection-router-bench] benchmark: `intfloat/multilingual-e5-small`
fine-tuned for multi-label routing into 11 collections (empty set = Uncollected).

Input format: `"query: " + state_text` (see the repo's `crb/prompts.py`),
max 512 tokens. Prediction: sigmoid >= 0.5 per label.

{metrics}

Trained with `scripts/train_classifier.py` (8 epochs, lr 5e-5, batch 16,
weight decay 0.01, warmup 10%, best epoch by validation micro-F1, seed 42).
"""


def space_card(user: str, gradio_version: str) -> str:
    return f"""---
title: collection-router-bench
emoji: 🗂️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
license: mit
datasets:
  - {user}/collection-router-bench
models:
  - {user}/collection-router-e5-small
tags:
  - benchmark
  - routing
pinned: false
---

# collection-router-bench (Space)

Results explorer for the benchmark comparing JSON generation (A), typed
logprob decisions (B) and a small classifier (C) for multi-label routing.
Tabs: Results, Explore, Try (arm C on CPU), About. No secrets needed.
"""


def gradio_pinned_version() -> str:
    try:
        import gradio

        return gradio.__version__
    except ImportError:
        return "5.49.1"


def dry_run(run_id: str | None, out: Path) -> None:
    user = "HF_USER"
    stats = None
    stats_path = REPO / "data" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    summary = None
    summary_path = REPO / "results" / "latest" / "summary.json"
    if not summary_path.exists() and run_id:
        summary_path = REPO / "results" / run_id / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    (out / "dataset").mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "space").mkdir(parents=True, exist_ok=True)
    (out / "dataset" / "README.md").write_text(dataset_card(user, stats), encoding="utf-8")
    (out / "model" / "README.md").write_text(model_card(user, summary), encoding="utf-8")
    (out / "space" / "README.md").write_text(
        space_card(user, gradio_pinned_version()), encoding="utf-8"
    )
    print(f"dry-run cards written to {out}")


def publish(run_id: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    user = hf_user(api)
    stats = None
    if (REPO / "data" / "stats.json").exists():
        stats = json.loads((REPO / "data" / "stats.json").read_text())
    results_dir = REPO / "results" / (run_id or "latest")
    if not results_dir.exists():
        # fall back to the newest run directory
        candidates = sorted((REPO / "results").glob("*/summary.json"))
        if candidates:
            results_dir = candidates[-1].parent
    summary = None
    if (results_dir / "summary.json").exists():
        summary = json.loads((results_dir / "summary.json").read_text())

    dataset_repo = f"{user}/{DATASET_NAME}"
    model_repo = f"{user}/{MODEL_NAME}"
    space_repo = f"{user}/{SPACE_NAME}"

    # 1) dataset
    api.create_repo(dataset_repo, repo_type="dataset", exist_ok=True, private=False)
    stage = REPO / "cards_out" / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage_split_files(stage)
    (stage / "README.md").write_text(dataset_card(user, stats), encoding="utf-8")
    if results_dir.exists():
        shutil.copytree(results_dir, stage / "results" / results_dir.name, dirs_exist_ok=True)
        shutil.copytree(results_dir, stage / "results" / "latest", dirs_exist_ok=True)
    api.upload_folder(repo_id=dataset_repo, repo_type="dataset", folder_path=str(stage))
    print(f"dataset: https://huggingface.co/datasets/{dataset_repo}")

    # 2) model
    api.create_repo(model_repo, exist_ok=True, private=False)
    model_dir = REPO / "models" / MODEL_NAME
    if model_dir.exists():
        api.upload_folder(repo_id=model_repo, folder_path=str(model_dir))
    else:
        print("model dir missing (train arm C first); uploading card only")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "README.md").write_text(model_card(user, summary), encoding="utf-8")
        api.upload_folder(repo_id=model_repo, folder_path=td)
    print(f"model: https://huggingface.co/{model_repo}")

    # 3) space
    api.create_repo(space_repo, repo_type="space", exist_ok=True, private=False)
    space_stage = REPO / "cards_out" / "space_stage"
    if space_stage.exists():
        shutil.rmtree(space_stage)
    shutil.copytree(REPO / "space", space_stage)
    (space_stage / "README.md").write_text(
        space_card(user, gradio_pinned_version()), encoding="utf-8"
    )
    if results_dir.exists():
        shutil.copytree(results_dir, space_stage / "results" / "latest", dirs_exist_ok=True)
        shutil.copy(REPO / "data" / "test.jsonl", space_stage / "data" / "test.jsonl")
        shutil.copy(REPO / "data" / "collections.yaml", space_stage / "data" / "collections.yaml")
    (space_stage / "runtime.json").write_text(
        json.dumps({"model_id": model_repo}), encoding="utf-8"
    )
    api.upload_folder(repo_id=space_repo, repo_type="space", folder_path=str(space_stage))
    print(f"space: https://huggingface.co/spaces/{space_repo}")

    # 4) collection
    slug = api.create_collection(
        title=COLLECTION_TITLE, namespace=user, private=False, exists_ok=True
    )
    for item, kind in [
        (dataset_repo, "dataset"),
        (model_repo, "model"),
        (space_repo, "space"),
        ("intfloat/multilingual-e5-small", "model"),
        ("Qwen/Qwen3-4B-Instruct-2507", "model"),
    ]:
        try:
            api.add_collection_item(slug, item=item, item_type=kind, exists_ok=True)
        except Exception as exc:  # noqa: BLE001 - collection items are best-effort
            print(f"collection item {item}: {exc}")
    print(f"collection: https://huggingface.co/collections/{slug}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args.run_id, REPO / "cards_out")
    else:
        publish(args.run_id)


if __name__ == "__main__":
    main()
