# collection-router-bench

[![CI](https://github.com/Grazinascito/collection-router-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Grazinascito/collection-router-bench/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Grazinascito/collection-router-bench/blob/main/notebooks/run_all.ipynb)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Dataset-blue)](https://huggingface.co/datasets/HF_USER/collection-router-bench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-blue)](https://huggingface.co/HF_USER/collection-router-e5-small)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Space-blue)](https://huggingface.co/spaces/HF_USER/collection-router-bench)

## Overview

Does reading the decision from **label logprobs** (`/v1/systemone`) change
latency and quality versus **generating JSON** (`/v1/chat/completions` with a
strict schema) — on the same model, the same vLLM server and the same GPU?
This benchmark answers it on the routing decision of a second-brain app
(História 3 of the Second Me PRD): each saved block goes into 0 or more of 11
collections; an empty set means Uncollected.

## Task

Multi-label classification of a saved block (note or link, with optional
preview data and content) into 11 collections defined by name, `includes` and
`excludes` (`data/collections.yaml`). PT-BR ~70% / EN ~30%. 360 test items,
440 train items (47 held-out validation in 13 groups), 25%+ hard boundary
items on documented confusable pairs.

## Method

| Arm | Decision mechanism | Endpoint |
| --- | --- | --- |
| A — generation | strict JSON schema, temperature 0, `max_tokens` 128 | `/v1/chat/completions` |
| B — typed decision | 11 `noul` (yes/no) questions, one per collection, fixed 0.5 threshold on P(yes) | `/v1/systemone` |
| C — classifier (extra) | fine-tuned `intfloat/multilingual-e5-small`, sigmoid ≥ 0.5 | in-process |

A and B always run **on the same model and the same server process**
(`server_id` recorded per request): only the decision mechanism differs.
Primary model `Qwen/Qwen3-4B-Instruct-2507` — the planned
`google/gemma-4-E4B-it-qat-w4a16-ct` has no read strategy in vLLM 0.32.0's
structured-decisions registry (`docs/api_check.md`, Q1), so the plan's rule
R11 switched the primary model. Protocol: warmup (10 items, not measured),
3 repetitions with alternating arm order (rep 1 A→B, rep 2 B→A, rep 3 A→B),
items shuffled with seed = rep, one request at a time, `cache_salt` per
run/rep/arm, 60 s timeout with one connection-error retry.

## How it works

```mermaid
flowchart LR
    B[Saved block\nstate_text] --> A[Arm A\nchat completions\nJSON schema]
    B --> S[Arm B\n/v1/systemone\n11 noul questions]
    A --> P1[parse JSON\ndedupe ids]
    S --> P2[P(yes) >= 0.5\nper collection]
    P1 --> M[metrics\nmicro/macro-F1, exact match,\nlatency p50/p95, paired bootstrap]
    P2 --> M
```

Arm A generates ~20 completion tokens per decision after one prompt prefill.
Arm B generates **zero** tokens: each of the 11 questions reads the logprobs
of the reply's first token (the label), so the cost is 11 prefills shared
through prefix caching plus 11 single-token reads. The token and decode
counts measured for each arm are recorded in `summary.json`
(`arms.A.tokens` / `arms.B.tokens`).

## Results

<!-- RESULTS:START -->
Results pending: run `notebooks/run_all.ipynb` (Colab T4, Run All) and then
`python scripts/update_readme.py`.
<!-- RESULTS:END -->

## Dataset

<!-- DATA:START -->
Dataset stats pending: run `python scripts/validate_data.py --stats` and then
`python scripts/update_readme.py`.
<!-- DATA:END -->

Synthetic data written by GLM (`builtin:zai-coding-plan/GLM-5.3`) through a
committed, seeded generator (`scripts/gen_data.py` + `scripts/banks_*.py`,
seed 42) — deterministic and reviewable. URLs are synthetic (they do not
open). 50-item human review sample ships in `data/review_sample.csv`;
corrections flow back through `scripts/apply_review.py`
(`data/review_stats.json`).

## Limitations

- Synthetic data with a 50-item human review; URLs do not open.
- One GPU class only (Colab T4, CC 7.5); no batch invariance (needs CC ≥ 8.0),
  so run-to-run variation is measured and reported instead.
- Arm C latency is measured in-process (no HTTP), unlike arms A/B.
- Prefix-cache asymmetry: arm A reuses the fixed prompt text across items;
  arm B reuses the state across its 11 questions.
- Arm B decides each collection independently (no joint normalization) with a
  fixed 0.5 threshold.
- One-day project: single task, single dataset, fixed prompts
  (`src/crb/prompts.lock`).

## Reproduce

```bash
# Mac / CPU: everything except the GPU run
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
python scripts/validate_data.py
python scripts/train_classifier.py --smoke
python scripts/run_bench.py --fake --limit 10 && python scripts/make_report.py --run-id fake
python scripts/publish_hf.py --dry-run
```

Full run: open [run_all.ipynb in Colab](
https://colab.research.google.com/github/Grazinascito/collection-router-bench/blob/main/notebooks/run_all.ipynb),
pick a **T4 GPU** runtime, add the `HF_TOKEN` secret, **Run All** (~50 min).
The notebook installs vLLM 0.32.0, starts one server for both arms, runs the
smoke checks with the plan's swap order, benches A/B with 3 reps, measures
throughput at concurrency 8 and 32, trains arm C, builds the report and
publishes dataset + model + Space + Collection to the Hugging Face Hub.

## Repository layout

```
data/            collections.yaml, train/test.jsonl, review sample
docs/            api_check.md (vLLM 0.32.0 source audit), decisions.md
notebooks/       run_all.ipynb (Colab, Run All)
scripts/         gen_data, validate_data, review, bench, train, report, publish
src/crb/         schema, data, prompts (+ prompts.lock), arms/, server, smoke,
                 bench, metrics, report, fake
space/           Gradio app (Results / Explore / Try / About)
tests/           unit + e2e tests against a fake server (no GPU needed)
results/         per-run outputs; raw/ is git-ignored
```

## License

MIT for code (see `LICENSE`). Dataset cc-by-4.0. Arm C model MIT, like its
base model.
