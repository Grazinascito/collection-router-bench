# Decisions (ASSUMPTIONs taken)

Entries marked **[plan]** are ASSUMPTIONs inherited from the approved plan
(2026-11-01). Entries marked **[new]** were taken during implementation.

1. **[plan, verified]** Primary model is `Qwen/Qwen3-4B-Instruct-2507`, not
   `google/gemma-4-E4B-it-qat-w4a16-ct`. Reason: Block 0 (see
   `docs/api_check.md`, Q1) shows vLLM 0.32.0 has no read strategy for Gemma 4
   architectures. The plan's rule R11 prescribed this switch. Gemma 4 remains
   the `FALLBACK_MODEL_ID` parameter, usable only for arm A; since A and B must
   share the model and server, the benchmark runs Qwen3 unless the operator
   overrides it.
2. **[plan, verified]** No `/reset_prefix_cache` calls: `cache_salt` is accepted
   by chat completions and by systemone (`docs/api_check.md`, Q3/Q5). Every
   request carries `cache_salt = "<run_id>-r<rep>-<arm>"`.
3. **[plan]** Docker tag `vllm/vllm-openai:v0.32.0` exists. Only relevant for
   risk R3 (Modal fallback).
4. **[plan]** Licenses: code MIT, dataset cc-by-4.0, arm C model MIT.
5. **[plan]** Collection descriptions (`includes`/`excludes`) in EN.
6. **[plan]** Synthetic URLs on real domains; the dataset card states they do
   not open.
7. **[plan]** No prompt tuning before the Run All; `prompts.lock` freezes the
   templates.
8. **[plan]** Repo names from section 2 of the plan (GitHub
   `collection-router-bench`; HF dataset `collection-router-bench`, model
   `collection-router-e5-small`, Space `collection-router-bench`).
9. **[new]** Local repo path:
   `~/.zcode/workspace/default/collection-router-bench` (Mac). The Colab run
   clones from GitHub, so the local path does not affect reproducibility.
10. **[new]** The 800 dataset items are produced by a committed, seeded
    generator (`scripts/gen_data.py` + `scripts/data_banks.py`) instead of
    one-off handwritten JSONL files. Reason: 800 items with realistic
    multilingual content cannot be hand-typed in the project's time box, and a
    seeded generator makes the dataset exactly reproducible. The content banks
    (topics, sentences, boundary cues, rationales) were written by GLM
    (model `builtin:zai-coding-plan/GLM-5.3`); the dataset card discloses this.
11. **[new]** The earlier repository `Grazinascito/second-me-router` (pushed
    2026-11-01 ~20:01, an earlier draft of this benchmark) is left untouched;
    this project creates the new `collection-router-bench` repo. Reason: the
    approved plan declares the project new and fixes the new name.
12. **[new]** Python for local dev/venv: 3.12 (matches CI and Colab).
    `requires-python = ">=3.11,<3.15"` follows vLLM 0.32.0's own range.
