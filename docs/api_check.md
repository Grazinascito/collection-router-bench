# Block 0 — API check against the vLLM 0.32.0 wheel

Method: the PyPI wheel `vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl` (307 MB)
was downloaded and unzipped on the Mac; every answer below cites `file:line`
inside that wheel. Paths are relative to the wheel root (`src/vllm/...`).

## Q1 — Is there a read strategy for Gemma 4?

**No.** The structured-decisions registry is
`entrypoints/generate/structured_decisions/strategies.py:814-821`:

```python
READ_STRATEGIES: dict[str, type[ReadStrategy]] = {
    "ClefOmniForJointSchema": JointSchemaStrategy,
    "Lfm2VlForConditionalGeneration": LiquidStrategy,
    "Qwen3ForCausalLM": NextTokenStrategy,
    "Qwen3_5ForConditionalGeneration": NextTokenStrategy,
    "Qwen3_5MoeForConditionalGeneration": NextTokenStrategy,
    "DiffusionGemmaForBlockDiffusion": DiffusionGemmaCanvasStrategy,
}
```

`select_read_strategy` (`strategies.py:823-846`) looks the architecture up in
this dict and raises `ValueError` when it is absent. There is no generic
fallback for generation models and no `Gemma4ForConditionalGeneration` /
`Gemma4ForCausalLM` entry anywhere in the module.

**Decision applied (plan rule R11): the primary model becomes
`Qwen/Qwen3-4B-Instruct-2507`** (`Qwen3ForCausalLM` → `NextTokenStrategy`).
The vLLM version stays 0.32.0. The only Gemma entry is
`DiffusionGemmaForBlockDiffusion` (`strategies.py:358`), which is a diffusion
model, not the Gemma 4 we planned to use.

## Q2 — How does the server build the systemone prompt?

Entry: `ServingStructuredDecisions.create_decision`
(`entrypoints/generate/structured_decisions/serving.py:81-144`), which calls
`self.strategy.read(questions, request.instructions, self.strategy.render_state(request.state), ...)`
(`serving.py:107-108`).

For `NextTokenStrategy` (the one Qwen3 uses):

- **Request-level `instructions` become the system turn.**
  `strategies.py:315-316`:
  `if instructions: messages.insert(0, {"role": "system", "content": instructions})`.
- **`state` becomes the user turn.** A string state is used verbatim
  (`strategies.py:440-445`, `render_state`); non-string states are JSON-dumped
  with `ensure_ascii=False` — sending the state as a plain string keeps accents
  readable.
- **Each question is one request**: user content is
  `f"{state}\n\n\nQUESTION:\n{ask}"` (`strategies.py:515-517`).
- **`noul` ask format** (`strategies.py:495-497`):
  `f"{q.instructions}\nYes: {yes}\nNo: {no}\n\nReply with yes or no only."`
  where `yes`/`no` are the descriptions of the two options built from
  `criteria.true` / `criteria.false` (`question_types.py:177-186`).
- The requests share the state; with prefix caching the state is prefilled once
  (`strategies.py:187-190`, class docstring).
- `chat_template_kwargs={"enable_thinking": true}` is rejected
  (`strategies.py:264-266`): a read is the reply's first token, so thinking
  must be off. Sending `{"enable_thinking": false}` is accepted (and is the
  default the probe hardcodes, `strategies.py:217`).
- Limits for this strategy: `max_questions=64` (`strategies.py:258-259`);
  question specs accept exactly `type`, `instructions`, `criteria`
  (`protocol.py:18-21`); unknown fields raise
  (`serving.py:41-44`, `spec.model_extra`).

## Q3 — Does ChatCompletionRequest accept `cache_salt` and `chat_template_kwargs`?

**Yes, both.**

- `cache_salt`: `entrypoints/openai/chat_completion/protocol.py:485` (field)
  and `:548-551` (validator `validate_cache_salt`).
- `chat_template_kwargs`: `entrypoints/openai/chat_completion/protocol.py:366`.

`validate_cache_salt` (`entrypoints/generate/base/protocol.py:42-58`) requires:
non-empty string, at most 128 characters, and no `@`, `/`, `\` or NUL.
So the plan's salt format `"<run_id>-r<rep>-A"` is valid as long as the run id
avoids those characters.

**Decision applied (plan rule for Q3): no `/reset_prefix_cache` calls are
needed; every request carries a `cache_salt` derived from run id, rep and arm.**

## Q4 — Syntax of `--logprobs-mode` and `--limit-mm-per-prompt`

- `--logprobs-mode`: `engine/arg_utils.py:985`
  (`model_group.add_argument("--logprobs-mode", **model_kwargs["logprobs_mode"])`),
  field declared at `:609` with default `ModelConfig.logprobs_mode`. Accepted
  values are constrained to `raw_logprobs` / `processed_logprobs` —
  `strategies.py:812` (`LOGPROBS_MODES = frozenset({"raw_logprobs",
  "processed_logprobs"})`) and the check at `:840-845`.
- `--limit-mm-per-prompt`: `engine/arg_utils.py:634-636`, type
  `dict[str, int | dict[str, int]]`. The server start passes it as a JSON
  string, e.g. `--limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}'`.
  (Relevant only for multimodal models such as Gemma 4; with the Qwen3 primary
  model the flag is not needed. The server helper keeps it conditional on a
  Gemma model id, for the optional second-model run.)

## Q5 — Does `/reset_prefix_cache` exist, and does it need `VLLM_SERVER_DEV_MODE=1`?

Exists and needs the env var. The route is
`entrypoints/serve/dev/cache/api_router.py:20-41` (`@router.post("/reset_prefix_cache")`),
and the dev routers are only mounted when `envs.VLLM_SERVER_DEV_MODE` is truthy
(`entrypoints/launchers/api_server/routers.py:40`). Not used in this project,
because Q3 confirmed `cache_salt` (see above).

## Q6 — systemone error codes and bodies

Route `/v1/systemone` with declared responses 400, 404, 500, 501
(`entrypoints/generate/structured_decisions/api_router.py:26-32`).

- **501**: when the model has no read strategy, the handler is `None` and the
  route raises `NotImplementedError("The model does not support the structured
  decision API")` (`api_router.py:48-51`). The same happens for wrong-task
  models and for missing `--logprobs-mode` on generate-task models
  (`strategies.py:829-846`, logged as "/v1/systemone is disabled" at
  `api_router.py:78-79`, leaving the endpoint absent → 404/501 depending on
  mount).
- **400**: `StructuredDecisionError` (a `ValueError`) is returned as
  `create_error_response(e)` → HTTP 400 (`serving.py:125-126`;
  `question_types.py:33-34` documents "An invalid request. Returned as a 400").
  Triggers include: unknown question fields (`serving.py:41-44`), more than 64
  questions (`serving.py:36-39`), malformed `criteria`
  (`question_types.py:178-183`), `enable_thinking: true`
  (`strategies.py:264-266`).
- Body shape: `ErrorResponse` (`entrypoints/serve/engine/protocol.py`),
  serialized via `result.model_dump()` with `status_code=result.error.code`
  (`api_router.py:52-54`).

## Response shape facts used by arm B

- `answers.<id>` for a `noul` question is
  `{"type": "noul", "noul": P(yes), "probabilities": {"yes": ..., "no": ...},
  "confidence": max(probs) * label_mass}`
  (`question_types.py:196-206`).
- `diagnostics.<id>` carries `label_mass` and `argmax_is_label`
  (`serving.py:128-131`; `QuestionDiagnostics` in `protocol.py`).
- `usage` carries `input_tokens` / `output_tokens` summed over reads
  (`serving.py:132-136`; `DecisionUsage` in `protocol.py`).
- Yes/no label token variants scored server-side: `yes/Yes/YES` and `no/No/NO`
  (`strategies.py:468-470`); a label scores its best variant
  (`strategies.py:479-481`).
