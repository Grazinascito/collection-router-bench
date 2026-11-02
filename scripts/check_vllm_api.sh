#!/usr/bin/env bash
# Block 0: audit the vLLM 0.32.0 wheel for the systemone API (results in docs/api_check.md).
set -euo pipefail
DIR="${1:-/tmp/vllm-0.32.0}"
mkdir -p "$DIR" && cd "$DIR"
URL=$(python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('https://pypi.org/pypi/vllm/0.32.0/json')); print(next(u['url'] for u in d['urls'] if u['packagetype']=='bdist_wheel'))")
[ -f vllm.whl ] || curl -sL -o vllm.whl "$URL"
[ -d src ] || unzip -q -o vllm.whl -d src
echo "== files with systemone ==" && grep -rl "systemone" src/vllm | head -20
echo "== read strategy registry ==" && grep -n "READ_STRATEGIES" -A 8 src/vllm/entrypoints/generate/structured_decisions/strategies.py | head -12
echo "== cache_salt (chat) ==" && grep -n "cache_salt" src/vllm/entrypoints/openai/chat_completion/protocol.py | head -5
echo "== logprobs-mode / limit-mm ==" && grep -n "logprobs-mode\|limit_mm_per_prompt" src/vllm/engine/arg_utils.py | head -5
echo "== reset_prefix_cache ==" && grep -rn "reset_prefix_cache" src/vllm/entrypoints/serve/dev/cache/api_router.py | head -5
