"""Arm A: JSON generation through /v1/chat/completions with a strict JSON schema.

No fallback: invalid JSON or finish_reason == "length" produce an empty
prediction with format_error=True so the failure shows up in the metrics.
"""

from __future__ import annotations

import json
import time

import httpx

from crb.prompts import generation_message
from crb.schema import Block, Collection

MAX_TOKENS = 128


class GenerationArm:
    arm = "A"

    def __init__(self, model_id: str, collections: list[Collection]) -> None:
        self.model_id = model_id
        self.collections = collections

    # ------------------------------------------------------------------
    def build_request(self, block: Block, run_id: str, rep: int) -> dict:
        enum_ids = [c.id for c in self.collections]
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": generation_message(block, self.collections)}],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "collections",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "collections": {
                                "type": "array",
                                "items": {"type": "string", "enum": enum_ids},
                            }
                        },
                        "required": ["collections"],
                        "additionalProperties": False,
                    },
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": f"{run_id}-r{rep}-A",
        }

    # ------------------------------------------------------------------
    def parse_completion(self, body: dict) -> dict:
        choice = (body.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        content = (choice.get("message") or {}).get("content", "")
        labels: list[str] = []
        format_error = False
        try:
            data = json.loads(content)
            raw = data.get("collections", [])
            valid = {c.id for c in self.collections}
            for item in raw:
                if item in valid and item not in labels:
                    labels.append(item)
        except (json.JSONDecodeError, AttributeError, TypeError):
            format_error = True
        if finish_reason == "length":
            format_error = True
            labels = []
        return {
            "pred": labels,
            "probs": None,
            "usage": body.get("usage") or {},
            "finish_reason": finish_reason,
            "format_error": format_error,
            "error": False,
        }

    # ------------------------------------------------------------------
    async def decide(
        self,
        client: httpx.AsyncClient,
        block: Block,
        run_id: str,
        rep: int,
        cache_salt: str | None = None,
    ) -> dict:
        request = self.build_request(block, run_id, rep)
        if cache_salt is not None:
            request["cache_salt"] = cache_salt
        start = time.perf_counter_ns()
        response = await client.post("/v1/chat/completions", json=request)
        latency_ms = (time.perf_counter_ns() - start) / 1e6
        out = self.parse_completion(response.json())
        out["latency_ms"] = latency_ms
        return out


async def decide_with_error_handling(
    arm: GenerationArm, client, block, run_id, rep, cache_salt=None
) -> dict:
    """Wrap decide(): transport errors become empty predictions with error=True."""
    try:
        return await arm.decide(client, block, run_id, rep, cache_salt)
    except httpx.HTTPError:
        return {
            "pred": [],
            "probs": None,
            "usage": {},
            "finish_reason": None,
            "format_error": False,
            "error": True,
            "latency_ms": 0.0,
        }
