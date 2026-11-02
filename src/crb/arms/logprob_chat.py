"""Arm B fallback (B'): read yes/no logprobs from /v1/chat/completions.

Used only when /v1/systemone fails on both models. Eleven concurrent requests
per item (one per collection), max_tokens=1, and P(yes) normalized over the
yes/no token variants present in the top-20 logprobs.
"""

from __future__ import annotations

import asyncio
import math
import time

import httpx

from crb.prompts import noul_question, state_text, systemone_instructions
from crb.schema import Block, Collection

TOP_LOGPROBS = 20

# The server-side noul scoring uses yes/Yes/YES and no/No/NO
# (strategies.py _label_groups); the chat fallback also admits a leading space.
YES_FORMS = ("yes", "Yes", "YES", " yes", " Yes")
NO_FORMS = ("no", "No", "NO", " no", " No")


class LogprobChatArm:
    arm = "B"
    arm_b_impl = "chat_logprobs"

    def __init__(self, model_id: str, collections: list[Collection]) -> None:
        self.model_id = model_id
        self.collections = collections

    # ------------------------------------------------------------------
    def build_requests(self, block: Block, run_id: str, rep: int) -> list[dict]:
        state = state_text(block)
        requests = []
        for c in self.collections:
            requests.append(
                {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": systemone_instructions(self.collections)},
                        {
                            "role": "user",
                            "content": (
                                f"{noul_question(c)} Answer yes or no only.\n\nBlock:\n{state}"
                            ),
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": TOP_LOGPROBS,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "cache_salt": f"{run_id}-r{rep}-B",
                }
            )
        return requests

    # ------------------------------------------------------------------
    @staticmethod
    def parse_top_logprobs(entry: dict | None) -> dict:
        """P(yes) and label_mass from one top_logprobs entry.

        Returns {"p_yes": float, "label_mass": float, "error": bool}; the error
        flag is set when neither yes nor no variants appear in the top-k.
        """
        if not entry:
            return {"p_yes": 0.0, "label_mass": 0.0, "error": True}
        yes_mass = 0.0
        no_mass = 0.0
        for token, logprob in entry.items():
            p = math.exp(logprob)
            if token in YES_FORMS:
                yes_mass += p
            elif token in NO_FORMS:
                no_mass += p
        label_mass = yes_mass + no_mass
        if label_mass <= 0.0:
            return {"p_yes": 0.0, "label_mass": 0.0, "error": True}
        return {"p_yes": yes_mass / label_mass, "label_mass": label_mass, "error": False}

    # ------------------------------------------------------------------
    async def decide(
        self,
        client: httpx.AsyncClient,
        block: Block,
        run_id: str,
        rep: int,
        cache_salt: str | None = None,
    ) -> dict:
        requests = self.build_requests(block, run_id, rep)
        if cache_salt is not None:
            for r in requests:
                r["cache_salt"] = cache_salt
        start = time.perf_counter_ns()
        responses = await asyncio.gather(
            *[client.post("/v1/chat/completions", json=r) for r in requests]
        )
        latency_ms = (time.perf_counter_ns() - start) / 1e6

        probs: dict[str, float] = {}
        masses: list[float] = []
        errors = 0
        for c, response in zip(self.collections, responses, strict=True):
            if response.status_code != 200:
                probs[c.id] = 0.0
                masses.append(0.0)
                errors += 1
                continue
            choice = (response.json().get("choices") or [{}])[0]
            top = (choice.get("logprobs") or {}).get("top_logprobs") or []
            parsed = self.parse_top_logprobs(top[0] if top else None)
            probs[c.id] = parsed["p_yes"]
            masses.append(parsed["label_mass"])
            errors += int(parsed["error"])
        pred = [c.id for c in self.collections if probs[c.id] >= 0.5]
        usage = {"prompt_tokens": 0, "completion_tokens": len(requests)}
        return {
            "pred": pred,
            "probs": probs,
            "label_mass": sum(masses) / len(masses) if masses else 0.0,
            "usage": usage,
            "finish_reason": None,
            "format_error": False,
            "error": errors > 0,
            "latency_ms": latency_ms,
        }
