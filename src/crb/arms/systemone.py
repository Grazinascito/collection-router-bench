"""Arm B: typed decisions through POST /v1/systemone (one noul question per
collection, threshold 0.5 on P(yes)).

HTTP 400/501 raise SystemOneUnavailable so the smoke step can switch to the
next configuration; other HTTP errors become empty predictions with error=True.
"""

from __future__ import annotations

import time

import httpx

from crb.prompts import noul_question, state_text, systemone_instructions
from crb.schema import Block, Collection

THRESHOLD = 0.5


class SystemOneUnavailable(Exception):
    """The server answered 400/501 for /v1/systemone."""


class SystemOneArm:
    arm = "B"
    arm_b_impl = "systemone"

    def __init__(self, model_id: str, collections: list[Collection]) -> None:
        self.model_id = model_id
        self.collections = collections

    # ------------------------------------------------------------------
    def build_request(self, block: Block, run_id: str, rep: int) -> dict:
        questions: dict[str, dict] = {}
        for c in self.collections:
            questions[c.id] = {
                "type": "noul",
                "instructions": noul_question(c),
                "criteria": {
                    "true": " ".join(c.includes.split()),
                    "false": " ".join(c.excludes.split()),
                },
            }
        return {
            "model": self.model_id,
            "state": state_text(block),
            "instructions": systemone_instructions(self.collections),
            "questions": questions,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": f"{run_id}-r{rep}-B",
        }

    # ------------------------------------------------------------------
    def parse_response(self, body: dict) -> dict:
        answers = body.get("answers", {})
        diagnostics = body.get("diagnostics", {})
        probs: dict[str, float] = {}
        label_masses: dict[str, float] = {}
        for c in self.collections:
            ans = answers.get(c.id, {})
            diag = diagnostics.get(c.id, {})
            probs[c.id] = float(ans.get("noul", 0.0))
            label_masses[c.id] = float(diag.get("label_mass", 0.0))
        pred = [c.id for c in self.collections if probs[c.id] >= THRESHOLD]
        mean_mass = sum(label_masses.values()) / len(label_masses) if label_masses else 0.0
        return {
            "pred": pred,
            "probs": probs,
            "label_mass": mean_mass,
            "usage": body.get("usage") or {},
            "finish_reason": None,
            "format_error": False,
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
        response = await client.post("/v1/systemone", json=request)
        latency_ms = (time.perf_counter_ns() - start) / 1e6
        if response.status_code in (400, 501):
            raise SystemOneUnavailable(f"HTTP {response.status_code}: {response.text[:200]}")
        response.raise_for_status()
        out = self.parse_response(response.json())
        out["latency_ms"] = latency_ms
        return out


async def decide_with_error_handling(
    arm: SystemOneArm, client, block, run_id, rep, cache_salt=None
) -> dict:
    """Wrap decide(): non-availability HTTP errors become error=True records."""
    try:
        return await arm.decide(client, block, run_id, rep, cache_salt)
    except SystemOneUnavailable:
        raise
    except httpx.HTTPError:
        return {
            "pred": [],
            "probs": None,
            "label_mass": None,
            "usage": {},
            "finish_reason": None,
            "format_error": False,
            "error": True,
            "latency_ms": 0.0,
        }
