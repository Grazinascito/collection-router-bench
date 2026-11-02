"""Smoke check: 5 train items through arms A and B before the real run.

Swap order (plan, Block 2): (primary, systemone) -> (fallback, systemone) ->
(fallback, chat_logprobs). Each attempt lands in run_meta["smoke_attempts"].
Success prints `SMOKE OK <model_id> <arm_b_impl>`.
"""

from __future__ import annotations

import math

import httpx

from crb.arms.generation import GenerationArm
from crb.arms.logprob_chat import LogprobChatArm
from crb.arms.systemone import SystemOneArm, SystemOneUnavailable
from crb.schema import Block, Collection

N_ITEMS = 5
MIN_MEAN_LABEL_MASS = 0.5


def build_arm_b(impl: str, model_id: str, collections: list[Collection]):
    if impl == "systemone":
        return SystemOneArm(model_id, collections)
    if impl == "chat_logprobs":
        return LogprobChatArm(model_id, collections)
    raise ValueError(f"unknown arm B implementation: {impl}")


def swap_order(primary_model: str, fallback_model: str) -> list[tuple[str, str]]:
    """(model_id, arm_b_impl) configurations in the plan's swap order."""
    return [
        (primary_model, "systemone"),
        (fallback_model, "systemone"),
        (fallback_model, "chat_logprobs"),
    ]


async def check_config(
    client: httpx.AsyncClient,
    model_id: str,
    arm_b_impl: str,
    items: list[Block],
    collections: list[Collection],
    run_id: str,
) -> bool:
    arm_a = GenerationArm(model_id, collections)
    arm_b = build_arm_b(arm_b_impl, model_id, collections)

    valid_json = 0
    for block in items:
        out = await arm_a.decide(client, block, run_id, rep=0, cache_salt="warmup")
        if not out["format_error"] and not out["error"]:
            valid_json += 1

    probs: list[float] = []
    masses: list[float] = []
    for block in items:
        out = await arm_b.decide(client, block, run_id, rep=0, cache_salt="warmup")
        values = (out.get("probs") or {}).values()
        probs.extend(values)
        masses.append(out.get("label_mass") or 0.0)

    ok_a = valid_json == N_ITEMS
    ok_b = (
        len(probs) == N_ITEMS * len(collections)
        and all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs)
        and (sum(masses) / len(masses)) >= MIN_MEAN_LABEL_MASS
    )
    return ok_a and ok_b


async def run_smoke(
    client_for,  # async callable: model_id -> httpx.AsyncClient
    items: list[Block],
    collections: list[Collection],
    primary_model: str,
    fallback_model: str,
    run_id: str,
    run_meta: dict,
) -> tuple[str, str]:
    """Try each configuration in swap order; raise if all fail.

    `client_for(model_id)` must return an AsyncClient bound to a healthy server
    serving that model.
    """
    attempts = run_meta.setdefault("smoke_attempts", [])
    last_error = None
    for model_id, impl in swap_order(primary_model, fallback_model):
        try:
            client = await client_for(model_id)
            ok = await check_config(client, model_id, impl, items[:N_ITEMS], collections, run_id)
            attempts.append({"model_id": model_id, "arm_b_impl": impl, "ok": ok, "error": None})
            if ok:
                run_meta["model_id"] = model_id
                run_meta["arm_b_impl"] = impl
                print(f"SMOKE OK {model_id} {impl}")
                return model_id, impl
            last_error = f"checks failed for {model_id}/{impl}"
        except SystemOneUnavailable as exc:  # noqa: PERF203
            attempts.append(
                {"model_id": model_id, "arm_b_impl": impl, "ok": False, "error": str(exc)[:200]}
            )
            last_error = str(exc)
    raise RuntimeError(f"smoke failed for every configuration; last error: {last_error}")
