"""Measurement protocol (plan, Block 2).

- Warmup: 10 train items on both arms with cache_salt "warmup" (not recorded).
- 3 reps; arm order alternates (rep 1 A->B, rep 2 B->A, rep 3 A->B); items
  shuffled with seed = rep number.
- One request at a time; latency = perf_counter_ns from send to full response.
- Resumable: (arm, rep, item_id) keys already in the raw JSONL are skipped.
- Timeout 60 s; one retry, connection errors only.
- Throughput (Should): concurrency 8 and 32, one pass over the test set per arm.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from crb.arms.generation import GenerationArm
from crb.arms.logprob_chat import LogprobChatArm
from crb.arms.systemone import SystemOneArm, SystemOneUnavailable
from crb.schema import Block, Collection

TIMEOUT_S = 60.0
WARMUP_ITEMS = 10


def arm_order(rep: int) -> list[str]:
    """Rep 1 A->B, rep 2 B->A, rep 3 A->B, ..."""
    return ["A", "B"] if rep % 2 == 1 else ["B", "A"]


def shuffled(items: list[Block], rep: int) -> list[Block]:
    ordered = list(items)
    random.Random(rep).shuffle(ordered)
    return ordered


def make_arm(arm: str, arm_b_impl: str, model_id: str, collections: list[Collection]):
    if arm == "A":
        return GenerationArm(model_id, collections)
    if arm == "B":
        if arm_b_impl == "systemone":
            return SystemOneArm(model_id, collections)
        return LogprobChatArm(model_id, collections)
    raise ValueError(f"unknown arm {arm}")


def existing_keys(raw_dir: Path) -> set[tuple[str, int, str]]:
    done: set[tuple[str, int, str]] = set()
    if not raw_dir.exists():
        return done
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["arm"], row["rep"], row["item_id"]))
    return done


def append_record(raw_dir: Path, arm: str, record: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / f"{arm}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_one(arm_obj, client, block, run_id, rep, cache_salt=None) -> dict:
    """One request with a single retry on connection errors."""
    last_exc: Exception | None = None
    for _attempt in range(2):
        try:
            return await arm_obj.decide(client, block, run_id, rep, cache_salt)
        except (httpx.ConnectError, httpx.TransportError) as exc:
            last_exc = exc
            await asyncio.sleep(1.0)
    return {
        "pred": [],
        "probs": None,
        "label_mass": None,
        "usage": {},
        "finish_reason": None,
        "format_error": False,
        "error": True,
        "latency_ms": 0.0,
        "_exception": str(last_exc)[:200],
    }


async def warmup(arms: dict, client, items, run_id) -> None:
    for block in items[:WARMUP_ITEMS]:
        for _arm_name, arm_obj in arms.items():
            try:
                await arm_obj.decide(client, block, run_id, rep=0, cache_salt="warmup")
            except httpx.HTTPError:
                pass


async def run_bench(
    arms: dict,
    client,
    items: list[Block],
    run_id: str,
    raw_dir: Path,
    reps: int = 3,
    meta: dict | None = None,
    progress=None,
) -> int:
    """Warm up, then run reps with alternating arm order. Returns record count."""
    await warmup(arms, client, items, run_id)
    done = existing_keys(raw_dir)
    written = 0
    for rep in range(1, reps + 1):
        for arm_name in arm_order(rep):
            arm_obj = arms[arm_name]
            for block in shuffled(items, rep):
                key = (arm_name, rep, block.id)
                if key in done:
                    continue
                out = await run_one(arm_obj, client, block, run_id, rep)
                record = {
                    "run_id": run_id,
                    "server_id": (meta or {}).get("server_id"),
                    "model_id": (meta or {}).get("model_id"),
                    "model_sha": (meta or {}).get("model_sha"),
                    "arm": arm_name,
                    "arm_b_impl": (meta or {}).get("arm_b_impl") if arm_name == "B" else None,
                    "rep": rep,
                    "item_id": block.id,
                    "latency_ms": round(out["latency_ms"], 3),
                    "pred": out["pred"],
                    "probs": out.get("probs"),
                    "label_mass": out.get("label_mass"),
                    "usage": out.get("usage"),
                    "finish_reason": out.get("finish_reason"),
                    "format_error": out.get("format_error", False),
                    "error": out.get("error", False),
                }
                append_record(raw_dir, arm_name, record)
                done.add(key)
                written += 1
                if progress is not None:
                    progress()
    return written


async def run_throughput(
    arms: dict,
    client,
    items: list[Block],
    run_id: str,
    concurrency: int,
) -> dict[str, dict]:
    """One pass over `items` per arm at fixed concurrency; wall-clock based."""
    results: dict[str, dict] = {}
    for arm_name, arm_obj in arms.items():
        latencies: list[float] = []
        errors = 0

        async def worker(block: Block, arm_obj=arm_obj, latencies=latencies) -> None:
            nonlocal errors
            out = await run_one(arm_obj, client, block, run_id, rep=0, cache_salt="warmup")
            latencies.append(out["latency_ms"])
            errors += int(out.get("error", False))

        sem = asyncio.Semaphore(concurrency)
        started = time.perf_counter()

        async def bounded(block: Block, sem=sem) -> None:
            async with sem:
                await worker(block)

        await asyncio.gather(*[bounded(block) for block in items])
        wall = time.perf_counter() - started
        latencies.sort()
        n = len(latencies)
        results[arm_name] = {
            "concurrency": concurrency,
            "n": n,
            "errors": errors,
            "wall_s": round(wall, 3),
            "decisions_per_hour": round(n / wall * 3600, 1) if wall else 0.0,
            "p50_ms": round(latencies[n // 2], 3) if n else 0.0,
            "p95_ms": round(latencies[int(n * 0.95)] if n else 0.0, 3),
        }
    return results


__all__ = [
    "arm_order",
    "shuffled",
    "make_arm",
    "existing_keys",
    "append_record",
    "run_one",
    "run_bench",
    "run_throughput",
    "SystemOneUnavailable",
]
