"""The measurement protocol: arm order, shuffle seeds, resume, record fields."""

from __future__ import annotations

import asyncio
import json

from crb.bench import append_record, arm_order, existing_keys, make_arm, run_bench, shuffled
from crb.fake import FakeTruth, make_client
from crb.prompts import state_text
from crb.schema import Record


def test_arm_order_alternates():
    assert arm_order(1) == ["A", "B"]
    assert arm_order(2) == ["B", "A"]
    assert arm_order(3) == ["A", "B"]


def test_shuffle_seed_is_rep(splits):
    rep1 = [b.id for b in shuffled(splits["test"], 1)]
    rep1_again = [b.id for b in shuffled(splits["test"], 1)]
    rep2 = [b.id for b in shuffled(splits["test"], 2)]
    assert rep1 == rep1_again
    assert rep1 != rep2
    assert sorted(rep1) == sorted(rep2) == sorted(b.id for b in splits["test"])


def test_resume_skips_done(splits, collections, tmp_path):
    items = splits["test"][:4]
    raw_dir = tmp_path / "raw"
    done = {("A", 1, items[0].id)}
    for arm, rep, item_id in done:
        append_record(raw_dir, arm, {"arm": arm, "rep": rep, "item_id": item_id})
    assert existing_keys(raw_dir) == done

    truth = FakeTruth({state_text(b): list(b.labels) for b in items})

    async def _run() -> int:
        client = make_client(truth)
        try:
            arms = {
                "A": make_arm("A", "systemone", "fake/model", collections),
                "B": make_arm("B", "systemone", "fake/model", collections),
            }
            return await run_bench(
                arms, client, items, "resume-run", raw_dir, reps=1, meta={"model_id": "fake/model"}
            )
        finally:
            await client.aclose()

    written = asyncio.run(_run())
    assert written == 2 * len(items) - 1  # (A,1,i1) was already done


def test_record_fields(splits, collections, tmp_path):
    items = splits["test"][:2]
    raw_dir = tmp_path / "raw2"

    async def _run() -> list[dict]:
        truth = FakeTruth({state_text(b): list(b.labels) for b in items})
        client = make_client(truth)
        try:
            arms = {
                "A": make_arm("A", "systemone", "fake/model", collections),
                "B": make_arm("B", "systemone", "fake/model", collections),
            }
            await run_bench(
                arms,
                client,
                items,
                "fields-run",
                raw_dir,
                reps=1,
                meta={
                    "model_id": "fake/model",
                    "server_id": "srv-1",
                    "model_sha": "abc",
                    "arm_b_impl": "systemone",
                },
            )
        finally:
            await client.aclose()
        rows = []
        for arm in ("A", "B"):
            rows += [json.loads(x) for x in (raw_dir / f"{arm}.jsonl").read_text().splitlines()]
        return rows

    rows = asyncio.run(_run())
    expected = {
        "run_id",
        "server_id",
        "model_id",
        "model_sha",
        "arm",
        "arm_b_impl",
        "rep",
        "item_id",
        "latency_ms",
        "pred",
        "probs",
        "label_mass",
        "usage",
        "finish_reason",
        "format_error",
        "error",
    }
    for row in rows:
        assert expected <= set(row)
        Record.model_validate(row)  # pydantic contract


def test_same_server_id_for_a_and_b(splits, collections, tmp_path):
    items = splits["test"][:3]
    raw_dir = tmp_path / "raw-same"
    truth = FakeTruth({state_text(b): list(b.labels) for b in items})

    async def _run() -> None:
        client = make_client(truth)
        try:
            arms = {
                "A": make_arm("A", "systemone", "fake/model", collections),
                "B": make_arm("B", "systemone", "fake/model", collections),
            }
            await run_bench(
                arms,
                client,
                items,
                "same-srv",
                raw_dir,
                reps=1,
                meta={"model_id": "fake/model", "server_id": "srv-42", "arm_b_impl": "systemone"},
            )
        finally:
            await client.aclose()

    asyncio.run(_run())
    ids = {
        json.loads(x)["server_id"]
        for arm in ("A", "B")
        for x in (raw_dir / f"{arm}.jsonl").read_text().splitlines()
    }
    assert ids == {"srv-42"}
