"""Full pipeline without GPU: bench -> metrics -> report against the fake server."""

from __future__ import annotations

import asyncio
import json

from crb.bench import make_arm, run_bench, run_throughput
from crb.fake import FakeTruth, make_client
from crb.prompts import state_text
from crb.report import write_all


def test_full_pipeline_fake_server(splits, collections, tmp_path):
    items = splits["test"][:10]
    truth = FakeTruth({state_text(b): list(b.labels) for b in items})

    async def _run() -> dict:
        client = make_client(truth)
        try:
            arms = {
                "A": make_arm("A", "systemone", "fake/model", collections),
                "B": make_arm("B", "systemone", "fake/model", collections),
            }
            written = await run_bench(
                arms,
                client,
                items,
                "e2e-fake",
                tmp_path / "raw",
                reps=3,
                meta={"model_id": "fake/model", "server_id": "fake", "arm_b_impl": "systemone"},
            )
            assert written == 10 * 2 * 3
            throughput = await run_throughput(arms, client, items, "e2e-fake", 4)
            return {"throughput": throughput}
        finally:
            await client.aclose()

    extra = asyncio.run(_run())
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "e2e-fake",
                "model_id": "fake/model",
                "arm_b_impl": "systemone",
                "server_id": "fake",
                "throughput": [dict(row, arm=arm) for arm, row in extra["throughput"].items()],
            }
        ),
        encoding="utf-8",
    )

    summary = write_all(tmp_path)
    # 5 kinds of outputs
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "facts.md").exists()
    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "figures" / "f1_vs_latency.png").exists()
    assert (tmp_path / "figures" / "throughput.png").exists()  # throughput ran
    assert summary["arms"]["A"]["quality"]["n"] == 10
    assert 0.0 < summary["arms"]["A"]["quality"]["micro_f1"] <= 1.0
    assert (
        summary["comparison"]["d_micro_f1_ci95"][0] <= summary["comparison"]["d_micro_f1_ci95"][1]
    )
