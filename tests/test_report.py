"""Report outputs exist, carry the summary keys, and derive only from summary."""

from __future__ import annotations

import asyncio
import json

from crb.bench import make_arm, run_bench
from crb.fake import FakeTruth, make_client
from crb.prompts import state_text
from crb.report import compute, report_md, write_all


def _fake_run(splits, collections, tmp_path, reps=2):
    items = splits["test"][:6]
    raw_dir = tmp_path / "raw"
    truth = FakeTruth({state_text(b): list(b.labels) for b in items})

    async def _run():
        client = make_client(truth)
        try:
            arms = {
                "A": make_arm("A", "systemone", "fake/model", collections),
                "B": make_arm("B", "systemone", "fake/model", collections),
            }
            await run_bench(
                arms, client, items, "unit-run", raw_dir, reps=reps, meta={"model_id": "fake/model"}
            )
        finally:
            await client.aclose()

    asyncio.run(_run())
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "unit-run",
                "model_id": "fake/model",
                "arm_b_impl": "systemone",
                "server_id": "fake",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_summary_keys(splits, collections, tmp_path):
    run_dir = _fake_run(splits, collections, tmp_path)
    summary = compute(run_dir)
    for key in ("run_id", "model_id", "arm_b_impl", "arms", "comparison"):
        assert key in summary
    for arm in ("A", "B"):
        for key in ("quality", "latency", "stability", "tokens", "slices"):
            assert key in summary["arms"][arm]
    assert summary["arms"]["A"]["quality"]["n"] == 6


def test_report_md_uses_summary(splits, collections, tmp_path):
    run_dir = _fake_run(splits, collections, tmp_path)
    summary = compute(run_dir)
    md = report_md(summary)
    for value in (
        f"{summary['arms']['A']['quality']['micro_f1']:.4f}",
        f"{summary['arms']['B']['quality']['micro_f1']:.4f}",
        f"{summary['comparison']['d_micro_f1']:+.4f}",
    ):
        assert value in md
    # numbers not present in the summary must not leak into the report
    assert "0.9999" not in md


def test_figures_created(splits, collections, tmp_path):
    run_dir = _fake_run(splits, collections, tmp_path)
    summary = write_all(run_dir)
    fig_dir = run_dir / "figures"
    assert (fig_dir / "f1_vs_latency.png").exists()
    assert (fig_dir / "latency_box.png").exists()
    assert (fig_dir / "per_collection_f1.png").exists()
    assert (run_dir / "predictions.parquet").exists()
    assert (run_dir / "facts.md").exists()
    assert "figures" in summary
