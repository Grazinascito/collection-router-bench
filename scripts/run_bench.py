"""Run the A/B benchmark (real server or --fake), with optional throughput.

Examples:
  python scripts/run_bench.py --fake --limit 10
  python scripts/run_bench.py --model Qwen/Qwen3-4B-Instruct-2507 --reps 3 --concurrency 8,32
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crb import fake as fake_mod
from crb.bench import make_arm, run_bench, run_throughput
from crb.data import load_collections, load_split
from crb.prompts import state_text
from crb.server import ServerHandle, dump_run_meta, gpu_info, model_sha, run_meta_environment
from crb.smoke import run_smoke

DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
FALLBACK_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


async def main_async(args: argparse.Namespace) -> str:
    collections = load_collections(DATA_DIR / "collections.yaml")
    splits = load_split(DATA_DIR / "train.jsonl", DATA_DIR / "test.jsonl")
    test_items = splits["test"][: args.limit] if args.limit else splits["test"]

    run_id = args.run_id or datetime.now().strftime(
        "fake-%Y%m%d-%H%M%S" if args.fake else "%Y%m%d-%H%M%S"
    )
    run_dir = RESULTS_DIR / run_id
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    run_meta: dict = {
        "run_id": run_id,
        "started_at": started_at,
        "repo_commit": git_commit(),
        "reps": args.reps,
        "environment": {"gpu": gpu_info(), "host": "colab" if _in_colab() else "local"},
        "smoke_attempts": [],
    }

    server: ServerHandle | None = None
    client = None
    try:
        if args.fake:
            truth = fake_mod.FakeTruth({state_text(b): list(b.labels) for b in test_items})
            client = fake_mod.make_client(truth)
            run_meta.update(
                model_id="fake/model",
                model_sha=None,
                server_id="fake-server",
                arm_b_impl="systemone",
            )
        else:
            model_id = args.model
            server = ServerHandle(model_id, port=args.port, log_dir=run_dir)
            run_meta["server_id"] = server.start()
            if not server.wait_healthy():
                raise RuntimeError("vllm server did not become healthy in 15 min")
            import httpx

            client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=60.0)
            run_meta.update(model_id=model_id, model_sha=model_sha(model_id))

            async def client_for(mid: str):  # noqa: ANN202
                return client

            model_id, arm_b_impl = await run_smoke(
                client_for,
                splits["train"],
                collections,
                model_id,
                args.fallback_model,
                run_id,
                run_meta,
            )
            run_meta.update(model_id=model_id, arm_b_impl=arm_b_impl)

        arms = {
            "A": make_arm(
                "A", run_meta.get("arm_b_impl", "systemone"), run_meta["model_id"], collections
            ),
            "B": make_arm(
                "B", run_meta.get("arm_b_impl", "systemone"), run_meta["model_id"], collections
            ),
        }

        counter = {"n": 0}

        def progress() -> None:
            counter["n"] += 1
            if counter["n"] % 50 == 0:
                print(f"  {counter['n']} requests done", flush=True)

        written = await run_bench(
            arms,
            client,
            test_items,
            run_id,
            raw_dir,
            reps=args.reps,
            meta=run_meta,
            progress=progress,
        )
        print(f"bench wrote {written} new records to {raw_dir}")

        if args.concurrency and not args.fake:
            throughput_rows = []
            for conc in [int(x) for x in args.concurrency.split(",")]:
                res = await run_throughput(arms, client, test_items, run_id, conc)
                for arm_name, row in res.items():
                    row["arm"] = arm_name
                    throughput_rows.append(row)
            run_meta["throughput"] = throughput_rows
    finally:
        if client is not None:
            await client.aclose()
        if server is not None:
            server.stop()
        run_meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        if not args.fake:
            try:
                run_meta["environment"].update(run_meta_environment())
            except Exception:  # noqa: BLE001 - keep meta writable when torch/vllm absent
                pass
        dump_run_meta(run_dir / "run_meta.json", **run_meta)
    return run_id


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use the deterministic fake server")
    ap.add_argument("--limit", type=int, default=None, help="benchmark only the first N test items")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fallback-model", default=FALLBACK_MODEL)
    ap.add_argument(
        "--concurrency", default=None, help="comma-separated levels, e.g. 8,32 (real runs only)"
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    run_id = asyncio.run(main_async(args))
    print(f"run {run_id} complete; now run: python scripts/make_report.py --run-id {run_id}")


if __name__ == "__main__":
    main()
