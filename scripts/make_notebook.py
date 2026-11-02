"""Build notebooks/run_all.ipynb (committed without outputs). Run from repo root:

python scripts/make_notebook.py
"""

from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        """# collection-router-bench — Run All

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Grazinascito/collection-router-bench/blob/main/notebooks/run_all.ipynb)

Before Run All:
1. **Runtime → Change runtime type → T4 GPU** (the notebook stops if no GPU).
2. Add a Colab secret named `HF_TOKEN` (write permission) and enable notebook
   access — it publishes the dataset, model and Space. Or set `PUBLISH = False`.

Everything else installs and runs in order (~50 min)."""
    ),
    code(
        '''MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
FALLBACK_MODEL_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
VLLM_VERSION = "0.32.0"
REPS = 3
RUN_THROUGHPUT = True
RUN_SECOND_MODEL = False
PUBLISH = True
REPO_URL = "https://github.com/Grazinascito/collection-router-bench"
REPO_REF = "main"'''
    ),
    code(
        """import subprocess

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    name, cc, total = [x.strip() for x in out.split(",")]
    print(f"GPU: {name} | compute capability {cc} | {total}")
except Exception as exc:
    raise RuntimeError(
        "No GPU visible. Runtime > Change runtime type > T4 GPU, then Run All again."
    ) from exc"""
    ),
    code(
        """import os


def _get_token():
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


HF_TOKEN = _get_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    from huggingface_hub import HfApi

    print("HF user:", HfApi().whoami()["name"])
elif PUBLISH:
    raise RuntimeError(
        "HF_TOKEN secret missing (Colab: Secrets > HF_TOKEN, enable notebook access), "
        "or set PUBLISH = False."
    )
else:
    print("HF_TOKEN missing; continuing without publishing")"""
    ),
    code(
        """import subprocess

REPO_DIR = "/content/collection-router-bench"
subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO_DIR], check=True)
COMMIT = subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
print("commit:", COMMIT)"""
    ),
    code(
        """import subprocess

subprocess.run(["pip", "install", "-q", "uv"], check=True)
subprocess.run(
    ["uv", "pip", "install", "--system", "-e", ".[gpu,train]", "--torch-backend=auto"], check=True
)
subprocess.run(["vllm", "download-kernels"], check=True)

import torch
import transformers
import vllm

print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__)
assert vllm.__version__ == VLLM_VERSION"""
    ),
    code(
        """import asyncio
import sys
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, "src")

from crb.data import load_collections, load_split
from crb.server import ServerHandle, model_sha
from crb.smoke import run_smoke

collections = load_collections("data/collections.yaml")
splits = load_split("data/train.jsonl", "data/test.jsonl")
RUN_ID = "colab-" + datetime.now().strftime("%Y%m%d-%H%M")
RESULTS_PATH = Path("results") / RUN_ID

run_meta = {"run_id": RUN_ID, "repo_commit": COMMIT, "smoke_attempts": []}
server: ServerHandle | None = None
client = httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=60.0)


async def client_for(model_id: str) -> httpx.AsyncClient:
    global server
    if server is None or server.model_id != model_id:
        if server is not None:
            server.stop()
        server = ServerHandle(model_id, log_dir=RESULTS_PATH)
        run_meta["server_id"] = server.start()
        assert server.wait_healthy(), "vLLM server did not become healthy in 15 min"
    return client


async def _smoke():
    return await run_smoke(
        client_for, splits["train"], collections, MODEL_ID, FALLBACK_MODEL_ID, RUN_ID, run_meta
    )


CHOSEN_MODEL, ARM_B_IMPL = asyncio.run(_smoke())
run_meta.update(model_id=CHOSEN_MODEL, model_sha=model_sha(CHOSEN_MODEL), arm_b_impl=ARM_B_IMPL)"""
    ),
    code(
        """import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "src")

from tqdm.auto import tqdm

from crb.bench import make_arm, run_bench

arms = {
    "A": make_arm("A", ARM_B_IMPL, CHOSEN_MODEL, collections),
    "B": make_arm("B", ARM_B_IMPL, CHOSEN_MODEL, collections),
}
total = REPS * 2 * len(splits["test"])
with tqdm(total=total, desc="A/B bench", unit="req") as bar:
    written = asyncio.run(
        run_bench(
            arms, client, splits["test"], RUN_ID, RESULTS_PATH / "raw",
            reps=REPS, meta=run_meta, progress=bar.update,
        )
    )
print(f"{written} new records (ETA above was for the full pass)")"""
    ),
    code(
        """import asyncio
import sys

sys.path.insert(0, "src")

from crb.bench import run_throughput

if RUN_THROUGHPUT:
    rows = []
    for conc in (8, 32):
        res = asyncio.run(run_throughput(arms, client, splits["test"], RUN_ID, conc))
        for arm_name, row in res.items():
            rows.append(dict(row, arm=arm_name))
            print(arm_name, row)
    run_meta["throughput"] = rows
else:
    print("RUN_THROUGHPUT = False; skipping")"""
    ),
    code(
        """import asyncio
import sys

sys.path.insert(0, "src")

from crb.server import dump_run_meta, gpu_used_mb, run_meta_environment

asyncio.run(client.aclose())
server.stop()
run_meta["environment"] = {"gpu": None, **run_meta_environment()}
from crb.server import gpu_info

run_meta["environment"]["gpu"] = gpu_info()
dump_run_meta(RESULTS_PATH / "run_meta.json", **run_meta)
used = gpu_used_mb()
print(f"GPU used after stop: {used:.0f} MiB" if used is not None else "nvidia-smi unavailable")"""
    ),
    code(
        """if RUN_SECOND_MODEL:
    print(
        "Second model protocol: set MODEL_ID = FALLBACK_MODEL_ID and RUN_ID will "
        "change automatically; then Run All again. The smoke swap order already "
        "falls back automatically when a model cannot serve /v1/systemone."
    )
else:
    print("RUN_SECOND_MODEL = False; skipping")"""
    ),
    code(
        """import shutil
import subprocess
import sys
from pathlib import Path

rc = subprocess.run([sys.executable, "scripts/train_classifier.py"], check=False).returncode
if rc != 0:
    raise SystemExit("arm C training failed; see the log above")
shutil.copy(
    Path("models/collection-router-e5-small/classifier_metrics.json"),
    RESULTS_PATH / "classifier_metrics.json",
)
print("arm C done")"""
    ),
    code(
        """import subprocess
import sys

subprocess.run([sys.executable, "scripts/make_report.py", "--run-id", RUN_ID], check=True)

from IPython.display import Image, Markdown
from pathlib import Path

display(Markdown(Path(f"results/{RUN_ID}/report.md").read_text()))
display(Image(filename=f"results/{RUN_ID}/figures/f1_vs_latency.png"))"""
    ),
    code(
        """import shutil
import subprocess
import sys
from pathlib import Path

if PUBLISH:
    rc = subprocess.run(
        [sys.executable, "scripts/publish_hf.py", "--run-id", RUN_ID], check=False
    ).returncode
    if rc != 0:
        shutil.make_archive(f"/content/results-{RUN_ID}", "zip", f"results/{RUN_ID}")
        print("Publish failed (R10): results.zip written to /content for manual download.")
else:
    shutil.make_archive(f"/content/results-{RUN_ID}", "zip", f"results/{RUN_ID}")
    print("PUBLISH = False: results.zip written to /content.")"""
    ),
    code(
        """from huggingface_hub import HfApi

user = HfApi().whoami()["name"]
print("GitHub:  https://github.com/Grazinascito/collection-router-bench")
print(f"Dataset: https://huggingface.co/datasets/{user}/collection-router-bench")
print(f"Model:   https://huggingface.co/{user}/collection-router-e5-small")
print(f"Space:   https://huggingface.co/spaces/{user}/collection-router-bench")"""
    ),
]

nb["cells"] = cells
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
)

path = "notebooks/run_all.ipynb"
nbf.write(nb, path)
print(f"wrote {path} with {len(cells)} cells")
