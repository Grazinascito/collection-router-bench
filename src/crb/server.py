"""Start/stop a vLLM OpenAI-compatible server and check the GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

HEALTH_TIMEOUT_S = 15 * 60
GEMMA_MM_LIMITS = '{"image": 0, "audio": 0, "video": 0}'


def build_server_cmd(
    model_id: str, port: int = 8000, logprobs_mode: str = "raw_logprobs"
) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--dtype",
        "half",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.90",
        "--logprobs-mode",
        logprobs_mode,
        "--enable-prefix-caching",
        "--seed",
        "0",
        "--port",
        str(port),
    ]
    # E4B is multimodal; the benchmark is text-only, so cap every modality at 0.
    # Syntax confirmed against vLLM 0.32.0 arg_utils (docs/api_check.md, Q4).
    if "gemma" in model_id.lower():
        cmd += ["--limit-mm-per-prompt", GEMMA_MM_LIMITS]
    return cmd


def gpu_info() -> dict | None:
    """GPU name, compute capability and total memory, or None off-GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    parts = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"name": parts[0], "compute_capability": parts[1], "memory_total": parts[2]}


def gpu_used_mb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return float(out.splitlines()[0])


class ServerHandle:
    """One `vllm serve` process; A and B share it so only the decision differs."""

    def __init__(self, model_id: str, port: int = 8000, log_dir: Path | None = None) -> None:
        self.model_id = model_id
        self.port = port
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self.server_id: str | None = None
        self.started_at: datetime | None = None

    def start(self) -> str:
        self.proc = subprocess.Popen(
            build_server_cmd(self.model_id, self.port),
            stdout=open(self._log_path(), "ab"),
            stderr=subprocess.STDOUT,
        )
        self.started_at = datetime.now()
        self.server_id = f"{self.proc.pid}-{self.started_at.strftime('%Y%m%d-%H%M%S')}"
        return self.server_id

    def _log_path(self) -> Path:
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            return self.log_dir / "server.log"
        return Path(subprocess.gettempdir()) / "crb-server.log"

    def wait_healthy(self, timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self.proc.returncode}; see {self._log_path()}"
                )
            try:
                if httpx.get(url, timeout=5).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(5)
        return False

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.proc = None
        # confirm the GPU actually freed (< 1 GiB)
        used = gpu_used_mb()
        if used is not None and used > 1024:
            raise RuntimeError(f"GPU still holds {used:.0f} MiB after server stop")


def model_sha(model_id: str) -> str | None:
    """The current commit sha of the HF model, via huggingface_hub."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        return getattr(info, "sha", None)
    except Exception:  # noqa: BLE001 - offline-safe
        return None


def run_meta_environment() -> dict:
    import torch

    return {
        "gpu": gpu_info(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": _vllm_version(),
    }


def _vllm_version() -> str | None:
    try:
        import vllm

        return vllm.__version__
    except ImportError:
        return None


def dump_run_meta(path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in extra.items() if v is not None}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
