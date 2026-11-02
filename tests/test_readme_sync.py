"""README: numbers only from the summary; required sections present."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from update_readme import load_summary  # noqa: E402

README = Path(__file__).resolve().parents[1] / "README.md"

REQUIRED_SECTIONS = (
    "## Overview",
    "## Task",
    "## Method",
    "## How it works",
    "## Dataset",
    "## Results",
    "## Limitations",
    "## Reproduce",
    "## Repository layout",
    "## License",
)

REQUIRED_LIMITATIONS = (
    "50-item",
    "T4",
    "batch invariance",
    "in-process",
    "prefix-cache",
    "independently",
    "0.5",
)


def test_required_sections():
    text = README.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"missing section {section}"
    for needle in REQUIRED_LIMITATIONS:
        assert needle.lower() in text.lower(), f"missing limitation mention: {needle}"
    assert "<!-- RESULTS:START -->" in text and "<!-- RESULTS:END -->" in text
    assert "<!-- DATA:START -->" in text and "<!-- DATA:END -->" in text


def test_results_block_matches_summary():
    summary_path = Path(__file__).resolve().parents[1] / "results" / "latest" / "summary.json"
    if not summary_path.exists():
        import pytest

        pytest.skip("no results/latest/summary.json yet (runs after the Colab Run All)")
    summary = json.loads(summary_path.read_text())
    text = README.read_text(encoding="utf-8")
    block = re.search(r"<!-- RESULTS:START -->\n(.*?)<!-- RESULTS:END -->", text, re.DOTALL).group(
        1
    )
    for name in ("A", "B"):
        arm = summary["arms"].get(name)
        if arm:
            assert f"{arm['quality']['micro_f1']:.4f}" in block
            assert f"{arm['latency']['p50_ms']:.1f}" in block
    comp = summary["comparison"]
    assert f"{comp['d_micro_f1']:+.4f}" in block


def test_load_summary_downloads_from_the_logged_in_users_dataset(tmp_path, monkeypatch):
    """No local summary for the run id -> fetch from <whoami>/collection-router-bench."""
    import huggingface_hub

    def fake_download(repo_id, filename, repo_type=None):
        assert repo_id == "grazitest/collection-router-bench", (
            f"expected the whoami user, got the literal {repo_id!r}"
        )
        assert filename == "results/latest/summary.json"
        assert repo_type == "dataset"
        path = tmp_path / "summary.json"
        path.write_text(json.dumps({"run_id": "remote-run"}))
        return str(path)

    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", lambda self: {"name": "grazitest"})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    summary = load_summary("no-such-local-run")

    assert summary == {"run_id": "remote-run"}


def test_load_summary_returns_none_without_a_token(monkeypatch):
    """No local summary and no HF token (the Mac) -> None, never a crash."""
    import huggingface_hub

    def no_token(self):
        raise RuntimeError("no token")

    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", no_token)

    assert load_summary("no-such-local-run") is None
