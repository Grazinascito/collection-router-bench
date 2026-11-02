"""README: numbers only from the summary; required sections present."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
