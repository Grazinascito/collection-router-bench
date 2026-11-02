"""Cards produced by publish_hf --dry-run parse as YAML and carry the configs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _front_matter(text: str) -> dict:
    assert text.startswith("---"), "card must start with YAML front matter"
    end = text.index("\n---", 3)
    return yaml.safe_load(text[3:end])


def test_dry_run_cards_parse(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/publish_hf.py", "--dry-run", "--run-id", "fake"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PATH": __import__("os").environ["PATH"], "HOME": __import__("os").environ["HOME"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    for name, path in (
        ("dataset", REPO / "cards_out" / "dataset" / "README.md"),
        ("model", REPO / "cards_out" / "model" / "README.md"),
        ("space", REPO / "cards_out" / "space" / "README.md"),
    ):
        assert path.exists(), f"{name} card missing"
        data = _front_matter(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, f"{name} front matter empty"


def test_dataset_card_has_configs():
    sys.path.insert(0, str(REPO / "scripts"))
    import publish_hf

    card = publish_hf.dataset_card("someuser", None)
    data = _front_matter(card)
    configs = data["configs"]
    names = [c["config_name"] for c in configs]
    assert names == ["train", "validation", "test"]
    for c in configs:
        assert c["data_files"][0]["split"] == c["config_name"]
        assert c["data_files"][0]["path"] == f"data/{c['config_name']}.jsonl"
    assert data["license"] == "cc-by-4.0"
    assert data["task_ids"] == ["multi-label-classification"]
