"""The notebook runs with Run All: no outputs, no secrets, ordered cells."""

from __future__ import annotations

import re

import nbformat

NOTEBOOK = nbformat.read("notebooks/run_all.ipynb", as_version=4)

SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def test_no_outputs():
    for cell in NOTEBOOK.cells:
        if cell.cell_type == "code":
            assert not cell.get("outputs"), f"cell has outputs: {cell.source[:50]}"
            assert not cell.get("execution_count"), "cell was executed before commit"


def test_parameter_names():
    first_code = next(c for c in NOTEBOOK.cells if c.cell_type == "code")
    for name in (
        "MODEL_ID",
        "FALLBACK_MODEL_ID",
        "VLLM_VERSION",
        "REPS",
        "RUN_THROUGHPUT",
        "RUN_SECOND_MODEL",
        "PUBLISH",
        "REPO_URL",
        "REPO_REF",
    ):
        assert name in first_code.source, f"parameter {name} not in the first code cell"


def test_no_secrets():
    for cell in NOTEBOOK.cells:
        source = cell.source
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(source), f"secret-like string in cell: {source[:60]}"


def test_cell_order():
    sources = [c.source for c in NOTEBOOK.cells if c.cell_type == "code"]
    groups = [
        ["nvidia-smi"],
        ["HF_TOKEN"],
        ["clone"],
        ["uv", "pip", "install"],
        ["run_smoke"],
        ["run_bench"],
        ["run_throughput"],
        ["server.stop"],
        ["train_classifier"],
        ["make_report"],
        ["publish_hf"],
        ["whoami"],
    ]
    cursor = 0
    for group in groups:
        while cursor < len(sources) and not all(marker in sources[cursor] for marker in group):
            cursor += 1
        assert cursor < len(sources), f"marker group {group} missing or out of order"
