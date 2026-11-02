"""Arm C: multilingual-e5-small fine-tuned for multi-label routing.

Training lives in scripts/train_classifier.py; this module covers model
building, scoring and latency measurement. Latency is measured in-process
(batch 1, GPU with torch.cuda.synchronize when available, no HTTP) — the
README notes this difference against arms A/B.
"""

from __future__ import annotations

import time

from crb.prompts import state_text
from crb.schema import Block, Collection

MAX_TOKENS = 512
THRESHOLD = 0.5


def id2label(collections: list[Collection]) -> dict[int, str]:
    return {i: c.name for i, c in enumerate(collections)}


def label2id(collections: list[Collection]) -> dict[str, int]:
    return {c.name: i for i, c in enumerate(collections)}


def build_model(model_name: str, collections: list[Collection]):
    """Load a sequence-classification head with 11 multi-label outputs."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(collections),
        problem_type="multi_label_classification",
        id2label=id2label(collections),
        label2id=label2id(collections),
    )
    return tokenizer, model


def encode(tokenizer, text: str):
    """E5 input format, capped at 512 tokens."""
    return tokenizer(
        f"query: {text}",
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )


def predict_labels(
    model, tokenizer, blocks: list[Block], collections: list[Collection], device=None
):
    """Return (labels per block, probability matrix) with sigmoid >= 0.5."""
    import torch

    model.eval()
    was_training = model.training
    name_by_idx = [c.id for c in collections]
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for block in blocks:
            batch = encode(tokenizer, state_text(block))
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits)[0].cpu().tolist()
            all_probs.append(probs)
            all_labels.append([name_by_idx[i] for i, p in enumerate(probs) if p >= THRESHOLD])
    if not was_training:
        model.train(False)
    return all_labels, all_probs


def measure_latency(model, tokenizer, blocks: list[Block], device=None, reps: int = 3) -> dict:
    """Batch-1 in-process latency over `reps` repetitions; returns p50/p95 in ms."""
    import torch

    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for _ in range(reps):
            for block in blocks:
                batch = encode(tokenizer, state_text(block))
                if device is not None:
                    batch = {k: v.to(device) for k, v in batch.items()}
                if device is not None and str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter_ns()
                model(**batch)
                if device is not None and str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                times.append((time.perf_counter_ns() - start) / 1e6)
    times.sort()
    n = len(times)

    def pct(p: float) -> float:
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return times[k]

    return {"n": n, "p50_ms": pct(0.5), "p95_ms": pct(0.95), "mean_ms": sum(times) / n}
