"""Train arm C (intfloat/multilingual-e5-small, multi-label) and evaluate on test.

Hyperparameters (plan, Block 2): 8 epochs, lr 5e-5, batch 16, fp16 (CUDA),
weight decay 0.01, warmup 10%, seed 42; best epoch by validation micro-F1.
--smoke: 1 epoch over 32 items on CPU/MPS, for training-path bugs only.
Writes classifier_metrics.json next to --out for the report to pick up.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crb.arms.classifier import measure_latency, predict_labels
from crb.data import load_collections, load_split
from crb.prompts import state_text

DATA_DIR = REPO / "data"
BASE_MODEL = "intfloat/multilingual-e5-small"
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--out", default=None, help="model output dir (default models/collection-router-e5-small)"
    )
    ap.add_argument(
        "--metrics-out", default=None, help="classifier_metrics.json target (default models/)"
    )
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_linear_schedule_with_warmup

    from crb.metrics import exact_match, to_vectors  # noqa: F401 (to_vectors used via metrics)

    collections = load_collections(DATA_DIR / "collections.yaml")
    idx_by_coll_id = {c.id: i for i, c in enumerate(collections)}
    splits = load_split(DATA_DIR / "train.jsonl", DATA_DIR / "test.jsonl")
    train_blocks = splits["train"]
    val_blocks = splits["validation"]
    test_blocks = splits["test"]

    if args.smoke:
        train_blocks = train_blocks[:32]
        val_blocks = val_blocks[:16]
        test_blocks = test_blocks[:16]

    epochs = args.epochs if args.epochs is not None else (1 if args.smoke else 8)
    batch_size = 8 if args.smoke else 16

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    use_amp = device.type == "cuda"
    print(f"device={device} amp={use_amp} epochs={epochs} batch={batch_size}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(collections),
        problem_type="multi_label_classification",
        id2label={i: c.name for i, c in enumerate(collections)},
        label2id={c.name: i for i, c in enumerate(collections)},
    ).to(device)

    class BlockDataset(Dataset):
        def __init__(self, blocks) -> None:
            self.blocks = blocks

        def __len__(self) -> int:
            return len(self.blocks)

        def __getitem__(self, idx: int):
            block = self.blocks[idx]
            target = [0.0] * len(collections)
            for lab in block.labels:
                target[idx_by_coll_id[lab]] = 1.0
            return block, torch.tensor(target, dtype=torch.float)

    def collate(batch):
        blocks, targets = zip(*batch, strict=True)
        enc = tokenizer(
            [f"query: {state_text(b)}" for b in blocks],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        return enc, torch.stack(targets)

    def micro_f1(dataset) -> float:
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate)
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for enc, targets in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                probs = torch.sigmoid(model(**enc).logits).cpu()
                preds = (probs >= 0.5).float()
                tp += int((preds * targets).sum())
                fp += int((preds * (1 - targets)).sum())
                fn += int(((1 - preds) * targets).sum())
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom else 0.0

    train_loader = DataLoader(
        BlockDataset(train_blocks), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for enc, targets in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**enc, labels=targets).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        f1 = micro_f1(BlockDataset(val_blocks))
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"epoch {epoch}: loss={avg_loss:.4f} val_micro_f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    labels, probs = predict_labels(model, tokenizer, test_blocks, collections, device=device)
    y_true = [to_vectors(list(b.labels)) for b in test_blocks]
    y_pred = [to_vectors(p) for p in labels]
    from sklearn.metrics import f1_score

    micro = float(f1_score(np.array(y_true), np.array(y_pred), average="micro", zero_division=0))
    macro = float(f1_score(np.array(y_true), np.array(y_pred), average="macro", zero_division=0))
    em = exact_match([list(b.labels) for b in test_blocks], labels)
    latency = measure_latency(model, tokenizer, test_blocks[:50], device=device)

    out_dir = (
        Path(args.out)
        if args.out
        else (
            REPO / "models" / "smoke"
            if args.smoke
            else REPO / "models" / "collection-router-e5-small"
        )
    )
    metrics_out = (
        Path(args.metrics_out) if args.metrics_out else out_dir / "classifier_metrics.json"
    )
    if not args.smoke:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "epochs": epochs,
                "best_val_micro_f1": round(best_f1, 4),
                "micro_f1": round(micro, 4),
                "macro_f1": round(macro, 4),
                "exact_match": round(em, 4),
                "latency": latency,
                "n_test": len(test_blocks),
                "smoke": args.smoke,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"classifier metrics at {metrics_out}: micro={micro:.4f} macro={macro:.4f} em={em:.4f}")


if __name__ == "__main__":
    main()
