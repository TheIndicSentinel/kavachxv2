"""
Training script for the General Safety intent classifier.

Architecture:
  DistilBERT (base, uncased) + LoRA via PEFT → 7-class sequence classification
  CPU-friendly, mirrors the DPDP moderator training pipeline.

Usage:
  cd backend/app/modules/general_safety
  python train.py                          # default 4 epochs
  python train.py --epochs 6 --batch-size 8
  python train.py --from-scratch           # ignore existing adapter, retrain fully

Output:
  model_output/    — adapter weights, tokenizer, label_map.json, val_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

from dataset import LABEL2ID, ID2LABEL, LABELS, generate_dataset

_MODULE_DIR = Path(__file__).parent
_OUTPUT_DIR = _MODULE_DIR / "model_output"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class GeneralSafetyDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
    bias="none",
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 3e-4,
    output_dir: Path = _OUTPUT_DIR,
    from_scratch: bool = False,
    val_split: float = 0.15,
    seed: int = 42,
) -> None:
    set_seed(seed)

    if from_scratch and output_dir.exists():
        print(f"[train] --from-scratch: removing {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    texts, labels = generate_dataset(seed=seed)
    print(f"[train] Dataset: {len(texts)} samples across {len(LABELS)} classes.")

    class_counts = {lbl: 0 for lbl in LABELS}
    for lbl_id in labels:
        class_counts[ID2LABEL[lbl_id]] += 1
    for lbl, cnt in class_counts.items():
        print(f"  {lbl:<25} {cnt}")

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=val_split, stratify=labels, random_state=seed
    )
    print(f"[train] Train={len(X_train)} | Val={len(X_val)}")

    # ── Tokeniser ────────────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_ds = GeneralSafetyDataset(X_train, y_train, tokenizer)
    val_ds   = GeneralSafetyDataset(X_val,   y_val,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    # ── Model + LoRA ─────────────────────────────────────────────────────────
    num_labels = len(LABELS)
    base = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        low_cpu_mem_usage=False,
    )
    model = get_peft_model(base, LORA_CONFIG)
    model.print_trainable_parameters()

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    device = torch.device("cpu")
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch["labels"].cpu().numpy())

        acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
        print(f"[epoch {epoch}/{epochs}] loss={avg_loss:.4f} | val_acc={acc:.3f}")

        if acc > best_val_acc:
            best_val_acc = acc
            # Save adapter weights (LoRA only — not full model)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

    # ── Final validation report ──────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch["labels"].cpu().numpy())

    report = classification_report(
        all_true, all_preds,
        target_names=LABELS,
        output_dict=True,
    )
    print("\n" + classification_report(all_true, all_preds, target_names=LABELS))

    # ── Save metadata ────────────────────────────────────────────────────────
    label_map = {"id2label": {str(i): lbl for i, lbl in ID2LABEL.items()}, "label2id": LABEL2ID}
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    (output_dir / "val_metrics.json").write_text(
        json.dumps({"best_val_accuracy": best_val_acc, "report": report}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_meta.json").write_text(
        json.dumps({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "val_split": val_split, "seed": seed,
            "num_labels": num_labels, "labels": LABELS,
            "dataset_size": len(texts),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n[train] Done. Best val accuracy: {best_val_acc:.3f}")
    print(f"[train] Model saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train General Safety classifier")
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--output-dir",  type=str,   default=str(_OUTPUT_DIR))
    parser.add_argument("--val-split",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--from-scratch",action="store_true",
                        help="Delete existing model_output and retrain from scratch")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=Path(args.output_dir),
        from_scratch=args.from_scratch,
        val_split=args.val_split,
        seed=args.seed,
    )
