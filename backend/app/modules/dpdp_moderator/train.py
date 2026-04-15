"""
Training script for DPDP intent classifier.

Architecture:
  DistilBERT (base, uncased) + LoRA via PEFT → sequence classification
  Fine-tuned on synthetic DPDP dataset, CPU-only.

Usage:
  python train.py [--epochs 3] [--batch-size 8] [--output-dir ./model_output]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

from dataset import LABEL2ID, ID2LABEL, LABELS, generate_dataset

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

class DPDPDataset(Dataset):
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
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# LoRA configuration for DistilBERT
# DistilBERT attention: q_lin, k_lin, v_lin, out_lin
# ---------------------------------------------------------------------------

LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                          # rank — low rank for CPU efficiency
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
    bias="none",
    modules_to_save=["pre_classifier", "classifier"],
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-4,
    max_length: int = 128,
    output_dir: str = "./model_output",
    samples_per_class: int = 150,
) -> None:
    set_seed(42)
    device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Generating synthetic dataset …")
    raw = generate_dataset(target_per_class=samples_per_class)
    texts = [s["text"] for s in raw]
    labels = [s["label"] for s in raw]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ── Tokeniser ────────────────────────────────────────────────────────────
    print("Loading DistilBERT tokeniser …")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_ds = DPDPDataset(X_train, y_train, tokenizer, max_length)
    val_ds = DPDPDataset(X_val, y_val, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ── Model + LoRA ─────────────────────────────────────────────────────────
    print("Loading DistilBERT and applying LoRA …")
    base_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = get_peft_model(base_model, LORA_CONFIG)
    model.to(device)
    model.print_trainable_parameters()

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs …\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                avg = total_loss / step
                print(f"  Epoch {epoch}/{epochs}  Step {step}/{len(train_loader)}  Loss {avg:.4f}")

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(batch["labels"].cpu().tolist())

        report = classification_report(
            all_true, all_preds, target_names=LABELS, zero_division=0
        )
        print(f"\n── Validation (epoch {epoch}) ──\n{report}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Persist label mappings for inference
    (out / "label_map.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}), encoding="utf-8"
    )
    print(f"\nModel saved → {out.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPDP intent classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples-per-class", type=int, default=150)
    parser.add_argument("--output-dir", default="./model_output")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=128,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
    )
