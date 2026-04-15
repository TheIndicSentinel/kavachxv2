"""
Incremental retraining script for the DPDP intent classifier.

Workflow
--------
1. Generate a fresh synthetic base dataset (configurable size)
2. Load any human-labelled feedback from model_output/feedback.jsonl
3. Merge and balance: feedback samples are up-weighted (repeated 3×) to
   ensure corrections override the base distribution
4. Fine-tune DistilBERT+LoRA from the *existing* checkpoint (or from
   scratch if --from-scratch is passed)
5. Evaluate on held-out val + test splits (stratified 70/15/15)
6. Save the new checkpoint, per-class metrics, and training metadata
7. Exit non-zero if any class F1 falls below --min-f1-threshold (CI gate)

Usage
-----
# First training (no existing checkpoint):
  python retrain.py --epochs 3 --output-dir ./model_output

# Incremental update after collecting feedback:
  python retrain.py --epochs 2 --resume ./model_output --output-dir ./model_output

# Full retrain from scratch ignoring existing checkpoint:
  python retrain.py --from-scratch --epochs 5

# With class-weighted loss for imbalance correction:
  python retrain.py --from-scratch --epochs 4 --class-weighted

# With focal loss (hard negative mining, gamma=2):
  python retrain.py --from-scratch --epochs 4 --focal-loss

# Non-interactive CI/CD mode with F1 gate:
  python retrain.py --epochs 3 --yes --min-f1-threshold 0.72
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from dataset import LABEL2ID, ID2LABEL, LABELS, generate_dataset


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for multi-class classification.
    Reduces the loss contribution from easy examples and focuses on hard ones.

    γ=2 is the standard recommendation; α=class_weights (optional) corrects imbalance.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # per-class weights, shape (num_classes,)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        # p_t: probability of the correct class
        probs = torch.exp(log_probs)
        p_t = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - p_t) ** self.gamma
        return (focal_factor * ce_loss).mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights: w_c = total / (num_classes * count_c).
    Caps each weight at 5× to prevent extreme upweighting of tiny classes.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)
        w = total / (num_classes * count)
        weights.append(min(w, 5.0))
    return torch.tensor(weights, dtype=torch.float)


class DPDPDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
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


def load_feedback(feedback_path: Path) -> list[dict]:
    """
    Load human corrections from JSONL feedback file.
    Deduplicates by (text, label) so repeated corrections of the same prompt
    don't dominate training signal — only unique (text, correct_label) pairs
    are kept.
    """
    if not feedback_path.exists():
        return []
    seen: set[tuple[str, int]] = set()
    entries = []
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            # Only use corrections / false_positive / false_negative — not confirmations
            if e.get("feedback_type", "correction") in ("correction", "false_positive", "false_negative"):
                key = (e["text"].strip().lower(), int(e["label"]))
                if key not in seen:
                    seen.add(key)
                    entries.append({"text": e["text"], "label": e["label"], "label_name": e["label_name"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return entries


def build_dataset(
    samples_per_class: int,
    feedback: list[dict],
    feedback_weight: int = 3,
) -> tuple[list[str], list[int]]:
    """
    Merge synthetic base dataset + feedback (up-weighted).
    Returns (texts, labels).
    """
    base = generate_dataset(target_per_class=samples_per_class)
    all_samples = list(base)
    for _ in range(feedback_weight):
        all_samples.extend(feedback)
    random.shuffle(all_samples)
    texts  = [s["text"] for s in all_samples]
    labels = [s["label"] for s in all_samples]
    return texts, labels


def print_class_dist(labels: list[int]) -> None:
    counts = Counter(labels)
    print("  Class distribution:")
    for label_id, label_name in ID2LABEL.items():
        print(f"    {label_name:<22} {counts.get(label_id, 0):>4}")


def compute_per_class_metrics(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
) -> dict:
    """
    Returns per-class precision, recall, F1, false-negative rate, and false-positive rate.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    num_classes = len(labels)
    metrics = {}
    for i, label_name in enumerate(labels):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == i and p != i)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != i and p == i)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != i and p != i)
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        class_report = report.get(label_name, {})
        metrics[label_name] = {
            "precision": round(class_report.get("precision", 0.0), 4),
            "recall":    round(class_report.get("recall", 0.0), 4),
            "f1":        round(class_report.get("f1-score", 0.0), 4),
            "support":   class_report.get("support", 0),
            "fn_rate":   round(fn_rate, 4),
            "fp_rate":   round(fp_rate, 4),
        }
    metrics["macro_avg"] = {
        "f1": round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "precision": round(report.get("macro avg", {}).get("precision", 0.0), 4),
        "recall": round(report.get("macro avg", {}).get("recall", 0.0), 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main retraining function
# ---------------------------------------------------------------------------

def retrain(
    resume_dir: str | None,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    samples_per_class: int,
    feedback_weight: int,
    from_scratch: bool,
    use_focal_loss: bool,
    use_class_weighted: bool,
    min_f1_threshold: float,
) -> None:
    set_seed(42)
    device = torch.device("cpu")
    print(f"Device         : {device}")
    print(f"Epochs         : {epochs}  Batch : {batch_size}  LR : {lr}")
    print(f"Focal loss     : {use_focal_loss}  Class-weighted : {use_class_weighted}")

    # ── Feedback ─────────────────────────────────────────────────────────────
    feedback_path = Path(output_dir) / "feedback.jsonl"
    feedback = load_feedback(feedback_path)
    print(f"\nFeedback samples loaded : {len(feedback)}")
    if feedback:
        fb_dist = Counter(e["label_name"] for e in feedback)
        for label, count in fb_dist.most_common():
            print(f"  {label:<22} {count}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nGenerating synthetic dataset ({samples_per_class} per class) …")
    texts, labels = build_dataset(samples_per_class, feedback, feedback_weight)
    print(f"Total samples  : {len(texts)}")
    print_class_dist(labels)

    # Stratified 70/15/15 split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    # val is 15% of original total ≈ 15/(85) of remaining
    val_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction, random_state=42, stratify=y_tmp
    )
    print(f"\nTrain : {len(X_train)}  |  Val : {len(X_val)}  |  Test : {len(X_test)}")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tok_src = resume_dir if (resume_dir and not from_scratch) else "distilbert-base-uncased"
    print(f"Tokeniser from : {tok_src}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(tok_src)

    train_ds = DPDPDataset(X_train, y_train, tokenizer)
    val_ds   = DPDPDataset(X_val,   y_val,   tokenizer)
    test_ds  = DPDPDataset(X_test,  y_test,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ── Loss function ─────────────────────────────────────────────────────────
    class_weights = None
    if use_class_weighted or use_focal_loss:
        class_weights = compute_class_weights(y_train, len(LABELS)).to(device)
        print(f"Class weights  : {class_weights.tolist()}")

    if use_focal_loss:
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
        print("Loss           : FocalLoss(γ=2)")
    elif use_class_weighted:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Loss           : CrossEntropyLoss (class-weighted)")
    else:
        criterion = None  # use model's built-in CE

    # ── Model ─────────────────────────────────────────────────────────────────
    if resume_dir and not from_scratch:
        print(f"Resuming from checkpoint : {resume_dir}")
        try:
            from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
            peft_cfg = PeftConfig.from_pretrained(resume_dir)
            base = DistilBertForSequenceClassification.from_pretrained(
                peft_cfg.base_model_name_or_path,
                num_labels=len(LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            model = PeftModel.from_pretrained(base, resume_dir)
            model.train()
            print("LoRA checkpoint loaded for fine-tuning.")
        except Exception as e:
            print(f"  Warning: could not load checkpoint ({e}). Training from scratch.")
            resume_dir = None

    if not resume_dir or from_scratch:
        from peft import LoraConfig, TaskType, get_peft_model
        print("Initialising DistilBERT + LoRA from scratch …")
        base = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            bias="none",
            modules_to_save=["pre_classifier", "classifier"],
        )
        model = get_peft_model(base, lora_cfg)

    model.to(device)
    model.print_trainable_parameters()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epoch(s) …\n")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            if criterion is not None:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = criterion(outputs.logits, batch["labels"])
            else:
                outputs = model(**batch)
                loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{epochs}  Step {step}/{len(train_loader)}  Loss {total_loss/step:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = torch.argmax(model(**batch).logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(batch["labels"].cpu().tolist())

        acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
        print(f"\n── Validation epoch {epoch} — Accuracy: {acc:.3f} ──")
        print(classification_report(all_true, all_preds, target_names=LABELS, zero_division=0))
        best_val_acc = max(best_val_acc, acc)

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n── Final Test Set Evaluation ──")
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = torch.argmax(model(**batch).logits, dim=-1)
            test_preds.extend(preds.cpu().tolist())
            test_true.extend(batch["labels"].cpu().tolist())

    test_acc = sum(p == t for p, t in zip(test_preds, test_true)) / len(test_true)
    print(f"Test Accuracy  : {test_acc:.3f}")
    print(classification_report(test_true, test_preds, target_names=LABELS, zero_division=0))

    per_class = compute_per_class_metrics(test_true, test_preds, LABELS)

    # Print FN/FP rates
    print("\nPer-class FN / FP rates (test set):")
    print(f"  {'Class':<22} {'F1':>6}  {'FN%':>6}  {'FP%':>6}")
    for label_name in LABELS:
        m = per_class[label_name]
        print(f"  {label_name:<22} {m['f1']:>6.3f}  {m['fn_rate']:>6.3f}  {m['fp_rate']:>6.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "label_map.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}), encoding="utf-8"
    )

    # Per-class metrics JSON
    (out / "val_metrics.json").write_text(
        json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "trained_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "epochs": epochs,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "samples_per_class": samples_per_class,
        "feedback_samples": len(feedback),
        "total_train_samples": len(X_train),
        "use_focal_loss": use_focal_loss,
        "use_class_weighted": use_class_weighted,
        "resumed_from": resume_dir,
        "macro_f1": per_class["macro_avg"]["f1"],
    }
    (out / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nModel saved         → {out.resolve()}")
    print(f"Per-class metrics   → {(out / 'val_metrics.json').resolve()}")
    print(f"Best val accuracy   : {best_val_acc:.3f}  |  Test accuracy : {test_acc:.3f}")

    # Invalidate in-process singleton so it reloads on next call
    try:
        import importlib
        import app.modules.dpdp_moderator.pipeline as _pl
        _pl._SINGLETON = None
        print("Pipeline singleton reset — new model will load on next inference request.")
    except Exception:
        pass

    # ── CI gate ───────────────────────────────────────────────────────────────
    if min_f1_threshold > 0.0:
        failed_classes = [
            name for name in LABELS
            if per_class[name]["f1"] < min_f1_threshold
        ]
        if failed_classes:
            print(f"\n[CI GATE FAILED] Classes below F1={min_f1_threshold:.2f}: {failed_classes}")
            sys.exit(1)
        else:
            print(f"\n[CI GATE PASSED] All classes ≥ F1={min_f1_threshold:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain DPDP AI intent classifier")
    parser.add_argument("--resume",             default="./model_output",
                        help="Load existing LoRA checkpoint for incremental training (default: ./model_output)")
    parser.add_argument("--output-dir",         default="./model_output",
                        help="Where to write the new checkpoint")
    parser.add_argument("--from-scratch",       action="store_true",
                        help="Ignore existing checkpoint; train from base DistilBERT")
    parser.add_argument("--epochs",             type=int,   default=3)
    parser.add_argument("--batch-size",         type=int,   default=8)
    parser.add_argument("--lr",                 type=float, default=3e-4)
    parser.add_argument("--samples-per-class",  type=int,   default=150)
    parser.add_argument("--feedback-weight",    type=int,   default=3,
                        help="How many times to repeat feedback samples (default: 3)")
    parser.add_argument("--focal-loss",         action="store_true",
                        help="Use Focal Loss (γ=2) instead of standard CrossEntropyLoss")
    parser.add_argument("--class-weighted",     action="store_true",
                        help="Use inverse-frequency class weights in the loss function")
    parser.add_argument("--min-f1-threshold",   type=float, default=0.0,
                        help="Minimum per-class F1 required to pass CI gate (default: 0=disabled)")
    parser.add_argument("--yes",                action="store_true",
                        help="Skip confirmation prompts (for CI/CD)")
    args = parser.parse_args()

    feedback_path = Path(args.output_dir) / "feedback.jsonl"
    feedback_count = sum(1 for _ in open(feedback_path)) if feedback_path.exists() else 0

    print("=" * 60)
    print("DPDP AI Moderator — Retraining")
    print("=" * 60)
    print(f"  Resume from      : {args.resume}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  From scratch     : {args.from_scratch}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Focal loss       : {args.focal_loss}")
    print(f"  Class-weighted   : {args.class_weighted}")
    print(f"  Min F1 threshold : {args.min_f1_threshold}")
    print(f"  Feedback         : {feedback_count} samples")
    print("=" * 60)

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    retrain(
        resume_dir        = None if args.from_scratch else args.resume,
        output_dir        = args.output_dir,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        samples_per_class = args.samples_per_class,
        feedback_weight   = args.feedback_weight,
        from_scratch      = args.from_scratch,
        use_focal_loss    = args.focal_loss,
        use_class_weighted = args.class_weighted,
        min_f1_threshold  = args.min_f1_threshold,
    )
