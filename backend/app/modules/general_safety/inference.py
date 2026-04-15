"""
General Safety Moderator — 2-layer inference pipeline.

Layer 1 — DistilBERT + LoRA classifier  (probabilistic intent, 7 classes)
Layer 2 — MiniLM embedding similarity   (catches adversarial paraphrases)

No regex layer — all intent detection is model-driven.

Each layer can be run standalone or through the full pipeline().
The pipeline() function is the public entrypoint used by governance_service.
"""
from __future__ import annotations

import unicodedata
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_MODULE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    ALLOW  = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK  = "BLOCK"


# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------

# L1 (classifier) — probability of the top harmful class
BLOCK_THRESHOLD  = 0.80   # high-confidence harmful → BLOCK
REVIEW_THRESHOLD = 0.50   # borderline → REVIEW

# L2 (MiniLM) — cosine similarity to known unsafe prompts
L2_BLOCK_THRESHOLD  = 0.85  # near-identical paraphrase → BLOCK  (was 0.92)
L2_REVIEW_THRESHOLD = 0.72  # close paraphrase → REVIEW          (was 0.82)

# Classes that are harmful (i.e. NOT safe)
_HARMFUL_CLASSES = {
    "self_harm",
    "violence",
    "hate_speech",
    "prompt_injection",
    "financial_crime",
    "eu_ai_act_violation",
}

# Maps classifier label → safety_scanner score key in governance_service
CLASS_TO_SCORE_KEY: dict[str, str] = {
    "self_harm":          "self_harm_score",
    "violence":           "violence_score",
    "hate_speech":        "toxicity_score",
    "prompt_injection":   "prompt_injection_score",
    "financial_crime":    "financial_crime_score",
    "eu_ai_act_violation":"eu_ai_act_score",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneralSafetyResult:
    verdict: Verdict = Verdict.ALLOW
    risk_score: float = 0.0           # 0.0 – 1.0 aggregate risk
    top_label: str = "safe"           # highest-probability class
    top_label_prob: float = 0.0       # probability of top label
    class_probs: dict[str, float] = field(default_factory=dict)  # all class probabilities
    score_map: dict[str, float] = field(default_factory=dict)    # score_key → score
    layer_decisions: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input normalisation — matches DPDP normaliser for consistency
# ---------------------------------------------------------------------------

_JAILBREAK_SURFACE = re.compile(
    r"\b(ignore (previous|all|your) instructions?|"
    r"pretend (you are|to be)|"
    r"you are now (an? )?unrestricted|"
    r"DAN mode|developer mode|"
    r"override (safety|content|filter))\b",
    re.IGNORECASE,
)

def normalise(text: str) -> str:
    """Lowercase, unicode-normalise, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Layer 1 — DistilBERT + LoRA classifier
# ---------------------------------------------------------------------------

class GeneralSafetyClassifier:
    """
    Lazy-loaded DistilBERT + LoRA classifier for general safety.
    Falls back to a uniform prior before the model is trained so the
    pipeline degrades gracefully (all verdicts → ALLOW, no false positives).
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._model_dir = Path(model_dir or (_MODULE_DIR / "model_output"))
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._num_labels: int = 7

    def _load(self) -> None:
        if self._loaded:
            return

        if not self._model_dir.exists():
            print(
                f"[GeneralSafetyClassifier] model_dir '{self._model_dir}' not found. "
                "Run train.py first. Using uniform prior — all verdicts will be ALLOW."
            )
            self._loaded = True
            return

        import json as _json
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        from peft import PeftModel, PeftConfig

        print(f"[GeneralSafetyClassifier] Loading from {self._model_dir} …")
        config = PeftConfig.from_pretrained(str(self._model_dir))

        label_map_path = self._model_dir / "label_map.json"
        if label_map_path.exists():
            label_map = _json.loads(label_map_path.read_text(encoding="utf-8"))
            id2label = {int(k): v for k, v in label_map["id2label"].items()}
            label2id = label_map["label2id"]
            num_labels = len(id2label)
        else:
            try:
                from app.modules.general_safety.dataset import ID2LABEL, LABEL2ID, LABELS
            except ImportError:
                from dataset import ID2LABEL, LABEL2ID, LABELS  # type: ignore[no-redef]
            id2label, label2id, num_labels = ID2LABEL, LABEL2ID, len(LABELS)

        # Schema alignment check
        try:
            from app.modules.general_safety.dataset import LABELS as _CURRENT
        except ImportError:
            from dataset import LABELS as _CURRENT  # type: ignore[no-redef]

        checkpoint_labels = [id2label[i] for i in range(len(id2label))]
        if checkpoint_labels != list(_CURRENT):
            import warnings
            msg = (
                f"[GeneralSafetyClassifier] SCHEMA MISMATCH — checkpoint labels "
                f"do not match current LABELS in dataset.py.\n"
                f"  checkpoint: {checkpoint_labels}\n"
                f"  expected  : {list(_CURRENT)}\n"
                f"  ACTION: retrain with current label set. Falling back to ALLOW."
            )
            print(msg)
            warnings.warn(msg, stacklevel=2)
            self._loaded = True
            return

        base = DistilBertForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            low_cpu_mem_usage=False,
            attn_implementation="eager",  # Avoid SDPA/_refs meta-dispatch on torch 2.2
        )
        # NOTE: merge_and_unload() triggers torch 2.2 meta-dispatch when combined
        # with PEFT 0.17 — causes "Tensor on device cpu is not on expected device meta".
        # Using PeftModel directly in eval mode is functionally identical for inference.
        self._model = PeftModel.from_pretrained(base, str(self._model_dir), is_trainable=False)
        self._model.eval()
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(str(self._model_dir))
        self._num_labels = self._model.config.num_labels
        self._loaded = True
        print(f"[GeneralSafetyClassifier] Loaded — {len(_CURRENT)} classes.")

    _MAX_TOKENS = 254
    _STRIDE     = 127

    def predict(self, text: str) -> np.ndarray:
        """
        Returns probability vector of shape (num_labels,).
        For long texts, uses a sliding window and averages per-window probs.
        Falls back to uniform prior when model is not loaded.
        """
        self._load()
        if self._model is None:
            return np.ones(self._num_labels, dtype=np.float32) / self._num_labels

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        input_ids = enc["input_ids"][0]
        total_tokens = len(input_ids)

        if total_tokens <= self._MAX_TOKENS + 2:
            enc_trunc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self._MAX_TOKENS + 2,
            )
            with torch.no_grad():
                logits = self._model(**enc_trunc).logits
            return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Sliding window for long texts
        window_probs: list[np.ndarray] = []
        for start in range(0, total_tokens, self._STRIDE):
            chunk_ids = input_ids[start: start + self._MAX_TOKENS]
            chunk_enc = {
                "input_ids": chunk_ids.unsqueeze(0),
                "attention_mask": torch.ones(1, len(chunk_ids), dtype=torch.long),
            }
            with torch.no_grad():
                logits = self._model(**chunk_enc).logits
            window_probs.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())
            if start + self._MAX_TOKENS >= total_tokens:
                break

        return np.mean(window_probs, axis=0)


# ---------------------------------------------------------------------------
# Layer 2 — MiniLM embedding similarity
# ---------------------------------------------------------------------------

class EmbeddingSimilarityGate:
    """
    Lazy-loaded MiniLM-L6 sentence encoder.
    Computes cosine similarity against KNOWN_UNSAFE_PROMPTS.
    Falls back gracefully (similarity 0.0) when sentence-transformers is absent.
    """

    def __init__(self) -> None:
        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            try:
                from app.modules.general_safety.dataset import KNOWN_UNSAFE_PROMPTS
            except ImportError:
                from dataset import KNOWN_UNSAFE_PROMPTS  # type: ignore[no-redef]
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embeddings = self._model.encode(
                KNOWN_UNSAFE_PROMPTS, normalize_embeddings=True, show_progress_bar=False
            )
        except ImportError:
            print(
                "[EmbeddingSimilarityGate] sentence-transformers not available. "
                "Layer 2 similarity check disabled."
            )
        self._loaded = True

    def max_similarity(self, text: str) -> float:
        """Returns cosine similarity against the closest known unsafe prompt."""
        self._load()
        if self._model is None or self._embeddings is None:
            return 0.0
        query_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        sims = (self._embeddings @ query_emb.T).squeeze()
        return float(np.max(sims))


# ---------------------------------------------------------------------------
# Module-level singletons (loaded lazily on first call)
# ---------------------------------------------------------------------------

_classifier   = GeneralSafetyClassifier()
_emb_gate     = EmbeddingSimilarityGate()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def pipeline(text: str) -> GeneralSafetyResult:
    """
    Runs the 2-layer general safety pipeline and returns a GeneralSafetyResult.

    Layer 1: DistilBERT+LoRA classifier → class probabilities
    Layer 2: MiniLM embedding similarity → adversarial paraphrase detection

    Verdict logic:
      - Layer 1 harmful class prob ≥ BLOCK_THRESHOLD  → BLOCK
      - Layer 1 harmful class prob ≥ REVIEW_THRESHOLD → REVIEW
      - Layer 2 similarity ≥ L2_BLOCK_THRESHOLD       → BLOCK  (override)
      - Layer 2 similarity ≥ L2_REVIEW_THRESHOLD       → REVIEW (floor)
      - Final verdict = max(L1_verdict, L2_verdict)
    """
    result = GeneralSafetyResult()

    try:
        from app.modules.general_safety.dataset import LABELS, ID2LABEL
    except ImportError:
        from dataset import LABELS, ID2LABEL  # type: ignore[no-redef]

    text_norm = normalise(text)

    # ── Layer 1: classifier ──────────────────────────────────────────────────
    probs = _classifier.predict(text_norm)
    class_probs = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    result.class_probs = class_probs

    # Top label
    top_idx  = int(np.argmax(probs))
    top_label = LABELS[top_idx]
    top_prob  = float(probs[top_idx])
    result.top_label      = top_label
    result.top_label_prob = top_prob

    # L1 verdict
    if top_label in _HARMFUL_CLASSES and top_prob >= BLOCK_THRESHOLD:
        l1_verdict = Verdict.BLOCK
        result.reasons.append(
            f"L1: {top_label} confidence {top_prob:.0%} ≥ block threshold"
        )
    elif top_label in _HARMFUL_CLASSES and top_prob >= REVIEW_THRESHOLD:
        l1_verdict = Verdict.REVIEW
        result.reasons.append(
            f"L1: {top_label} confidence {top_prob:.0%} — borderline, flagged for review"
        )
    else:
        l1_verdict = Verdict.ALLOW

    result.layer_decisions["l1_classifier"] = l1_verdict.value

    # ── Layer 2: embedding similarity ────────────────────────────────────────
    max_sim = _emb_gate.max_similarity(text_norm)
    if max_sim >= L2_BLOCK_THRESHOLD:
        l2_verdict = Verdict.BLOCK
        result.reasons.append(
            f"L2: near-identical to known unsafe prompt (similarity {max_sim:.2f})"
        )
    elif max_sim >= L2_REVIEW_THRESHOLD:
        l2_verdict = Verdict.REVIEW
        result.reasons.append(
            f"L2: close to known unsafe prompt (similarity {max_sim:.2f})"
        )
    else:
        l2_verdict = Verdict.ALLOW

    result.layer_decisions["l2_embedding"] = l2_verdict.value

    # ── Final verdict: take the highest severity ──────────────────────────────
    _priority = {Verdict.ALLOW: 0, Verdict.REVIEW: 1, Verdict.BLOCK: 2}
    final = max(l1_verdict, l2_verdict, key=lambda v: _priority[v])
    result.verdict = final

    # ── Risk score: probability of the top harmful class (or similarity floor) ─
    harmful_prob = top_prob if top_label in _HARMFUL_CLASSES else max(
        float(probs[i]) for i, lbl in enumerate(LABELS) if lbl in _HARMFUL_CLASSES
    )
    result.risk_score = round(max(harmful_prob, max_sim * 0.7), 4)

    # ── Score map: individual scores for each safety category ────────────────
    score_map: dict[str, float] = {}
    for label, score_key in CLASS_TO_SCORE_KEY.items():
        prob = class_probs.get(label, 0.0)
        # Boost score if this is the top harmful class and L2 is high
        if label == top_label and top_label in _HARMFUL_CLASSES:
            score = max(prob, max_sim * 0.9)
        else:
            score = prob
        if score > 0.1:  # only propagate meaningful signals
            score_map[score_key] = round(score, 4)
    result.score_map = score_map

    # toxicity rolls up hate_speech + violence + self_harm
    result.score_map["toxicity_score"] = round(max(
        score_map.get("toxicity_score", 0.0),
        score_map.get("violence_score", 0.0),
        score_map.get("self_harm_score", 0.0),
    ), 4)

    # prompt_injection_score is an alias for injection_score in the policy engine
    if "prompt_injection_score" in score_map:
        result.score_map["injection_score"] = score_map["prompt_injection_score"]

    return result


# ---------------------------------------------------------------------------
# Async wrapper — mirrors DPDP moderator interface
# ---------------------------------------------------------------------------

import asyncio
import concurrent.futures

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="gs_moderator")


async def moderate(text: str) -> GeneralSafetyResult:
    """
    Async entry point — runs the CPU-bound pipeline in a thread pool
    so the FastAPI event loop is not blocked.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, pipeline, text)


def warmup() -> None:
    """
    Eagerly load both models so the first real request does not block.
    Call once at application startup (in the FastAPI lifespan handler).
    """
    _classifier._load()
    _emb_gate._load()
    print("[GeneralSafety] Warmup complete.")
