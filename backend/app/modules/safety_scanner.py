"""
KavachX Safety Scanner
======================
Architecture: model-first, regex-free.

All intent-based detection is delegated to the fine-tuned model pipeline
(DistilBERT + LoRA → MiniLM). Regex cannot reliably distinguish intent
("what is hawala?" vs "set up a hawala network") and does not scale —
every new edge case requires a manual pattern update.

This module maintains the scanner interface expected by governance_service
so that downstream code requires no changes as model coverage expands.

Expansion roadmap
-----------------
Phase 1 (active) : DPDP privacy violations — DistilBERT + LoRA (8 classes)
Phase 2           : General safety — self_harm, violence, hate_speech,
                    prompt_injection
Phase 3           : Financial crime — financial_crime intent detection
Phase 4           : EU AI Act Art. 5 prohibited practices + PII harvesting

When a phase is ready:
  1. Add the new class(es) to the model training pipeline (dataset.py)
  2. Retrain and evaluate against gold_eval.py
  3. Map the new class verdict into the relevant score field below
  4. Enable the corresponding policy in policy_engine.py

Training data reference
-----------------------
All retired regex patterns are preserved in safety_training_data.py,
organised by category and annotated with target model class mappings.
Use them to generate seed examples when writing new training data.
"""

from typing import Dict


# Canonical score keys returned by this scanner.
# All downstream consumers (governance_service, policy_engine) rely on these keys.
# A score of 0.0 means "no signal from scanner" — the model verdict is the
# authoritative source once Phase 2+ training is complete.
_ZERO_SCORES: Dict[str, float] = {
    "financial_crime_score":  0.0,  # Phase 3 — pending model training
    "self_harm_score":        0.0,  # Phase 2 — pending model training
    "violence_score":         0.0,  # Phase 2 — pending model training
    "toxicity_score":         0.0,  # Phase 2 — pending model training (rolls up above)
    "injection_score":        0.0,  # Phase 2 — pending model training
    "prompt_injection_score": 0.0,  # alias for injection_score
    "pii_harvesting_score":   0.0,  # Phase 1 partial via DPDP moderator / pii_scanner
    "eu_ai_act_score":        0.0,  # Phase 4 — pending model training
}


class SafetyScanner:
    """
    Intent-detection scanner — model-delegated.

    scan() and analyze_exchange() maintain their original signatures so
    governance_service requires no changes. All scores are 0.0 until the
    corresponding model phase is trained and wired in.

    To plug in a model for a category:
        score = model.predict(text, category="self_harm")
        results["self_harm_score"] = score
        results["toxicity_score"]  = max(results["toxicity_score"], score)
    """

    def scan(self, text: str) -> Dict[str, float]:
        """
        Returns safety scores for the given text.
        Currently all zeros — detection delegated to fine-tuned model.
        See _ZERO_SCORES for the full score schema and phase roadmap.
        """
        return dict(_ZERO_SCORES)

    def analyze_exchange(self, input_text: str, prediction_text: str) -> Dict[str, float]:
        """
        Scores a full AI exchange (prompt + response).
        Returns the element-wise maximum across both sides.
        Currently passes through zeros — model pipeline provides intent signals.
        """
        input_scores = self.scan(input_text)
        pred_scores  = self.scan(prediction_text)
        return {key: max(input_scores[key], pred_scores[key]) for key in input_scores}
