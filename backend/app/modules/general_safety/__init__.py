"""
KavachX General Safety Moderator
=================================
Phase 2 of the model-first safety architecture.

Handles all general-purpose harm categories that were previously covered
by regex (and retired):
  - self_harm          : suicide and self-injury intent
  - violence           : weapons, terrorism, harm to others
  - hate_speech        : toxicity, slurs, harassment, incitement
  - prompt_injection   : jailbreak and instruction-override attempts
  - financial_crime    : AML, money laundering, hawala, tax evasion
  - eu_ai_act_violation: prohibited AI practices (biometric categorisation,
                         social scoring, real-time facial ID, emotion recognition)

Architecture: 2-layer pipeline
  L1 — DistilBERT + LoRA classifier (probabilistic intent, 7 classes)
  L2 — MiniLM embedding similarity  (catches adversarial paraphrases)

No regex layer — intent nuance is entirely model-driven.
"""
