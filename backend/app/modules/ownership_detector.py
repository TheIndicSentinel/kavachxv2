"""
Ownership Detector — Post-ML rule-based layer.

Distinguishes first-person own-data requests (SELF) from third-party
data requests (OTHER) so the ML models' topic signal does not drive the
final decision alone.

Decision table
--------------
  ownership=SELF   + data topic  → ALLOW_WITH_AUTH  (user exercising own rights)
  ownership=OTHER  + data topic  → reinforce BLOCK   (third-party access attempt)
  ownership=UNKNOWN              → defer to ML verdict

Examples
--------
  SELF  : "Show my transactions"       → ALLOW_WITH_AUTH
          "Update my phone number"     → ALLOW_WITH_AUTH
          "Delete my account"          → ALLOW_WITH_AUTH
          "Download my data"           → ALLOW_WITH_AUTH

  OTHER : "Show user 9921 transactions"  → BLOCK
          "Get someone's bank data"       → BLOCK
          "Fetch account data for Rahul"  → BLOCK

  UNKNOWN: "What is transaction history?" → defer to ML

This module is intentionally rule-only — it runs AFTER the ML models
and can override their verdict when ownership signal is unambiguous.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Ownership enum
# ---------------------------------------------------------------------------

class Ownership(str, Enum):
    SELF    = "self"     # clear first-person own-data signal
    OTHER   = "other"    # clear third-party / specific target signal
    UNKNOWN = "unknown"  # no decisive signal — defer to ML


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class OwnershipResult:
    ownership:       Ownership
    confidence:      float        # 0.0 – 1.0
    matched_pattern: str          # which pattern triggered
    decision:        str          # ALLOW_WITH_AUTH | BLOCK | UNCERTAIN
    reason:          str          # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "ownership":        self.ownership.value,
            "confidence":       round(self.confidence, 2),
            "matched_pattern":  self.matched_pattern,
            "decision":         self.decision,
            "reason":           self.reason,
        }


# ---------------------------------------------------------------------------
# Data-topic vocabulary — prompts must reference personal data to be relevant
# ---------------------------------------------------------------------------

_DATA_TOPICS = re.compile(
    r"\b("
    r"transaction(s)?|account|balance|bank|statement(s)?|payment(s)?|"
    r"profile|email|phone|address|password|username|name|"
    r"data|record(s)?|information|history|detail(s)?|"
    r"medical|health|prescription|diagnosis|report(s)?|"
    r"document(s)?|file(s)?|photo(s)?|id|identity|"
    r"consent|privacy|personal|aadhaar|pan|kyc|"
    r"number|settings|preference(s)?"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# SELF patterns — first-person possessive or action on own data
# ---------------------------------------------------------------------------

_SELF_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # "my <data>" — strongest signal
    ("my_possessive",
     re.compile(
         r"\b(my|mine)\b.{0,40}\b("
         r"transaction(s)?|account|balance|bank|statement|payment(s)?|"
         r"profile|email|phone|address|password|data|record(s)?|history|"
         r"detail(s)?|information|consent|document(s)?|file(s)?|settings|"
         r"medical|health|report(s)?|preference(s)?"
         r")\b",
         re.IGNORECASE,
     ),
     0.95),

    # "show me my …", "update my …", "delete my …"
    ("verb_my",
     re.compile(
         r"\b(show|give|fetch|get|display|view|see|download|export|"
         r"update|change|edit|modify|delete|remove|close|access|"
         r"check|verify|reset|download)\b.{0,20}\bmy\b",
         re.IGNORECASE,
     ),
     0.92),

    # "I want to [see/update/delete] my …"
    ("i_want_my",
     re.compile(
         r"\b(i want to|i need to|i'd like to|can i|how do i|let me)\b"
         r".{0,30}\b(my|mine)\b",
         re.IGNORECASE,
     ),
     0.88),

    # "my own …"
    ("my_own",
     re.compile(r"\bmy own\b", re.IGNORECASE),
     0.93),

    # "download/export my data", "delete my account"
    ("action_my_noun",
     re.compile(
         r"\b(download|export|backup|delete|close|deactivate|erase|"
         r"remove|update|change|reset)\s+(my\s+)?(own\s+)?"
         r"(account|data|profile|information|records|history|email|phone|"
         r"password|address|transactions|balance|documents)\b",
         re.IGNORECASE,
     ),
     0.90),
]


# ---------------------------------------------------------------------------
# OTHER patterns — third-party / specific target reference
# ---------------------------------------------------------------------------

_OTHER_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # "user 9921", "user ID 4421", "account #12345"
    ("user_id",
     re.compile(
         r"\b(user|customer|account|client|member|subscriber)\s*"
         r"(id|#|number|no\.?)?\s*[0-9]{3,}",
         re.IGNORECASE,
     ),
     0.97),

    # "someone's data/account/transactions"
    ("someones",
     re.compile(
         r"\b(someone('s)?|another (person|user|customer|individual)('s)?|"
         r"other (person|user|customer)('s)?|"
         r"third.party|third party)\b",
         re.IGNORECASE,
     ),
     0.95),

    # "get/show/fetch [person's name/their] data"
    ("get_third_party",
     re.compile(
         r"\b(get|show|fetch|retrieve|pull|access|extract|find|look up|lookup)\b"
         r".{0,40}\b(his|her|their|the user('s)?|the customer('s)?|"
         r"the account('s)?|the person('s)?)\b",
         re.IGNORECASE,
     ),
     0.90),

    # "bank data/transactions of Rahul / for user Priya"
    ("data_of_for_name",
     re.compile(
         r"\b(data|transactions?|records?|account|information|details?|history)\b"
         r".{0,20}\b(of|for|belonging to|associated with)\b.{0,30}"
         r"\b(user|customer|client|account|[A-Z][a-z]{2,})\b",
         re.IGNORECASE,
     ),
     0.88),

    # Explicit "another user's …", "other people's …"
    ("other_possessive",
     re.compile(
         r"\b(another|other|different)\s+(user|person|individual|customer|account)('s)?\b",
         re.IGNORECASE,
     ),
     0.92),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(text: str) -> OwnershipResult:
    """
    Analyse text and return an OwnershipResult.

    The result's .decision is one of:
      ALLOW_WITH_AUTH — first-person own-data request; allow if authenticated
      BLOCK           — clear third-party data request; ML block should stand
      UNCERTAIN       — no ownership signal; defer to ML verdict
    """
    has_data_topic = bool(_DATA_TOPICS.search(text))

    # ── Check OTHER first (higher specificity — e.g. "user 9921") ───────────
    for name, pattern, conf in _OTHER_PATTERNS:
        if pattern.search(text):
            if has_data_topic:
                return OwnershipResult(
                    ownership=Ownership.OTHER,
                    confidence=conf,
                    matched_pattern=name,
                    decision="BLOCK",
                    reason=f"Third-party data request detected ({name})",
                )
            # Pattern matched but no data topic — still flag
            return OwnershipResult(
                ownership=Ownership.OTHER,
                confidence=conf * 0.8,
                matched_pattern=name,
                decision="BLOCK",
                reason=f"Third-party reference without clear data topic ({name})",
            )

    # ── Check SELF ───────────────────────────────────────────────────────────
    for name, pattern, conf in _SELF_PATTERNS:
        if pattern.search(text):
            if has_data_topic:
                return OwnershipResult(
                    ownership=Ownership.SELF,
                    confidence=conf,
                    matched_pattern=name,
                    decision="ALLOW_WITH_AUTH",
                    reason="User requesting their own data — allow if authenticated",
                )
            # "my" without a data topic — still lean SELF but lower confidence
            return OwnershipResult(
                ownership=Ownership.SELF,
                confidence=conf * 0.6,
                matched_pattern=name,
                decision="ALLOW_WITH_AUTH",
                reason="First-person possessive detected (no data topic — cautious allow)",
            )

    # ── Unknown ──────────────────────────────────────────────────────────────
    return OwnershipResult(
        ownership=Ownership.UNKNOWN,
        confidence=0.0,
        matched_pattern="none",
        decision="UNCERTAIN",
        reason="No ownership signal — defer to ML moderator verdict",
    )
