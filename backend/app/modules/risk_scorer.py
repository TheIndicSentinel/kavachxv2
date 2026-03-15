"""
KavachX Risk Scoring Engine
Computes composite governance risk scores from multiple signals.
Produces realistic ranges: 0–0.20 (Low), 0.21–0.60 (Moderate), 0.61–1.0 (High)
"""
from typing import Dict, List
from app.models.schemas import RiskLevel


class RiskScorer:
    """
    Computes a composite risk score (0.0–1.0) from:
    - Confidence level
    - Fairness flag severity
    - Policy violation count/severity
    - India-specific context signals
    """

    WEIGHTS = {
        "confidence": 0.15,
        "fairness": 0.30,
        "policy": 0.45,
        "context": 0.10,
    }

    SEVERITY_SCORES = {
        "critical": 1.0,
        "high": 0.75,
        "medium": 0.50,
        "low": 0.25,
    }

    def compute(
        self,
        confidence: float,
        fairness_flags: List[Dict],
        policy_violations: List[Dict],
        context: Dict = None
    ) -> float:
        """Returns risk score between 0.0 (safe) and 1.0 (maximum risk)."""
        context = context or {}

        # If no violations, no fairness flags, and decent confidence → very low risk
        if not policy_violations and not fairness_flags and confidence >= 0.55:
            # Base risk from confidence only (0.0 for 1.0 confidence, 0.07 for 0.95, etc.)
            base = max(0.0, (1.0 - confidence) * self.WEIGHTS["confidence"])
            return round(base, 3)

        # 1. Confidence component (low confidence = higher risk)
        confidence_risk = max(0.0, 1.0 - confidence)

        # 2. Fairness component
        fairness_risk = self._compute_fairness_risk(fairness_flags)

        # 3. Policy violation component
        policy_risk = self._compute_policy_risk(policy_violations)

        # 4. Context component (India-specific signals)
        context_risk = self._compute_context_risk(context)

        # Weighted composite
        risk_score = (
            self.WEIGHTS["confidence"] * confidence_risk +
            self.WEIGHTS["fairness"] * fairness_risk +
            self.WEIGHTS["policy"] * policy_risk +
            self.WEIGHTS["context"] * context_risk
        )

        # Weighted composite
        risk_score = (
            self.WEIGHTS["confidence"] * confidence_risk +
            self.WEIGHTS["fairness"] * fairness_risk +
            self.WEIGHTS["policy"] * policy_risk +
            self.WEIGHTS["context"] * context_risk
        )

        return round(min(1.0, max(0.0, risk_score)), 3)

    def _action_rank(self, action: str) -> int:
        return {"pass": 0, "alert": 1, "human_review": 2, "block": 3}.get(str(action).lower(), 0)

    def _compute_fairness_risk(self, flags: List[Dict]) -> float:
        if not flags:
            return 0.0
        max_disparity = max(f.get("disparity", 0) for f in flags)
        flag_count_penalty = min(0.3, len(flags) * 0.1)
        return min(1.0, max_disparity * 2.0 + flag_count_penalty)

    def _compute_policy_risk(self, violations: List[Dict]) -> float:
        if not violations:
            return 0.0
        max_severity = max(
            self.SEVERITY_SCORES.get(v.get("severity", "low"), 0.25)
            for v in violations
        )
        count_penalty = min(0.2, len(violations) * 0.05)
        return min(1.0, max_severity + count_penalty)

    def _compute_context_risk(self, context: Dict) -> float:
        """India-specific context risk signals."""
        risk = 0.0
        model_category = context.get("model_category", "")
        if model_category in ["financial_credit", "healthcare", "law_enforcement"]:
            risk += 0.5
        elif model_category in ["hr_recruitment", "education"]:
            risk += 0.3

        if context.get("informal_economy_indicator"):
            risk += 0.2

        if context.get("jurisdiction") == "IN" and not context.get("dpdp_compliant"):
            risk += 0.3

        return min(1.0, risk)

    def get_risk_level(self, score: float) -> RiskLevel:
        if score >= 0.60:
            return RiskLevel.HIGH
        if score >= 0.21:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
