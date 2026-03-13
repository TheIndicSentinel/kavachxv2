from typing import List, Dict, Any
from app.models.schemas import EnforcementDecision

BUILT_IN_POLICIES = [
    {
        "id": "builtin-rbi-001",
        "name": "RBI Fair Lending — Caste Match",
        "description": "Ensures no systematic bias against caste-proxy indicators in credit scoring.",
        "policy_type": "fairness",
        "severity": "critical",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-ctx-001", "condition": "caste_proxy_detected", "threshold": 0.08, "action": "block", "message": "High correlation (V > 0.08) with caste-proxy indicators — violates RBI Fair Lending codes."},
        ]
    },
    {
        "id": "builtin-rbi-003",
        "name": "RBI Digital Lending — DTI Cap",
        "description": "Enforces 40% Debt-to-Income cap for unsecured digital lending.",
        "policy_type": "compliance",
        "severity": "high",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-rbi-003", "condition": "debt_ratio_exceeds_threshold", "threshold": 0.40, "action": "alert", "message": "Debt-to-Income ratio exceeds RBI 40% cap."},
        ]
    },
    {
        "id": "principle-in-002",
        "name": "Constitutional Equity (Art. 15)",
        "description": "Prohibits discrimination on grounds of religion, race, caste, sex or place of birth.",
        "policy_type": "fairness",
        "severity": "critical",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "principle-in-002", "condition": "gender_disparity_detected", "threshold": 0.10, "action": "block", "message": "Gender disparity exceeds constitutional Art. 15 limits."},
        ]
    },
    {
        "id": "builtin-dpdp-001",
        "name": "DPDP 2023 Consent Gate",
        "description": "Enforces explicit consent for processing of personal data.",
        "policy_type": "compliance",
        "severity": "critical",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-dpdp-001", "condition": "personal_data_without_consent", "threshold": None, "action": "block", "message": "DPDP Act violation: Personal data processed without verifiable consent."},
        ]
    },
    {
        "id": "builtin-llm-safety",
        "name": "LLM Safety Guard",
        "description": "Blocks toxic or harmful outputs in LLM systems.",
        "policy_type": "llm_safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-llm-safety", "condition": "toxicity_exceeds_threshold", "threshold": 0.50, "action": "block", "message": "Toxicity level exceeds safety threshold."},
        ]
    },
    {
        "id": "principle-fwd-001",
        "name": "Adversarial Robustness",
        "description": "Detects and blocks prompt injection and jailbreak attempts.",
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "principle-fwd-001", "condition": "injection_detected", "threshold": 0.70, "action": "block", "message": "High-confidence prompt injection attempt detected."},
        ]
    },
    {
        "id": "builtin-meity-001",
        "name": "MeitY High-Risk Domain",
        "description": "Requires enhanced confidence for healthcare-related AI inferences.",
        "policy_type": "compliance",
        "severity": "high",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-meity-001", "condition": "healthcare_low_confidence", "threshold": 0.70, "action": "alert", "message": "MeitY: Healthcare inference below 70% confidence threshold."},
        ]
    },
    {
        "id": "builtin-sc-001",
        "name": "Low Confidence Gate",
        "description": "Ensures low-confidence predictions are reviewed by humans.",
        "policy_type": "safety",
        "severity": "medium",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-sc-001", "condition": "confidence_below_threshold", "threshold": 0.55, "action": "human_review", "message": "Confidence below safety floor (55%)."},
        ]
    },
    {
        "id": "principle-in-007",
        "name": "Gig Economy Accountability",
        "description": "Ensures algorithmic deactivation of workers follows due process.",
        "policy_type": "compliance",
        "severity": "high",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "principle-in-007", "condition": "algorithmic_deactivation", "threshold": None, "action": "human_review", "message": "Algorithmic deactivation detected — mandatory human review required."},
        ]
    },
    {
        "id": "builtin-nha-001",
        "name": "ABDM Data Sovereignty",
        "description": "Protects health data linked to Ayushman Bharat Digital Mission.",
        "policy_type": "compliance",
        "severity": "critical",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-nha-001", "condition": "abdm_consent_missing", "threshold": None, "action": "block", "message": "ABDM-linked data accessed without verified consent."},
        ]
    },
    {
        "id": "principle-in-006",
        "name": "EdTech Non-Surveillance",
        "description": "Restricts behavioral profiling of minor students.",
        "policy_type": "compliance",
        "severity": "high",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "principle-in-006", "condition": "student_surveillance", "threshold": None, "action": "block", "message": "NEP 2020: Profile tracking of minor students without consent is prohibited."},
        ]
    },
    {
        "id": "builtin-irdai-001",
        "name": "IRDAI Explainability Mandate",
        "description": "Requires insurance claims to have explainable rationale.",
        "policy_type": "compliance",
        "severity": "high",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-irdai-001", "condition": "unexplainable_insurance_decision", "threshold": 0.40, "action": "human_review", "message": "IRDAI: Insurance rationale below explainability floor (40%)."},
        ]
    },
    {
        "id": "builtin-perf-001",
        "name": "Model Drift Monitor",
        "description": "Alerts when model performance degrades beyond baseline.",
        "policy_type": "performance",
        "severity": "medium",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-perf-001", "condition": "drift_exceeds_threshold", "threshold": 0.20, "action": "alert", "message": "Model PSI exceeds drift threshold (0.20)."},
        ]
    },
    {
        "id": "builtin-ctx-002",
        "name": "Multilingual Equity",
        "description": "Ensures performance parity across supported Indian languages.",
        "policy_type": "fairness",
        "severity": "high",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-ctx-002", "condition": "multilingual_accuracy_gap", "threshold": 0.08, "action": "alert", "message": "Language accuracy gap exceeds +/- 8% equity threshold."},
        ]
    },
    {
        "id": "builtin-shadow-ai",
        "name": "Shadow AI & Tool Discovery",
        "description": "Detects usage of unauthorized or public AI tools (e.g., ChatGPT, Claude) via browser traffic signatures.",
        "policy_type": "compliance",
        "severity": "high",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-shadow-ai-001", "condition": "unauthorized_tool_detected", "threshold": None, "action": "alert", "message": "Shadow AI Usage: Prompt sent to unauthorized external AI tool detected via browser signature."},
        ]
    },
    {
        "id": "builtin-informal-economy",
        "name": "Economic Inclusion Policy",
        "description": "Ensures AI models do not systematically exclude or penalize informal economy workers (BPL, daily wagers).",
        "policy_type": "fairness",
        "severity": "medium",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-iec-001", "condition": "economic_equity_gap", "threshold": 0.15, "action": "alert", "message": "High disparity in outcomes for informal economy/BPL segments."},
        ]
    }
]


class PolicyEngine:
    """Evaluates inference events against registered governance policies."""

    def __init__(self, policies: List[Dict] = None):
        self.policies = policies or BUILT_IN_POLICIES

    def evaluate(
        self,
        inference_data: Dict[str, Any],
        fairness_results: List[Dict],
        risk_score: float
    ) -> tuple[List[Dict], EnforcementDecision]:
        """
        Evaluate all active policies against an inference event.
        Returns (violations list, final enforcement decision).
        """
        violations = []
        highest_action = EnforcementDecision.PASS
        action_priority = {
            "pass": 0, "alert": 1, "human_review": 2, "block": 3
        }

        for policy in self.policies:
            for rule in policy.get("rules", []):
                triggered = self._evaluate_rule(rule, inference_data, fairness_results, risk_score)
                if triggered:
                    violations.append({
                        "policy_id": policy["id"],
                        "policy_name": policy["name"],
                        "rule_id": rule["rule_id"],
                        "severity": policy["severity"],
                        "action": rule["action"],
                        "message": rule["message"],
                        "jurisdiction": policy.get("jurisdiction", "GLOBAL"),
                    })
                    action = rule["action"]
                    if action_priority.get(action, 0) > action_priority.get(highest_action.value.lower(), 0):
                        highest_action = self._map_action(action)

        return violations, highest_action

    def _evaluate_rule(self, rule: Dict, inference_data: Dict, fairness_results: List, risk_score: float) -> bool:
        condition = rule["condition"]
        threshold = rule.get("threshold")
        
        # Flatten structure for easy access
        input_data = inference_data.get("input_data", {})
        context = inference_data.get("context", {})
        confidence = inference_data.get("confidence", 1.0)

        # 1. Fairness conditions
        # Each condition first checks the FairnessMonitor flag results,
        # then falls back to direct signal fields from simulation scenarios.
        if condition == "caste_proxy_detected":
            # Check FairnessMonitor result first
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "caste_proxy_correlation"), None)
            if disparity is not None:
                return disparity > (threshold or 0.08)
            # Fallback: direct signal from simulation payload
            direct = max(
                float(input_data.get("caste_proxy_score", 0)),
                float(context.get("caste_proxy_disparity", 0)),
            )
            return direct > (threshold or 0.08)

        elif condition == "gender_disparity_detected":
            # Check FairnessMonitor result first
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "gender_disparity"), None)
            if disparity is not None:
                return disparity > (threshold or 0.10)
            # Fallback: direct signal from simulation payload
            direct = max(
                float(input_data.get("gender_proxy", 0)),
                float(context.get("gender_disparity", 0)),
                float(input_data.get("name_gender_signal", 0)) - 0.5 if input_data.get("name_gender_signal", 0) > 0.5 else 0,
            )
            return direct > (threshold or 0.10)

        elif condition == "multilingual_accuracy_gap":
            # Check FairnessMonitor result first
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "multilingual_equity"), None)
            if disparity is not None:
                return disparity > (threshold or 0.08)
            # Fallback: performance_gap_pct field from simulation payload (convert percent to decimal)
            gap_pct = float(input_data.get("performance_gap_pct", 0))
            return (gap_pct / 100.0) > (threshold or 0.08)

        # 2. Safety conditions
        elif condition == "confidence_below_threshold":
            return confidence < (threshold or 0.55)

        elif condition == "toxicity_exceeds_threshold":
            tox = input_data.get("toxicity_score", 0)
            return tox > (threshold or 0.50)

        elif condition == "injection_detected":
            inject = input_data.get("prompt_injection_score", 0)
            return inject > (threshold or 0.70)

        # 3. Compliance conditions
        elif condition == "debt_ratio_exceeds_threshold":
            ratio = input_data.get("debt_ratio", 0)
            return ratio > (threshold or 0.40)

        elif condition == "healthcare_low_confidence":
            in_healthcare = context.get("domain") == "healthcare" or context.get("model_category") == "healthcare"
            return in_healthcare and confidence < (threshold or 0.70)

        elif condition == "personal_data_without_consent":
            return input_data.get("personal_data_used") is True and input_data.get("consent_verified") is False

        elif condition == "abdm_consent_missing":
            return (input_data.get("abdm_linked") is True or context.get("abdm") is True) and input_data.get("consent_verified") is False

        elif condition == "student_surveillance":
            in_education = context.get("domain") == "education"
            monitored = input_data.get("continuous_monitoring") is True
            no_consent = input_data.get("parental_consent") is False
            return in_education and monitored and no_consent

        elif condition == "algorithmic_deactivation":
            return context.get("algorithmic_deactivation") is True

        elif condition == "unexplainable_insurance_decision":
            exp_score = input_data.get("explainability_score", 1.0)
            in_insurance = context.get("domain") == "insurance" or context.get("explainability_required") is True
            return in_insurance and exp_score < (threshold or 0.40)

        elif condition == "drift_exceeds_threshold":
            psi = input_data.get("psi_score", 0)
            return psi > (threshold or 0.20)

        # 4. Built-in system conditions
        elif condition == "risk_score_exceeds_threshold":
            return risk_score > (threshold or 0.85)

        elif condition == "unauthorized_tool_detected":
            return context.get("shadow_ai_detected") is True or input_data.get("external_tool_signature") is not None

        elif condition == "economic_equity_gap":
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "economic_equity"), None)
            if disparity is not None:
                return disparity > (threshold or 0.15)
            return float(input_data.get("economic_proxy_score", 0)) > (threshold or 0.15)

        return False


    def _map_action(self, action: str) -> EnforcementDecision:
        mapping = {
            "alert": EnforcementDecision.ALERT,
            "human_review": EnforcementDecision.HUMAN_REVIEW,
            "block": EnforcementDecision.BLOCK,
        }
        return mapping.get(action, EnforcementDecision.PASS)
