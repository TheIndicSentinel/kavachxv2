from typing import List, Dict, Any, Tuple
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
            {"rule_id": "builtin-rbi-003", "condition": "debt_ratio_exceeds_threshold", "threshold": 0.40, "action": "human_review", "message": "RBI Advisory: Debt-to-Income ratio (55%) requires manual review for secondary factors."},
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
        "id": "builtin-financial-safety",
        "name": "AML & Anti-Fraud Guard",
        "description": "Blocks prompts related to money laundering, tax evasion, and financial crimes.",
        "policy_type": "compliance",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-aml-001", "condition": "financial_crime_detected", "threshold": 0.70, "action": "block", "message": "BLOCKED — Financial crime indicators detected (money laundering, hawala, tax evasion, or fraud facilitation). Violates Prevention of Money Laundering Act 2002 (PMLA) §3/§4, Income Tax Act 1961 §276C, and FEMA 1999 §13. RBI Master Circular on KYC/AML requires mandatory reporting."},
        ]
    },
    {
        "id": "builtin-self-harm",
        "name": "Self-Harm Prevention",
        "description": "Detects and blocks content related to suicide or self-harm.",
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-sh-002", "condition": "self_harm_review", "threshold": 0.30, "action": "human_review", "message": "Possible self-harm or suicide intent detected — flagged for urgent human review. [Mental Health Act 2017 §115; WHO Safe Messaging Guidelines]"},
            {"rule_id": "builtin-sh-001", "condition": "self_harm_detected", "threshold": 0.50, "action": "block", "message": "CRITICAL — Self-harm or suicide facilitation detected. Blocked under Mental Health Care Act 2017 §115 (abetment of suicide) and IPC §306. WHO Safe Messaging Guidelines require immediate escalation."},
        ]
    },
    {
        "id": "builtin-violence-prevention",
        "name": "Violence Prevention Guard",
        "description": "Protects against prompts related to weapons, bombs, and physical violence.",
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-v-001", "condition": "violence_detected", "threshold": 0.70, "action": "block", "message": "CRITICAL — Violent or dangerous content detected. Blocked under IPC §153A (promoting enmity), §307 (attempt to murder), and IT Act 2000 §67 (obscene/violent publication). EU AI Act Art. 5.1(b) prohibits AI that exploits vulnerable persons."},
        ]
    },
    {
        "id": "builtin-llm-safety",
        "name": "LLM Safety Guard",
        "description": "Blocks toxic or harmful outputs in LLM systems. Routes borderline toxicity to human review.",
        "policy_type": "llm_safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-llm-safety-review", "condition": "toxicity_review", "threshold": 0.35, "action": "human_review", "message": "Elevated toxicity detected (35–50% confidence) — flagged for human review. [IT Act 2000 §67; IPC §153A; Platform liability under IT (Intermediary Guidelines) Rules 2021 Rule 3(1)(b)]"},
            {"rule_id": "builtin-llm-safety", "condition": "toxicity_exceeds_threshold", "threshold": 0.50, "action": "block", "message": "BLOCKED — Toxic or hateful content exceeds safety threshold. Violates IT Act 2000 §67 (obscene/harmful online content), IPC §153A (promoting enmity), and IT (Intermediary Guidelines) Rules 2021 Rule 3(1)(b)."},
        ]
    },
    {
        "id": "principle-fwd-001",
        "name": "Adversarial Robustness",
        "description": "Detects and blocks prompt injection and jailbreak attempts. Routes borderline injection signals to human review.",
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "principle-fwd-001-review", "condition": "injection_review", "threshold": 0.50, "action": "human_review", "message": "Possible prompt injection or jailbreak attempt detected (50–70% confidence) — flagged for human review. [IT Act 2000 §43 (unauthorized data access); EU AI Act Art. 15]"},
            {"rule_id": "principle-fwd-001", "condition": "injection_detected", "threshold": 0.70, "action": "block", "message": "BLOCKED — High-confidence prompt injection / jailbreak detected. Constitutes unauthorized system access attempt under IT Act 2000 §43 and §66 (computer-related offence)."},
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
    # External AI usage is captured in context_metadata for compliance logging
    # but does NOT generate a policy violation/alert on every harmless prompt.

    # ── BASCG T2-A: PII & Sensitive Document Shield ─────────────────────────────
    # Self-contained: fires on pii_detected flag set by the governance service.
    # Covers Indian (Aadhaar, PAN, VPA, Passport, DL, Voter ID, GSTIN, IFSC)
    # and Global (SSN, Credit Card, IBAN, NHS, NI, SIN, Medicare, TFN) formats.
    # Two-tier: scanner-confirmed third-party intent → BLOCK; own-data / ambiguous → HUMAN_REVIEW.
    {
        "id": "builtin-pii-shield",
        "name": "PII & Sensitive Document Shield (Indian + Global)",
        "description": "Blocks prompts with confirmed third-party PII access intent; routes ambiguous / self-service PII to human review.",
        "policy_type": "privacy",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-pii-001", "condition": "pii_block_confirmed", "threshold": None, "action": "block",
             "message": "Sensitive PII with third-party access intent detected. Blocked under DPDP Act 2023 / GDPR."},
            {"rule_id": "builtin-pii-002", "condition": "pii_detected_review", "threshold": None, "action": "human_review",
             "message": "PII detected without confirmed third-party intent — routed for human review under DPDP Act 2023."},
        ]
    },
    # ── DPDP 2023 / IT Act 2000: PII Harvesting Intent ──────────────────────────
    # Fires when a prompt's INTENT is to obtain or aggregate personal data about
    # third parties — even if no actual PII is present in the text.
    # Covers: ID-document requests, bulk contact harvesting, medical record access,
    # data scraping from platforms, and sensitive profiling dataset generation.
    {
        "id": "builtin-dpdp-pii-harvesting",
        "name": "PII Harvesting Intent Blocker (DPDP 2023 / IT Act)",
        "description": (
            "Blocks prompts that express intent to obtain, aggregate, or scrape personal "
            "data of third parties without consent — violating DPDP Act 2023 §4 (purpose "
            "limitation), §7 (data minimisation), IT Act 2000 §43A, and GDPR Art. 5."
        ),
        "policy_type": "privacy",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {
                "rule_id": "builtin-dpdp-pii-harvest-001",
                "condition": "pii_harvesting_intent",
                "threshold": 0.70,
                "action": "block",
                "message": (
                    "DPDP Act 2023 violation: Prompt indicates intent to harvest, scrape, or "
                    "access personal data of third parties without consent. Blocked under §4 "
                    "(purpose limitation) and §7 (data minimisation). Reference: IT Act §43A."
                ),
            },
        ],
    },
    # ── EU AI Act Article 5 — Prohibited Practices ───────────────────────────────
    # The following AI practices are PROHIBITED by EU AI Act Article 5 (in force 2026):
    #   • Biometric categorisation inferring race, religion, politics, sexuality (Art. 5.1(g))
    #   • Real-time remote biometric identification in public spaces (Art. 5.1(d))
    #   • Social scoring systems by public/private entities (Art. 5.1(c))
    #   • Emotion recognition in workplace / educational settings (Art. 5.1(f))
    #   • Exploitation of vulnerable groups (Art. 5.1(b))
    #   • Subliminal manipulation techniques (Art. 5.1(a))
    # Indian equivalents: Constitution Art. 15, DPDP 2023 §9, IT Act §43A.
    {
        "id": "builtin-eu-ai-act-art5",
        "name": "EU AI Act Article 5 — Prohibited AI Practices",
        "description": (
            "Blocks AI uses explicitly prohibited by EU AI Act Article 5 and their Indian "
            "constitutional equivalents: biometric categorisation by religion/caste/race, "
            "real-time facial recognition in public spaces, social scoring, and emotion "
            "recognition in workplaces or schools."
        ),
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {
                "rule_id": "builtin-eu-ai-act-art5-001",
                "condition": "eu_ai_act_prohibited",
                "threshold": 0.70,
                "action": "block",
                "message": (
                    "EU AI Act Article 5 violation: This request involves a prohibited AI "
                    "practice — biometric categorisation by religion, caste, ethnicity, or "
                    "race; real-time facial identification in public spaces; social scoring; "
                    "or emotion recognition in restricted contexts. Blocked under Art. 5 and "
                    "Indian Constitution Art. 15 / DPDP 2023 §9."
                ),
            },
        ],
    },
    # ── DPDP AI Moderator Policy ──────────────────────────────────────────────
    # Fires when the 3-layer AI moderator (DistilBERT+LoRA + MiniLM) has
    # classified the prompt as a DPDP Act violation with high confidence.
    # This is a learned complement to the regex-only PII harvesting rule above.
    {
        "id": "builtin-dpdp-ai-moderator",
        "name": "DPDP AI Intent Classifier",
        "description": (
            "Blocks prompts classified by the DistilBERT+LoRA intent model as "
            "DPDP Act violations (personal data, financial data, health data, "
            "profiling, surveillance, or illegal activity) with risk score ≥ 0.80."
        ),
        "policy_type": "compliance",
        "severity": "critical",
        "jurisdiction": "IN",
        "rules": [
            {
                "rule_id": "builtin-dpdp-ai-001",
                "condition": "dpdp_ai_block",
                "threshold": None,
                "action": "block",
                "message": (
                    "DPDP Act 2023 violation detected by AI intent classifier. "
                    "The prompt was classified as a high-risk data privacy violation "
                    "(personal/health/financial data access, profiling, surveillance, "
                    "or illegal activity). Blocked under DPDP §4, §7, §9 and IT Act §43A."
                ),
            },
            {
                "rule_id": "builtin-dpdp-ai-002",
                "condition": "dpdp_ai_review",
                "threshold": None,
                "action": "human_review",
                "message": (
                    "DPDP Act 2023: Prompt flagged for human review by AI intent classifier "
                    "(risk score 0.50–0.79). Possible personal data, profiling, or surveillance "
                    "intent detected. Review required before processing."
                ),
            },
        ],
    },
    # ── Composite Risk Gate ───────────────────────────────────────────────────
    # Catch-all: fires when aggregate risk_score exceeds threshold regardless of
    # which individual rules fired. Prevents high-aggregate-risk prompts slipping
    # through when no single policy rule reaches its own trigger threshold.
    {
        "id": "builtin-composite-risk-gate",
        "name": "Composite Risk Gate",
        "description": (
            "Blocks or reviews prompts whose aggregate risk score exceeds the composite "
            "threshold, even if no individual policy rule reached its own trigger point. "
            "Prevents high-risk prompts from passing through via threshold gaps."
        ),
        "policy_type": "safety",
        "severity": "high",
        "jurisdiction": "GLOBAL",
        "rules": [
            {
                "rule_id": "builtin-crg-review",
                "condition": "risk_score_exceeds_threshold",
                "threshold": 0.60,
                "action": "human_review",
                "message": "Composite risk score exceeds 60% — elevated aggregate risk detected across multiple signals. Flagged for human review.",
            },
            {
                "rule_id": "builtin-crg-block",
                "condition": "risk_score_exceeds_threshold",
                "threshold": 0.80,
                "action": "block",
                "message": "Composite risk score exceeds 80% — high aggregate risk. Blocked by composite safety gate.",
            },
        ],
    },
    {
        "id": "builtin-informal-economy",
        "name": "Economic Inclusion Policy",
        "description": "Ensures AI models do not systematically exclude or penalize informal economy workers (BPL, daily wagers).",
        "policy_type": "fairness",
        "severity": "medium",
        "jurisdiction": "IN",
        "rules": [
            {"rule_id": "builtin-iec-001", "condition": "economic_equity_gap", "threshold": 0.15, "action": "alert", "message": "Economic Diversity Alert: Minor outcomes disparity (18%) detected for informal economy segments."},
            {"rule_id": "builtin-iec-002", "condition": "economic_equity_gap", "threshold": 0.30, "action": "block", "message": "Regulatory Block: Systematic exclusion of informal economy/BPL segments detected."},
        ]
    },
    {
        "id": "builtin-ownership-override",
        "name": "Data Ownership Override",
        "description": "Post-ML rule layer: allows first-person own-data requests (SELF) "
                       "and reinforces blocks for confirmed third-party data access (OTHER). "
                       "Runs after ML models — ownership signal takes precedence over topic signal.",
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-own-001", "condition": "data_request_third_party", "threshold": None,
             "action": "block",
             "message": "BLOCKED — Third-party personal data access detected without consent. Requesting or accessing another individual's data violates DPDP Act 2023 §4 (lawful processing) and §7 (data minimisation), IT Act 2000 §43 (unauthorized access), and GDPR Art. 6 (lawful basis). The Data Principal whose data is targeted has not granted consent."},
        ]
    },
    {
        "id": "builtin-general-safety",
        "name": "General Safety Guard",
        "description": "Acts on verdicts from the General Safety ML moderator (DistilBERT+LoRA + MiniLM). "
                       "Direct verdict binding — not score-threshold dependent.",
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {"rule_id": "builtin-gs-review", "condition": "general_safety_review", "threshold": None, "action": "human_review",
             "message": "General Safety AI Moderator flagged this prompt for urgent human review — possible harmful, violent, or manipulative content detected. [IT Act 2000 §67; EU AI Act Art. 5.1(a)-(b)]"},
            {"rule_id": "builtin-gs-block",  "condition": "general_safety_block",  "threshold": None, "action": "block",
             "message": "BLOCKED by General Safety AI Moderator — high-confidence detection of harmful content (self-harm, violence, hate speech, financial crime, prohibited AI practice, or prompt injection). Blocked under applicable provisions of IT Act 2000 §66/§67, IPC §153A/§306, PMLA 2002, and EU AI Act Art. 5."},
        ]
    },
    # ── Infrastructure Credential Shield ─────────────────────────────────────
    # Hard Block: requests for connection strings, API keys, passwords, or any
    # infrastructure secrets are ALWAYS blocked — regardless of self-proclaimed role.
    # Identity / role claims ("I am a researcher/admin") inside a prompt are NOT
    # accepted as bypass. Authentication happens at the system boundary only.
    # Rationale: DPDP Act 2023 Reasonable Security Safeguard mandate; IT Act 2000
    # §43A (failure to protect data); ISO 27001 access control requirements.
    {
        "id": "builtin-credential-shield",
        "name": "Infrastructure Credential Shield",
        "description": (
            "Hard blocks any prompt requesting database connection strings, API keys, "
            "passwords, private keys, or internal infrastructure details. Self-proclaimed "
            "roles (researcher, admin, security team) are NOT accepted as bypass — "
            "authentication happens at the system boundary, not inside a prompt."
        ),
        "policy_type": "safety",
        "severity": "critical",
        "jurisdiction": "GLOBAL",
        "rules": [
            {
                "rule_id": "builtin-cred-001",
                "condition": "credential_exposure_detected",
                "threshold": None,
                "action": "block",
                "message": (
                    "BLOCKED — Request for infrastructure credentials or secrets detected. "
                    "Connection strings, API keys, passwords, private keys, and internal "
                    "IP addresses are hard-blocked regardless of stated role or context. "
                    "Self-proclaimed identity ('I am an admin/researcher') is never accepted "
                    "as a bypass. Violates DPDP Act 2023 Reasonable Security Safeguards mandate, "
                    "IT Act 2000 §43A (failure to implement reasonable security practices), "
                    "and ISO 27001 access control requirements (A.9)."
                ),
            },
        ],
    },
    # ── DPDP §16 Cross-Border Transfer to Restricted Jurisdiction ─────────────
    # DPDP Act 2023 §16: personal data of Indian data principals may only be
    # transferred to countries/territories notified by the Central Government.
    # Transfer to a "restricted", "blacklisted", or "non-notified" jurisdiction
    # for cost-saving or convenience reasons is a direct statutory violation.
    {
        "id": "builtin-dpdp-s16-cross-border",
        "name": "DPDP §16 Cross-Border Transfer Restriction",
        "description": (
            "Blocks prompts describing transfer of Indian personal data to restricted, "
            "blacklisted, or non-notified countries in violation of DPDP Act 2023 §16. "
            "Cost-saving rationale does not override the statutory requirement for "
            "adequacy notification by the Central Government."
        ),
        "policy_type": "compliance",
        "severity": "critical",
        "jurisdiction": "IN",
        "rules": [
            {
                "rule_id": "builtin-s16-001",
                "condition": "cross_border_restricted_country",
                "threshold": None,
                "action": "block",
                "message": (
                    "BLOCKED — DPDP Act 2023 §16 Violation: Transfer of personal data to a "
                    "restricted, blacklisted, or non-notified jurisdiction detected. "
                    "The Central Government has not notified this country as a permitted "
                    "destination for Indian personal data. Cost-saving or operational "
                    "convenience does not create a legal basis for this transfer. "
                    "Penalty: DPDP Act §33 (financial penalty up to ₹250 crore). "
                    "Reference: IT Act 2000 §43A; RBI data localisation norms."
                ),
            },
        ],
    },
    # ── Shadow AI Detection ───────────────────────────────────────────────────
    # Detects employees processing company/customer data through personal or
    # unauthorized AI accounts (ChatGPT personal tier, etc.) — bypassing the
    # organization's governed AI infrastructure with no audit trail or DPA in place.
    # This constitutes a DPDP compliance failure: no consent audit trail, no data
    # processor agreement with the external AI provider, no access controls.
    {
        "id": "builtin-shadow-ai",
        "name": "Shadow AI / Unauthorized External AI Usage",
        "description": (
            "Detects and blocks employees processing company or customer data through "
            "personal, free-tier, or unauthorized external AI accounts — bypassing "
            "organizational governance, audit trails, and data processor agreements."
        ),
        "policy_type": "compliance",
        "severity": "high",
        "jurisdiction": "GLOBAL",
        "rules": [
            {
                "rule_id": "builtin-shadow-ai-001",
                "condition": "shadow_ai_company_data",
                "threshold": None,
                "action": "block",
                "message": (
                    "BLOCKED — Shadow AI usage detected: company or customer data is being "
                    "processed through a personal or unauthorized external AI account. "
                    "This bypasses organizational data governance, creates an unaudited data "
                    "processing channel, and violates the requirement for a signed Data "
                    "Processing Agreement with any third-party AI provider. "
                    "Violations: DPDP Act 2023 §8 (data fiduciary obligations), §9 (sensitive "
                    "data safeguards), IT Act 2000 §43A. EU AI Act Art. 28 (provider obligations). "
                    "Report this incident to your Data Protection Officer immediately."
                ),
            },
        ],
    },
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
    ) -> Tuple[List[Dict], EnforcementDecision]:
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
        if condition == "caste_proxy_detected":
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "caste_proxy_correlation"), None)
            if disparity is not None:
                return disparity > (threshold or 0.08)
            direct = max(
                float(input_data.get("caste_proxy_score", 0)),
                float(context.get("caste_proxy_disparity", 0)),
            )
            return direct > (threshold or 0.08)

        elif condition == "gender_disparity_detected":
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "gender_disparity"), None)
            if disparity is not None:
                return disparity > (threshold or 0.10)
            direct = max(
                float(input_data.get("gender_proxy", 0)),
                float(context.get("gender_disparity", 0)),
                float(input_data.get("name_gender_signal", 0)) - 0.5 if input_data.get("name_gender_signal", 0) > 0.5 else 0,
            )
            return direct > (threshold or 0.10)

        elif condition == "multilingual_accuracy_gap":
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "multilingual_equity"), None)
            if disparity is not None:
                return disparity > (threshold or 0.08)
            gap_pct = float(input_data.get("performance_gap_pct", 0))
            return (gap_pct / 100.0) > (threshold or 0.08)

        # 2. Safety conditions
        elif condition == "confidence_below_threshold":
            return confidence < (threshold or 0.55)

        elif condition == "toxicity_exceeds_threshold":
            tox = input_data.get("toxicity_score", 0)
            return tox > (threshold or 0.50)

        elif condition == "toxicity_review":
            tox = input_data.get("toxicity_score", 0)
            return tox > (threshold or 0.35)

        elif condition == "injection_detected":
            inject = input_data.get("prompt_injection_score", 0)
            return inject > (threshold or 0.70)

        elif condition == "injection_review":
            inject = input_data.get("prompt_injection_score", 0)
            return inject > (threshold or 0.50)

        elif condition == "financial_crime_detected":
            score = input_data.get("financial_crime_score", 0)
            return score > (threshold or 0.70)

        elif condition == "self_harm_review":
            score = input_data.get("self_harm_score", 0)
            return score > (threshold or 0.30)

        elif condition == "self_harm_detected":
            score = input_data.get("self_harm_score", 0)
            return score > (threshold or 0.50)

        elif condition == "violence_detected":
            score = input_data.get("violence_score", 0)
            return score > (threshold or 0.70)

        elif condition == "general_safety_block":
            return bool(input_data.get("general_safety_block"))

        elif condition == "general_safety_review":
            return bool(input_data.get("general_safety_review"))

        elif condition == "data_request_third_party":
            return bool(input_data.get("data_request_third_party"))

        # 3. Compliance conditions
        elif condition == "debt_ratio_exceeds_threshold":
            ratio = input_data.get("debt_ratio", 0)
            return ratio > (threshold or 0.40)

        elif condition == "healthcare_low_confidence":
            in_healthcare = context.get("domain") == "healthcare" or context.get("model_category") == "healthcare"
            return in_healthcare and confidence < (threshold or 0.70)

        elif condition == "pii_block_confirmed":
            # Fires when scanner confirmed PII + third-party access intent (should_block=True)
            return input_data.get("pii_detected") is True and input_data.get("pii_scanner_should_block") is True

        elif condition == "pii_detected_review":
            # Fires when PII is present but no confirmed third-party intent — route to review
            return input_data.get("pii_detected") is True and not input_data.get("pii_scanner_should_block")

        elif condition == "personal_data_without_consent":
            # Treat missing consent_verified as unconsented (None != False — explicit fix)
            personal_data = input_data.get("personal_data_used") is True
            consent_verified = input_data.get("consent_verified")
            return personal_data and consent_verified is not True

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

        # 3b. Privacy — PII harvesting intent (DPDP 2023 / IT Act / GDPR)
        elif condition == "pii_harvesting_intent":
            score = input_data.get("pii_harvesting_score", 0)
            return score > (threshold or 0.70)

        # 3c. EU AI Act Article 5 — prohibited AI practices
        elif condition == "eu_ai_act_prohibited":
            score = input_data.get("eu_ai_act_score", 0)
            return score > (threshold or 0.70)

        # 4. Built-in system conditions
        elif condition == "risk_score_exceeds_threshold":
            return risk_score > (threshold or 0.70)

        elif condition == "unauthorized_tool_detected":
            return context.get("shadow_ai_detected") is True or input_data.get("external_tool_signature") is not None

        elif condition == "economic_equity_gap":
            disparity = next((f.get("disparity", 0) for f in fairness_results if f.get("metric") == "economic_equity"), None)
            if disparity is not None:
                return disparity > (threshold or 0.15)
            return float(input_data.get("economic_proxy_score", 0)) > (threshold or 0.15)

        # ── DPDP AI Moderator conditions ──────────────────────────────────────
        elif condition == "dpdp_ai_block":
            # Fires when the trained DistilBERT+LoRA classifier returns BLOCK
            return input_data.get("dpdp_ai_verdict") == "BLOCK"

        elif condition == "dpdp_ai_review":
            # Fires when classifier returns REVIEW (but NOT BLOCK — avoid double-firing)
            return input_data.get("dpdp_ai_verdict") == "REVIEW"

        # ── Infrastructure Credential Shield ───────────────────────────────────
        elif condition == "credential_exposure_detected":
            return bool(input_data.get("credential_exposure_detected"))

        # ── DPDP §16 Cross-Border Transfer to Restricted Jurisdiction ──────────
        elif condition == "cross_border_restricted_country":
            return bool(input_data.get("cross_border_restricted_country"))

        # ── Shadow AI Detection ────────────────────────────────────────────────
        elif condition == "shadow_ai_company_data":
            return bool(input_data.get("shadow_ai_company_data"))

        return False


    def _map_action(self, action: str) -> EnforcementDecision:
        normalized = str(action or "").lower()
        if normalized in ("allow", "pass"):
            return EnforcementDecision.PASS
        if normalized == "alert":
            return EnforcementDecision.ALERT
        if normalized in ("human_review", "review"):
            return EnforcementDecision.HUMAN_REVIEW
        if normalized in ("block", "suspend", "suspension"):
            return EnforcementDecision.BLOCK
        return EnforcementDecision.PASS
