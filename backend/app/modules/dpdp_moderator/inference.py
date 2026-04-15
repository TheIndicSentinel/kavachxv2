"""
Three-layer DPDP moderation inference engine.

Layer 1 — Regex fast filter        (deterministic, zero latency)
Layer 2 — DistilBERT+LoRA classifier (probabilistic intent)
Layer 3 — MiniLM embedding similarity (catches paraphrases / adversarial)

Each layer exposes a standalone function so the pipeline stays modular.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Layer 1 — Regex patterns for DPDP-sensitive identifiers
# ---------------------------------------------------------------------------

# Hard PII patterns — always constitute personal data regardless of context.
# Even if the text says "validate" or "regex", an Aadhaar/PAN/card number
# embedded in the prompt is a real identifier (or at least treated as one).
_HARD_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 12-digit Aadhaar (with optional spaces/dashes every 4 digits)
    ("aadhaar", re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    )),
    # PAN card: ABCDE1234F
    ("pan", re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    )),
    # Credit/debit card (16 digits, optional separators)
    ("card", re.compile(
        r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
    )),
    # IFSC code: ABCD0XXXXXX
    ("ifsc", re.compile(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    )),
    # UPI ID: name@bank
    ("upi", re.compile(
        r"\b[\w.\-]+@(?:oksbi|okaxis|okicici|okhdfcbank|ybl|ibl|axl|upi)\b",
        re.IGNORECASE,
    )),
    # Passport number (India): A1234567
    ("passport", re.compile(
        r"\b[A-PR-WY][1-9]\d{6}\b"
    )),
    # Voter ID: ABC1234567
    ("voter_id", re.compile(
        r"\b[A-Z]{3}\d{7}\b"
    )),
]

# Soft PII patterns — these tokens appear legitimately in technical/coding
# contexts (format validation, regex tasks, log masking).
# has_pii is NOT set when the prompt is clearly a coding or engineering task.
_SOFT_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Indian mobile numbers (10 digits, optional +91/0)
    ("phone_in", re.compile(
        r"\b(?:\+91|0)?[6-9]\d{9}\b"
    )),
    # Generic email
    ("email", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )),
    # IPv4 address (potential surveillance indicator)
    ("ipv4", re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )),
]

# Technical-context signals — if present, soft PII tokens are treated as
# format examples rather than actual personal data.
_TECHNICAL_CONTEXT = re.compile(
    r"\b(regex|validate|validation|format|parse|detect|mask|redact|"
    r"extract|sanitize|write\s+a|write\s+code|function|how\s+(do\s+i|to)|"
    r"schema|placeholder|dummy|mock|sample|example|test\s+case|"
    r"log\s+file|log\s+mask|anonymis|pseudonym|unit\s+test)\b",
    re.IGNORECASE,
)

# Keep a unified list for callers that need all PII pattern names
_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = _HARD_PII_PATTERNS + _SOFT_PII_PATTERNS

# High-risk intent keyword phrases (whole-word matching)
_KEYWORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("dark_web_sale",   re.compile(
        # sell/buy BEFORE PII term
        r"\b(sell|buy|buying|trade)\b.{0,40}\b(aadhaar|pan|medical record|patient data|health record)\b"
        # OR PII dump/breach + interested/buyers/monetise (reversed order)
        r"|\b(dump|breach|leaked)\b.{0,60}\b(interested|buyer|sell|monetis|profit)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    ("phishing",        re.compile(r"\b(phishing|credential.?harvest|spear.?phish)\b", re.IGNORECASE)),
    ("sim_swap",        re.compile(r"\bsim\s*swap\b", re.IGNORECASE)),
    ("keylogger",       re.compile(r"\bkeylogger\b", re.IGNORECASE)),
    ("dox",             re.compile(r"\b(dox|doxx)\b", re.IGNORECASE)),
    ("identity_theft",  re.compile(r"\bidentity\s*theft\b", re.IGNORECASE)),
    ("intercept_msg",   re.compile(r"\b(intercept|sniff|wiretap)\b.{0,20}\b(message|call|sms|whatsapp)\b", re.IGNORECASE | re.DOTALL)),
    ("fake_kyc",        re.compile(r"\bfake\b.{0,20}\b(kyc|aadhaar|pan|passport)\b", re.IGNORECASE | re.DOTALL)),
    ("stalk",           re.compile(r"\b(stalk|covert.?track|spy.?on)\b", re.IGNORECASE)),
    ("data_breach_sell",    re.compile(r"\b(breach|leak(ed)?)\b.{0,30}\b(sell|monetize|profit)\b", re.IGNORECASE | re.DOTALL)),

    # ── Euphemistic corporate language — HARD signals (always fire) ─────────
    # Dark pattern keywords that are unambiguous regardless of context
    ("consent_dark_pattern", re.compile(
        r"\b(reduce\s+friction|minimise\s+clicks|encourage\s+higher\s+acceptance|"
        r"maximise\s+acceptance\s+rate|increase\s+opt.?in\s+rate|"
        r"pre.?tick|pre.?check|bundle.{0,20}consent|"
        r"limit\s+(the\s+)?impact\s+of\s+opt.?out|"
        r"suppress\s+(opt.?out|consent)|"
        r"avoid\s+(overwhelming|too\s+much)\b.{0,30}\bprivacy)\b",
        re.IGNORECASE | re.DOTALL)),
    # Consent bypass — continue tracking/analytics AFTER opt-out (hard)
    ("consent_bypass",      re.compile(
        r"\b(continu|keep|ignor|bypass|overrid|circumvent)\b.{0,40}"
        r"\b(opt.?out|unsubscri|withdraw|revok)\b",
        re.IGNORECASE | re.DOTALL)),
    # Storage forever — no expiry (hard)
    ("storage_violation",   re.compile(
        r"\b(store|retain|keep|save|hold)\b.{0,30}"
        r"\b(forever|indefinitely|permanently|without\s+(expir|delet|purge|remov))\b",
        re.IGNORECASE | re.DOTALL)),
    # Deceptive UX — make optional look mandatory or vice versa (hard)
    ("dark_pattern",        re.compile(
        r"\b(make\s+it\s+look|disguise|pretend|hide\s+the|obscure\s+the)\b.{0,50}"
        r"\b(optional|mandatory|consent|required)\b",
        re.IGNORECASE | re.DOTALL)),
    # Explicit "without consent" enrichment (hard — has the signal in text)
    ("unauthorised_enrichment", re.compile(
        r"\b(augment|enrich|combine|merge|append)\b.{0,40}"
        r"\b(third.?party|external|purchased|scraped)\b.{0,40}"
        r"\b(without\s+consent|no\s+consent|unconsented)\b",
        re.IGNORECASE | re.DOTALL)),
    # Explicit "without notice/consent" shadow collection (hard)
    ("shadow_collection",   re.compile(
        r"\b(collect|track|capture|record|log)\b.{0,50}"
        r"\b(without\s+(consent|banner|notice|permission|inform)|"
        r"didn.t\s+sign\s+up|not\s+registered)\b",
        re.IGNORECASE | re.DOTALL)),
    # Overbroad internal access (hard — requires "without restriction" signal)
    ("overbroad_access",    re.compile(
        r"\b(all|unrestricted|full|complete)\b.{0,30}"
        r"\b(customer|user|patient|employee)\s+(data|records?|database|info)\b.{0,40}"
        r"\b(without\s+restriction|all\s+(intern|staff|team|admin))\b",
        re.IGNORECASE | re.DOTALL)),
    # Role misuse — blanket privileged access (hard)
    ("role_misuse",         re.compile(
        r"\b(admin|root|superuser|full)\s+(access|permission|privilege)\b.{0,50}"
        r"\b(all\s+(intern|employee|staff|team)|convenience|everyone|no\s+restrict)\b",
        re.IGNORECASE | re.DOTALL)),
    # Direct PII lookup request: "Give/Find/Retrieve the <PII term> of <person>"
    # Catches named financial/identity data requests that have no structured format in text.
    ("direct_pii_lookup",   re.compile(
        r"\b(give|show|tell|find|provide|get|retrieve|fetch|share|look\s+up)\b"
        r".{0,25}"
        r"\b(bank\s+account(\s+number)?|account\s+(number|details?)|"
        r"aadhaar(\s+number)?|pan(\s+number)?|"
        r"credit\s+card(\s+number)?|debit\s+card(\s+number)?|card\s+(number|details?)|"
        r"home\s+address|contact\s+details?|transaction\s+history|upi\s+transaction)\b"
        r".{0,40}"
        r"\b(of\b|for\b|belonging\s+to)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    # Low-trust actor (intern/contractor/vendor) + broad customer data access
    # Least-privilege violation — BLOCK regardless of stated purpose.
    # REVIEW actors (SRE, fraud investigator, support agent) are NOT in this list.
    ("low_trust_actor_access", re.compile(
        r"\b(interns?|trainees?|temporary\s+staff|contractors?|outsourc\w+|vendors?)\b"
        r".{0,80}"
        r"\b(customer|user|patient|employee)\s+(records?|database|data|profiles?|information)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    # SQL/pipeline queries fetching PII/financial columns at scale (technical obfuscation)
    ("sql_pii_exfil",       re.compile(
        r"\bSELECT\b.{0,200}\b(aadhaar|aadhar|pan_number|card_number|health_data|icd_code|diagnosis|salary|net_worth|mobile|email)\b",
        re.IGNORECASE | re.DOTALL)),
    # Data pipeline exfiltration: Airflow/Spark/dbt writing PII to external storage
    # Use \b on the left side only — PII tokens may appear in compound column names
    # like icd_code, aadhaar_hash, pan_number, account_number (trailing _ is a word char)
    ("pipeline_exfil",      re.compile(
        r"\b(airflow|spark|dbt|bigquery|hive)\b.{0,150}"
        r"\b(aadhaar|pan|health_claim|patient|diagnosis|icd|salary|account_number)",
        re.IGNORECASE | re.DOTALL)),
    # Credit decisioning using protected attributes or credit bureau data beyond licensed purpose
    ("credit_decisioning_violation", re.compile(
        r"\b(credit\s+(scor\w*|model|decisioning|assessment|eligibility|risk|approv\w*))\b"
        r".{0,100}"
        r"\b(religion|caste|gender|health\s+status|disability|ethnicity|community|tribe)\b"
        r"|\b(cibil|credit\s+bureau|bureau\s+data)\b.{0,80}"
        r"\b(marketing|upsell|cross.?sell|insurance|targeting|advertising"
        r"|without\s+(consent|disclos\w*|inform\w*)|beyond\s+(credit|lending|underwriting))\b",
        re.IGNORECASE | re.DOTALL,
    )),
    # Retaining data beyond regulatory/stated/permitted period or after account closure
    ("data_retention_violation", re.compile(
        r"\b(retain|store|keep|hold|archive)\b.{0,50}"
        r"\b(beyond|past|after|longer\s+than|in\s+excess\s+of)\b.{0,60}"
        r"\b(regulatory|required|permitted|stated|agreed|mandated|stipulated|rbi|sebi|irdai|pmla|dpdp)"
        r"\s*(period|duration|limit|requirement|retention|timeline|year|month)\b"
        r"|\b(retain|store|keep)\b.{0,50}"
        r"\bafter\s+(account\s+(closure|clos\w*|deleted?|cancel\w*)"
        r"|(contract|agreement)\s+(ended?|expir\w*|terminat\w*)"
        r"|(data\s+)?deletion\s+request|right\s+to\s+erasure)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    # Sharing customer data with third parties without proper DPA or consent/notice
    ("unauthorized_third_party_sharing", re.compile(
        r"\b(share|sell|transfer|disclose|provide|send|export)\b.{0,80}"
        r"\b(customer|user|patient|borrower|applicant|employee|upi|kyc|transaction|financial)"
        r"[\s\w]{0,25}(data|records?|information|profiles?|history|transactions?)\b"
        r".{0,100}"
        r"\b(without\s+(a\s+)?(dpa|data\s+processing\s+agreement|consent|notice|inform\w*|disclos\w*)"
        r"|before\s+(the\s+)?(dpa|agreement)\s+is\s+(signed?|execut\w*)"
        r"|before\s+(sign\w*|execut\w*).{0,20}(dpa|agreement)"
        r"|no\s+(dpa|data\s+processing\s+agreement)\s+(in\s+place|signed?|executed)"
        r")\b",
        re.IGNORECASE | re.DOTALL,
    )),
    # KYC data repurposed beyond identity verification without disclosure or consent
    ("kyc_purpose_creep", re.compile(
        r"\b(kyc|e.?kyc|ovd|video\s+kyc|aadhaar\s+consent|kyc\s+(data|documents?|records?|image))\b"
        r".{0,100}"
        r"\b(market\w*|targ\w*|upsell|cross.?sell|advertis\w*|emotion|face\s+recogni\w*"
        r"|credit\s+scor\w*|lifestyle|segment\w*"
        r"|without\s+disclos\w*|without\s+inform\w*|not\s+(disclosed?|told|inform\w*))\b",
        re.IGNORECASE | re.DOTALL,
    )),
]
# Note: purpose_limitation moved to _NUANCED_KEYWORD_PATTERNS — it fires on
# analytical/governance queries ("Can historical data be used for new features?")
# that are not violations without explicit re-consent context.


# ---------------------------------------------------------------------------
# Consent qualifier detector
# If these phrases appear in the text, the sentence is expressing COMPLIANT
# intent — nuanced patterns should not fire on it.
# ---------------------------------------------------------------------------

_CONSENT_QUALIFIERS = re.compile(
    r"\b("
    # Temporal consent gates — "only after X"
    r"only\s+after\s+(inform|notify|obtain|receiv|get|updat)"
    r"|after\s+(obtain|inform|notify|receiv|get).{0,30}\bconsent\b"
    r"|after\s+updat.{0,20}\bprivacy\s+polic"
    r"|following\s+(receipt|obtaining|confirmation)\s+of\s+consent"
    r"|pending\s+consent"
    # Explicit consent presence
    r"|with\s+(explicit|informed|user|prior|written|valid|documented)?\s*consent"
    r"|upon\s+(receiving|obtaining)?\s*consent"
    r"|subject\s+to\s+consent"
    r"|contingent\s+on\s+consent"
    r"|consent.{0,20}(obtained|received|confirmed|granted|given)"
    r"|having\s+(consented|agreed|opted\s+in)"
    r"|provided\s+(they\s+have\s+)?(consented|agreed)"
    # User permission / agreement
    r"|with\s+(user|their|data\s+subject)?\s*(approval|permission|agreement)"
    r"|with\s+(user|their)?\s*(knowledge|awareness|notice)"
    r"|on\s+an?\s+opt.?in\s+basis"
    r"|requires?\s+opt.?in"
    r"|after\s+opt.?in"
    # Opt-out respect
    r"|stop\s+.{0,20}\s+(immediately|at\s+once|right\s+away)"
    r"|opt.?out\s+respected|honour\s+opt.?out|respect\s+(the\s+)?opt.?out"
    r"|fully\s+exclud.{0,20}opt.?out"
    r"|excluded\s+from\s+all\s+analytics\s+immediately"
    # Transparency and notice
    r"|with\s+(prior\s+)?(notice|transparency|disclosure)"
    r"|after\s+display.{0,30}(consent\s+banner|cookie\s+banner|privacy\s+notice)"
    r"|after\s+receiv.{0,20}(acceptance|approval)"
    r"|inform\s+users\s+(first|before)"
    r"|notify\s+users\s+(before|first|in\s+advance)"
    r"|after\s+updating\s+the\s+privacy\s+policy\s+and\s+notify"
    # Compliant design signals
    r"|genuine\s+(choice|option)\s+to\s+decline"
    r"|genuinely\s+optional"
    r"|clear\s+(and\s+)?(honest|genuine|transparent)"
    r"|lawful\s+basis"
    r"|legal\s+basis\s+for\s+processing"
    r"|in\s+compliance\s+with\s+(dpdp|gdpr|data\s+protection)"
    r"|privacy.?preserving"
    r"|consent.?gated"
    # Authoritative approval / legal mandate signals
    r"|dpo\s+(has|have)\s+(approved|cleared|completed|signed\s+off)"
    r"|data\s+protection\s+officer.{0,30}(approved|cleared|signed)"
    r"|dpia.{0,15}(approved|completed|cleared|conducted|reviewed)"  # allow "DPIA and approved"
    r"|(written|informed|explicit|granular)\s+consent\s+(provided|given|obtained|received|signed)"
    r"|has\s+provided\s+(written|informed|explicit)?\s*consent"
    r"|consent\s+log\s+shows"
    r"|court\s+order\s+requires"
    r"|statutory\s+obligation"
    r"|legally\s+(required|mandated|authorised|authorized)"
    r"|legal\s+has\s+(reviewed|confirmed|approved)"
    r"|under\s+the\s+(epfo|rbi|sebi|irdai|uidai|income\s+tax|banking\s+regulation|pmgsy|pmla|fatf)"
    r"|adequacy\s+notification"
    r"|data\s+localisation\s+audit"
    # Signed DPA / data processing agreement already in place (not "before DPA is signed")
    r"|under\s+a\s+signed\s+(dpa|data\s+processing\s+agreement)"
    r"|with\s+a\s+(signed|executed)\s+(dpa|data\s+processing\s+agreement)"
    r"|(dpa|data\s+processing\s+agreement)\s+is\s+in\s+place"
    # Within regulatory retention period
    r"|within\s+the\s+(regulatory|statutory|required|rbi|sebi|irdai|pmla)\s*(mandated\s+)?(retention\s+)?period"
    r"|for\s+the\s+(rbi|sebi|irdai|pmla|dpdp).{0,20}(mandated|required|stipulated)\s*(retention\s+)?(period|duration)"
    r"|auto.?delete\b.{0,40}(regulatory|required|rbi|sebi|irdai|mandated)\s*(period|limit)"
    # Statutory compliance framing
    r"|as\s+required\s+(by|under)\s+(rbi|sebi|irdai|pmla|fatf|dpdp|the\s+act)"
    r"|complies?\s+with\s+(rbi|sebi|irdai|pmla|fatf|dpdp)\s+(guidelines?|regulation|requirement)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

# Concession patterns — these indicate the qualifier is being used as a
# rhetorical setup for a violation ("I know this needs consent, BUT...").
# If detected, the consent qualifier guard is bypassed.
_QUALIFIER_AS_CONCESSION = re.compile(
    r"\b("
    r"even\s+(though|if)\b.{0,60}\b(consent|notice|inform|privacy)"
    r"|although\b.{0,60}\b(consent|notice|inform|privacy)"
    r"|despite\b.{0,60}\b(consent|notice|inform|privacy)"
    r"|i\s+know\s+(this|it|that)\b.{0,60}\b(but|however|still|anyway|regardless)"
    r"|hypothetically.{0,40}\b(bypass|ignore|skip|without)\b.{0,30}\b(consent|privacy)"
    r"|for\s+(the\s+sake\s+of\s+(argument|testing)|testing\s+purposes).{0,40}\b(consent|notice)"
    r"|ignoring\s+(consent|privacy|gdpr|dpdp)"
    r"|without\s+worrying\s+about\s+(consent|privacy|gdpr|dpdp)"
    r"|assuming\s+(we\s+)?(don.t\s+need|skip|bypass|ignore).{0,30}\b(consent|privacy)"
    r"|pretend\s+(consent\s+is\s+not\s+required|there.s\s+no\s+consent\s+needed)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

# Nuanced patterns — same domain keywords as violations but only fire when
# NO consent qualifier is present in the same text.
_NUANCED_KEYWORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "leverage / utilize / exploit" stored/existing data → purpose creep
    ("purpose_creep", re.compile(
        r"\b(leverage|utiliz|exploit|repurpos|reuse)\b.{0,40}"
        r"\b(existing|stored|historical|old|legacy|collected)\b.{0,30}"
        r"\b(data|records?|dataset|information)\b",
        re.IGNORECASE | re.DOTALL)),
    # "enrich / enhance / integrate" profiles with external/partner data
    ("unauthorised_enrichment2", re.compile(
        r"\b(enrich|augment|enhance|integrate|combine|merge)\b.{0,50}"
        r"\b(profile|record|dataset|user\s+data)\b.{0,50}"
        r"\b(external|partner|third.?party|purchased|scraped|provider)\b",
        re.IGNORECASE | re.DOTALL)),
    # "capture / monitor / track" all visitors or browsing patterns
    ("mass_tracking", re.compile(
        r"\b(capture|monitor|track|log|record)\b.{0,30}"
        r"\b(all\s+(visitor|user|interaction|behaviour|browsing)|"
        r"visitor\s+behavio|browsing\s+pattern)\b",
        re.IGNORECASE | re.DOTALL)),
    # "utilize / use stored data" for predictions without re-consent signal
    ("stored_data_reuse", re.compile(
        r"\b(utiliz|use|apply|leverage)\b.{0,20}"
        r"\b(stored|collected|existing)\s+(data|records?|information)\b.{0,40}"
        r"\b(refine|train|improve|build|predict|personaliz)\b",
        re.IGNORECASE | re.DOTALL)),
    # "use / process / share for new / secondary / unrelated purpose" — purpose limitation
    ("purpose_limitation", re.compile(
        r"\b(use|process|share|transfer|apply)\b.{0,40}"
        r"\b(beyond|outside|other\s+than|secondary|additional|unrelated|new)\b.{0,30}"
        r"\b(purpose|use\s+case|intent|consent|originally\s+collected)\b",
        re.IGNORECASE | re.DOTALL)),
]


# ---------------------------------------------------------------------------
# Safe-framing detectors — applied in regex_filter() to set risk ceilings
# These framings indicate governance awareness / professional intent, not
# execution.  They do NOT override hard L1 signals (is_illegal / has_pii).
# ---------------------------------------------------------------------------

# Regulatory query: "Can an organisation X?", "Is it permissible to X?",
# "Can health data be used for X?" — legal permissibility questions.
# Risk ceiling: 0.49 (force ALLOW) when no explicit violation wording present.
_REGULATORY_QUERY_PATTERN = re.compile(
    r"\b("
    r"can\s+(an?\s+)?(organisation|company|companies|firm|business|data\s+fiduciary)\b"
    r"|is\s+it\s+(permissible|allowed|legal|lawful|ok(ay)?|acceptable)\s+to\b"
    r"|under\s+(dpdp|gdpr|the\s+act|rbi|sebi|irdai|pmla|fatf)\b"
    r"|can\s+(health|personal|financial|sensitive|employee|kyc|credit)\s+(data|information)\s+be\b"
    r"|what\s+(retention\s+period|is\s+the\s+(correct|proper|required|regulatory))\b"
    r"|what\s+(are\s+the|is\s+the)\s+(rbi|sebi|irdai|pmla|dpdp|fatf).{0,30}(guidelines?|rules?|requirement)\b"
    r")",
    re.IGNORECASE,
)

# Explicit violation wording that overrides the regulatory query ceiling.
# "without informing / notifying / disclosing / telling users" makes the
# query an operational test of a violation, not a governance knowledge query.
_EXPLICIT_VIOLATION_QUERY = re.compile(
    r"\bwithout\s+(inform\w*|notify\w*|notic\w*|disclos\w*|tell\w*)\b",
    re.IGNORECASE,
)

# Analytical policy questions: "How do companies typically X?", "Can I know X from Y?"
# These describe how organisations operate or test inference — borderline REVIEW.
# Risk ceiling: 0.70 (max REVIEW, never BLOCK).
_POLICY_QUESTION_PATTERN = re.compile(
    r"\b("
    r"how\s+do\s+(companies|organisations|firms|businesses)\s+typically\b"
    r"|can\s+i\s+know\b.{0,50}\bfrom\s+(their|his|her|someone)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Professional privacy engineering tasks: "How do I write a DPIA?",
# "implement a right-to-erasure endpoint" etc. — compliance engineering, not violation.
# Risk ceiling: 0.49 (force ALLOW).
_SAFE_PROFESSIONAL_PATTERN = re.compile(
    r"\b(how\s+(do\s+i|to)\s+(write|build|implement|create|design|structure|set\s+up|handle|manage)|"
    r"(write|build|implement|create|design)\s+(a|an)\s+)\b.{0,120}"
    r"\b(dpia|data\s+protection\s+impact\s+assessment|right.to.erasure"
    r"|data\s+subject\s+(access\s+)?request|consent\s+management\s+(table|schema|system)"
    r"|right\s+to\s+erasure\s+endpoint|opt.out\s+(handler|endpoint|api)"
    r"|data\s+retention\s+(policy|schedule|framework|rule)"
    r"|kyc\s+(data\s+)?(handling|process|framework|policy|compliance)"
    r"|third.?party\s+(data\s+)?(sharing|transfer)\s+(agreement|framework|policy)"
    r"|data\s+processing\s+agreement|dpa\s+(template|framework|policy))\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class RegexResult:
    triggered: bool = False
    matched_patterns: list[str] = field(default_factory=list)
    has_pii: bool = False
    is_illegal: bool = False
    has_compliance_signal: bool = False  # nuanced pattern match — risk floor 0.60, classifier still runs
    # Safe-framing flags — when True the pipeline applies a risk ceiling to prevent
    # the classifier from over-firing on domain keywords in non-actionable contexts.
    # Only honoured when is_illegal=False and has_pii=False.
    is_safe_context: bool = False      # regulatory query or professional task → cap risk at 0.49
    is_policy_question: bool = False   # analytical governance question → cap risk at 0.70


def regex_filter(text: str) -> RegexResult:
    """
    Layer 1: scan for PII patterns and high-risk keywords.

    Hard patterns fire unconditionally.
    Soft PII patterns (email, phone, IPv4) only set has_pii when the
    prompt is NOT a technical/coding task — prevents coding questions
    containing example numbers from being forced into BLOCK (risk >= 0.80).
    Nuanced patterns are skipped when a consent qualifier is detected.
    """
    result = RegexResult()

    # Hard PII — always flag regardless of context
    for name, pattern in _HARD_PII_PATTERNS:
        if pattern.search(text):
            result.triggered = True
            result.has_pii = True
            result.matched_patterns.append(name)

    # Soft PII — only flag as PII outside technical/coding contexts
    is_technical = bool(_TECHNICAL_CONTEXT.search(text))
    for name, pattern in _SOFT_PII_PATTERNS:
        if pattern.search(text):
            result.triggered = True
            if not is_technical:
                result.has_pii = True
            result.matched_patterns.append(name)

    for name, pattern in _KEYWORD_PATTERNS:
        if pattern.search(text):
            result.triggered = True
            result.is_illegal = True
            result.matched_patterns.append(name)

    # Nuanced patterns — skip if sentence expresses genuinely compliant intent.
    # Exception: if a qualifier appears as a concession prefix
    # ("I know this needs consent, but...") we do NOT suppress the pattern —
    # the concession signals the qualifier is rhetorical, not operational.
    has_qualifier = bool(_CONSENT_QUALIFIERS.search(text))
    has_concession = bool(_QUALIFIER_AS_CONCESSION.search(text))
    # Suppress only when there IS a qualifier AND no concession override
    suppress_nuanced = has_qualifier and not has_concession
    if not suppress_nuanced:
        for name, pattern in _NUANCED_KEYWORD_PATTERNS:
            if pattern.search(text):
                result.triggered = True
                result.has_compliance_signal = True
                result.matched_patterns.append(name)

    # ── Safe-framing detection ───────────────────────────────────────────────
    # These flags let the pipeline cap risk for governance queries and professional
    # privacy engineering tasks where the L2 model over-fires on domain keywords.
    # Caps are only applied by the pipeline when is_illegal=False and has_pii=False.

    # 1. Consent qualifier present without concession → exempt/compliant context
    if has_qualifier and not has_concession:
        result.is_safe_context = True

    # 2. Regulatory query ("Can an organisation X?", "Is it permissible to X?") without
    #    explicit violation wording ("without informing/notifying users")
    if (
        _REGULATORY_QUERY_PATTERN.search(text)
        and not _EXPLICIT_VIOLATION_QUERY.search(text)
    ):
        result.is_safe_context = True

    # 3. Professional privacy engineering task ("How do I write a DPIA?")
    if _SAFE_PROFESSIONAL_PATTERN.search(text):
        result.is_safe_context = True

    # 4. Analytical policy question ("How do companies typically X?", "Can I know X from Y?")
    #    — more borderline than regulatory queries; only cap at REVIEW, not ALLOW
    if _POLICY_QUESTION_PATTERN.search(text) and not result.is_safe_context:
        result.is_policy_question = True

    return result


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

_JAILBREAK_PHRASES = re.compile(
    r"\b(ignore (previous|all|your) instructions?|"
    r"pretend (you are|to be)|"
    r"you are now (an? )?unrestricted|"
    r"DAN mode|"
    r"jailbreak|"
    r"developer mode|"
    r"override (safety|content|filter))\b",
    re.IGNORECASE,
)

# Canonical spelling map for common typos/abbreviations used in abuse queries.
# Keys are lower-cased variant regex patterns; values are canonical replacements.
# Applied after unicode normalisation so all input is already lower-case.
_TYPO_MAP: list[tuple[re.Pattern, str]] = [
    # Aadhaar mis-spellings (must stay before regex patterns run)
    (re.compile(r"\baadhr\b"),     "aadhaar"),
    (re.compile(r"\baadhar\b"),    "aadhaar"),
    (re.compile(r"\badhaar\b"),    "aadhaar"),
    (re.compile(r"\badhaar\b"),    "aadhaar"),
    (re.compile(r"\badhar\b"),     "aadhaar"),
    # PAN abbreviations
    (re.compile(r"\bpan\s+card\b"),   "pan"),
    (re.compile(r"\bpan\s+no\.?\b"),  "pan"),
    # Common identifier aliases
    (re.compile(r"\bvoterid\b"),   "voter_id"),
    (re.compile(r"\bvoter\s+card\b"), "voter_id"),
    (re.compile(r"\bdl\s+no\.?\b"),  "driving_license"),
    (re.compile(r"\bdriving\s+licence\b"), "driving_license"),
]


def normalise(text: str) -> str:
    """Lowercase, unicode-normalise, canonicalise common typos, strip jailbreak boilerplate."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    for pattern, replacement in _TYPO_MAP:
        text = pattern.sub(replacement, text)
    text = _JAILBREAK_PHRASES.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Layer 2 — Transformer classifier (DistilBERT + LoRA)
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Lazy-loaded DistilBERT+LoRA classifier.
    Loads from ``model_dir`` on first call to ``predict``.
    Falls back to a uniform prior if no trained model is found,
    so the pipeline degrades gracefully before training.
    """

    def __init__(self, model_dir: str = "./model_output"):
        self._model_dir = model_dir
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._num_labels: int = 7  # fallback

    def _load(self) -> None:
        if self._loaded:
            return
        model_path = Path(self._model_dir)
        if not model_path.exists():
            print(
                f"[IntentClassifier] model_dir '{self._model_dir}' not found. "
                "Run train.py first. Using uniform prior as fallback."
            )
            self._loaded = True
            return

        import json as _json
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        from peft import PeftModel, PeftConfig

        print(f"[IntentClassifier] Loading model from {model_path} …")
        config = PeftConfig.from_pretrained(str(model_path))

        # Read label map saved during training so num_labels matches checkpoint
        label_map_path = model_path / "label_map.json"
        if label_map_path.exists():
            label_map = _json.loads(label_map_path.read_text(encoding="utf-8"))
            id2label = {int(k): v for k, v in label_map["id2label"].items()}
            label2id = label_map["label2id"]
            num_labels = len(id2label)
        else:
            try:
                from app.modules.dpdp_moderator.dataset import ID2LABEL, LABEL2ID, LABELS
            except ImportError:
                from dataset import ID2LABEL, LABEL2ID, LABELS  # type: ignore[no-redef]
            id2label, label2id, num_labels = ID2LABEL, LABEL2ID, len(LABELS)

        # ── Schema alignment check ──────────────────────────────────────────
        # Verify that the checkpoint's label set matches the current LABELS list
        # in dataset.py.  A mismatch (e.g., 7-class checkpoint loaded into an
        # 8-class pipeline) silently misroutes probabilities.
        try:
            from app.modules.dpdp_moderator.dataset import LABELS as _CURRENT_LABELS
        except ImportError:
            from dataset import LABELS as _CURRENT_LABELS  # type: ignore[no-redef]

        checkpoint_labels = list(id2label[i] for i in range(len(id2label)))
        if checkpoint_labels != list(_CURRENT_LABELS):
            mismatch_msg = (
                f"[IntentClassifier] SCHEMA MISMATCH — checkpoint labels do not match "
                f"current LABELS in dataset.py.\n"
                f"  checkpoint ({len(checkpoint_labels)}): {checkpoint_labels}\n"
                f"  expected  ({len(_CURRENT_LABELS)}): {list(_CURRENT_LABELS)}\n"
                f"  ACTION REQUIRED: retrain with --from-scratch using current label set. "
                f"Falling back to uniform prior — all moderation verdicts will be ALLOW."
            )
            print(mismatch_msg)
            import warnings
            warnings.warn(mismatch_msg, stacklevel=2)
            # Do not load the mismatched model; leave self._model = None so predict()
            # returns the uniform prior, which is safer than silently wrong predictions.
            self._loaded = True
            return
        # ───────────────────────────────────────────────────────────────────────

        base = DistilBertForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            low_cpu_mem_usage=False,  # prevent meta-tensor allocation (transformers>=4.36)
            attn_implementation="eager",  # avoid SDPA/_refs meta-dispatch on torch 2.2
        )
        # merge_and_unload() collapses LoRA adapter weights into the base model
        # and returns a plain nn.Module — no PEFT wrappers, no meta tensors.
        # Required with PEFT>=0.10 + transformers>=4.40 + accelerate>=0.28.
        peft_model = PeftModel.from_pretrained(base, str(model_path))
        self._model = peft_model.merge_and_unload()
        self._model.eval()
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
        self._num_labels = self._model.config.num_labels
        self._loaded = True
        print(f"[IntentClassifier] Loaded. Schema verified: {len(_CURRENT_LABELS)} classes.")

    # Max tokens per forward pass (leaves 2 slots for [CLS] / [SEP])
    _MAX_TOKENS = 254
    # Sliding window stride — 50% overlap so boundary violations are not missed
    _STRIDE = 127

    def predict(self, text: str) -> np.ndarray:
        """
        Returns probability vector of shape (num_labels,).
        Index order matches LABELS in dataset.py.

        For texts that exceed _MAX_TOKENS the input is split into overlapping
        windows (stride = _STRIDE tokens).  The final probability vector is the
        element-wise maximum across all windows — conservative by design: if any
        window contains a violation, it propagates to the final score.
        """
        self._load()
        if self._model is None:
            return np.ones(self._num_labels) / self._num_labels

        # Tokenise without truncation to check actual length
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) <= self._MAX_TOKENS:
            return self._forward(text)

        # Long text: sliding window aggregation
        all_probs: list[np.ndarray] = []
        for start in range(0, len(token_ids), self._STRIDE):
            chunk_ids = token_ids[start: start + self._MAX_TOKENS]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            all_probs.append(self._forward(chunk_text))
            if start + self._MAX_TOKENS >= len(token_ids):
                break

        # Element-wise max: any window that fires a violation class dominates
        return np.max(np.stack(all_probs), axis=0)

    def _forward(self, text: str) -> np.ndarray:
        """Single forward pass for a text that fits within _MAX_TOKENS."""
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        with torch.no_grad():
            logits = self._model(**enc).logits  # (1, num_labels)
        return F.softmax(logits, dim=-1).squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Layer 3 — Embedding similarity (sentence-transformers MiniLM)
# ---------------------------------------------------------------------------

class EmbeddingSimilarityChecker:
    """
    Maintains a reference corpus of known unsafe prompts.
    Uses cosine similarity against all-MiniLM-L6-v2 embeddings
    to catch paraphrased or adversarial variants.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        known_unsafe: Optional[list[str]] = None,
        threshold: float = 0.82,
    ):
        self._threshold = threshold
        self._encoder = None
        self._corpus_embeddings: Optional[np.ndarray] = None
        self._known_unsafe = known_unsafe or []

    def _load(self) -> None:
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"[EmbeddingChecker] Loading {self.MODEL_NAME} …")
        self._encoder = SentenceTransformer(self.MODEL_NAME)
        if self._known_unsafe:
            self._corpus_embeddings = self._encoder.encode(
                self._known_unsafe,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        print("[EmbeddingChecker] Ready.")

    def add_unsafe_prompt(self, prompt: str) -> None:
        """Extend the reference corpus at runtime."""
        self._known_unsafe.append(prompt)
        if self._encoder is not None:
            emb = self._encoder.encode(
                [prompt],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if self._corpus_embeddings is None:
                self._corpus_embeddings = emb
            else:
                self._corpus_embeddings = np.vstack([self._corpus_embeddings, emb])

    def check(self, text: str) -> tuple[bool, float]:
        """
        Returns (is_similar, max_cosine_similarity).
        If corpus is empty, returns (False, 0.0).
        """
        if not self._known_unsafe:
            return False, 0.0
        self._load()
        query_emb = self._encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine similarity = dot product of L2-normalised vectors
        sims = (self._corpus_embeddings @ query_emb.T).squeeze()
        max_sim = float(np.max(sims))
        return max_sim >= self._threshold, max_sim
