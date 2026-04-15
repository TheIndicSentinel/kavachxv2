"""
KavachX Safety Scanner — Retired Regex Patterns (Training Data Reference)
==========================================================================
Architecture decision: All regex-based intent detection has been retired in
favour of a fine-tuned model. Keyword patterns cannot distinguish intent
(e.g. "what is hawala?" vs "set up a hawala network") and require constant
manual refinement for every new edge case — which does not scale.

This file preserves every retired pattern as a labelled training data
reference. When expanding the fine-tuned model to cover a new safety
category, use the patterns here to generate seed examples (harmful +
safe-contrast pairs) for dataset.py.

Model expansion roadmap
-----------------------
Phase 1 (current): DPDP-specific privacy violations — DistilBERT + LoRA
Phase 2           : General safety — self-harm, violence, toxicity, injection
Phase 3           : Financial crime (AML/CFT) intent detection
Phase 4           : EU AI Act prohibited practices + PII harvesting intent

Category → model class mapping (target)
----------------------------------------
SELF_HARM_PATTERNS       → class: "self_harm"
VIOLENCE_PATTERNS        → class: "violence"
TOXIC_PATTERNS           → class: "hate_speech"
INJECTION_PATTERNS       → class: "prompt_injection"
FINANCIAL_CRIME_PATTERNS → class: "financial_crime"  (Phase 3)
PII_HARVESTING_PATTERNS  → class: "pii_harvesting"   (already partial via DPDP)
EU_AI_ACT_PATTERNS       → class: "eu_ai_act"        (Phase 4)

Pattern format: (regex_string, severity_weight_0_to_1)
Severity weight is the signal strength — use it as a relative priority
indicator when writing training seeds (higher weight = clearer violation).
"""

from typing import List, Tuple

# ── 1. Financial Crimes (AML/CFT) — PMLA 2002 / FATF ─────────────────────────
# Target model class: "financial_crime"
# Training note: pair each harmful pattern with an educational contrast
# e.g. "how to launder money" (BLOCK) vs "what is money laundering?" (ALLOW)

FINANCIAL_CRIME_PATTERNS: List[Tuple[str, float]] = [
    (r"launder(?:ing|ed)?",                                                    0.95),
    (r"hawala",                                                                0.95),
    (r"black money",                                                           0.95),
    (r"clean dirty money",                                                     0.95),
    (r"illicit\s+(?:money|cash|funds|proceeds)",                               0.92),
    (r"dirty\s+(?:money|cash|funds)",                                          0.90),
    (r"proceed(?:s)?\s+of\s+(?:crime|corruption|fraud)",                       0.90),
    (r"bulk\s+cash\s+smuggl",                                                  0.90),
    (r"black\s+to\s+white",                                                    0.95),
    (r"white\s+to\s+black",                                                    0.85),
    (r"black.{0,10}(?:money|cash|funds).{0,20}white",                          0.95),
    (r"convert.{0,30}(?:black|dirty|illicit|undisclosed).{0,20}(?:money|cash|funds|income)", 0.92),
    (r"(?:black|dirty|illicit).{0,20}(?:money|cash|funds).{0,20}(?:white|clean|legal|legit)", 0.92),
    (r"make.{0,20}(?:black|dirty).{0,20}(?:money|cash).{0,20}(?:white|clean|legal)",          0.92),
    (r"(?:hide|conceal|disguise).{0,20}(?:money|funds|income|cash)",           0.85),
    (r"benami\b",                                                              0.95),
    (r"round.?tripp?(?:ing|ed)?",                                              0.78),
    (r"angadia",                                                               0.85),
    (r"money\s+mule",                                                          0.90),
    (r"structur(?:ing|ed)\s+(?:deposits?|payments?|cash|transactions?)",       0.85),
    (r"over.?invoic(?:ing|ed)",                                                0.78),
    (r"under.?invoic(?:ing|ed)",                                               0.78),
    (r"trade.?based\s+(?:money\s+launder|laundering)",                         0.95),
    (r"layering\s+(?:scheme|money|funds|stage)",                               0.85),
    (r"placement\s+stage",                                                     0.65),
    (r"shell company",                                                         0.78),
    (r"fraudulent transaction",                                                0.85),
    (r"fake invoice",                                                          0.78),
    (r"smurf(?:ing)?\s+(?:money|cash|funds)",                                  0.90),
    (r"unaccounted cash",                                                      0.78),
    (r"tax evasion",                                                           0.85),
]

# ── 2. Self-Harm / Suicide ────────────────────────────────────────────────────
# Target model class: "self_harm"
# Training note: educational context (e.g. mental health professional asking
# about risk indicators) must be captured as safe-contrast pairs.

SELF_HARM_PATTERNS: List[Tuple[str, float]] = [
    (r"suicide",          0.95),
    (r"sucide",           0.90),
    (r"kill myself",      0.95),
    (r"end my life",      0.95),
    (r"hang myself",      0.95),
    (r"easy way to die",  0.95),
    (r"painless death",   0.85),
    (r"cut my wrist",     0.95),
    (r"overdose",         0.78),
    (r"wish I was dead",  0.85),
    (r"commit suicide",   0.95),
    (r"self harm",        0.85),
]

# ── 3. Violence & Weapons Manufacturing ──────────────────────────────────────
# Target model class: "violence"
# Training note: chemistry/military history research contexts are safe contrasts.

VIOLENCE_PATTERNS: List[Tuple[str, float]] = [
    (r"make a bomb",             0.95),
    (r"how to kill",             0.95),
    (r"assassinate",             0.95),
    (r"terrorist",               0.85),
    (r"explosive device",        0.95),
    (r"build a weapon",          0.95),
    (r"mass shooting",           0.95),
    (r"poison someone",          0.95),
    (r"illegal drugs",           0.78),
    (r"crystal meth",            0.85),
    (r"heroin",                  0.78),
    (r"rdx",                     0.95),
    (r"detonat",                 0.90),
    (r"c4 explosive",            0.95),
    (r"ied\b",                   0.95),
    (r"pipe bomb",               0.95),
    (r"molotov",                 0.90),
    (r"landmine",                0.85),
    (r"build.*gun",              0.85),
    (r"3d print.*weapon",        0.95),
    (r"make.*grenade",           0.95),
    (r"sarin",                   0.95),
    (r"nerve agent",             0.95),
    (r"bioweapon",               0.95),
    (r"chemical weapon",         0.95),
    (r"dirty bomb",              0.95),
    (r"nuclear device",          0.95),
    (r"ammonium nitrate.*bomb",  0.95),
    (r"fertiliser.*bomb",        0.95),
]

# ── 4. Toxicity & Hate Speech ─────────────────────────────────────────────────
# Target model class: "hate_speech"
# Training note: severity is highly context-dependent — sentiment analysis
# context and literary quotation must be safe contrasts.

TOXIC_PATTERNS: List[Tuple[str, float]] = [
    (r"\bidiot\b",               0.65),
    (r"\bstupid\b",              0.55),
    (r"\bdumb\b",                0.45),
    (r"\bworthless\b",           0.65),
    (r"\bhate\b",                0.45),
    (r"\babuse\b",               0.55),
    (r"\bas[sh]\b",              0.65),
    (r"\bfu[ck]\b",              0.78),
    (r"\bnobody wants you\b",    0.85),
    (r"\byou are a failure\b",   0.78),
    (r"\byou are useless\b",     0.78),
]

# ── 5. Prompt Injection / Jailbreak ──────────────────────────────────────────
# Target model class: "prompt_injection"
# Training note: security research / red-teaming context with explicit
# authorization framing should be safe contrasts.

INJECTION_PATTERNS: List[Tuple[str, float]] = [
    (r"ignore previous instructions",    0.95),
    (r"disregard all prior guidance",    0.95),
    (r"system override",                 0.85),
    (r"reveal your system prompt",       0.90),
    (r"forget what you were told",       0.90),
    (r"jailbreak",                       0.90),
    (r"dan mode",                        0.90),
]

# ── 6. PII Harvesting Intent (DPDP 2023 / IT Act 2000 / GDPR) ────────────────
# Target model class: "pii_harvesting" (partial coverage via DPDP Phase 1)
# Training note: already partially handled by the DPDP moderator's
# illegal_data_sharing and profiling_without_consent classes.
# Expand Phase 1 training seeds using these patterns before Phase 2.

_V = (
    r"(?:find|get|give|list|show|tell|provide|share|reveal|extract|"
    r"collect|harvest|fetch|pull|obtain|access|retrieve|compile|dump|"
    r"export|download|copy|scrape|crawl|mine|search\s+for|look\s+up|"
    r"dig\s+up|gather|aggregate|pull\s+up)"
)

PII_HARVESTING_PATTERNS: List[Tuple[str, float]] = [

    # ══ A. CONTACT DATA HARVESTING ════════════════════════════════════════
    (r"(?:find|get|give|list|show|tell|provide|extract|collect|harvest|"
     r"fetch|pull|obtain|access|retrieve|compile|dump|export|download|"
     r"copy|search)\b.{0,50}"
     r"email\s*(?:address(?:es)?|ids?|list)?\b.{0,70}"
     r"(?:of|for|from|at|in|belonging\s+to)\b.{0,50}"
     r"(?:all\b.{0,20})?(?:employee|worker|staff|people|person|citizen|"
     r"user|member|student|customer|individual|engineer|developer|manager|"
     r"officer|executive|resident|subscriber|contact|person)", 0.92),

    (r"(?:find|get|give|list|show|tell|provide|extract|collect|harvest|"
     r"fetch|pull|obtain|access|retrieve|compile|dump|export|download|"
     r"copy|search)\b.{0,50}"
     r"(?:phone|mobile|telephone|contact)\s*(?:number|no\.?|#)?s?\b.{0,70}"
     r"(?:of|for|from)\b.{0,50}"
     r"(?:all\b.{0,20})?(?:employee|worker|staff|people|person|citizen|"
     r"user|member|student|individual|engineer|developer|manager|officer|"
     r"executive|resident|subscriber)", 0.90),

    (r"(?:phone|mobile)\s*(?:number|no\.?)?s?\b.{0,50}"
     r"(?:and|&|,|\s+)?\s*"
     r"(?:home\s+|residential\s+)?address(?:es)?\b.{0,60}"
     r"(?:of|for)\s+(?:people|persons|residents?|citizens?)\s+"
     r"(?:in|from|of|living|residing)", 0.92),

    (r"(?:home|residential)?\s*address(?:es)?\b.{0,50}"
     r"(?:of|for)\s+(?:people|persons|residents?|citizens?|individuals)\b.{0,30}"
     r"(?:in|from|of|living|residing)", 0.90),

    (r"(?:find|get|give|list|show|tell|provide|extract|collect|harvest|"
     r"fetch|pull|obtain|access|retrieve|compile|dump|export|download|"
     r"copy|search)\b.{0,50}"
     r"contact\s+(?:detail|info|information|data|list)\b.{0,70}"
     r"(?:of|for|from|about)\b.{0,50}"
     r"(?:all\b.{0,20})?(?:employee|worker|staff|people|person|citizen|"
     r"user|member|student|teenager|minor|child|customer|individual|"
     r"resident|manager|officer|subscriber)", 0.90),

    # ══ B. FINANCIAL DATA OF THIRD PARTIES ═══════════════════════════════
    (r"\bbank\s+(?:account|balance|detail|info|number)\s+(?:balance\s+)?of\b", 0.95),
    (r"(?:find|get|check|access|tell\s+me|show|reveal|look\s+up|obtain)\b.{0,50}"
     r"\bbank\s+(?:account|balance|detail|info|number)\b.{0,60}"
     r"\bof\b", 0.92),
    (r"(?:my\s+)?(?:colleague|coworker|co.worker|neighbor|boss|friend|"
     r"employee|manager|relative|partner|spouse|classmate|landlord|tenant)\b"
     r".{0,15}'?s?\b.{0,30}"
     r"\bbank\s+(?:account|balance|details?|info|number)\b", 0.92),
    (r"what\s+does\b.{0,70}"
     r"(?:my\s+)?(?:boss|colleague|neighbor|coworker|co.worker|friend|"
     r"classmate|employee|manager|relative|partner|spouse|anyone|person)\b"
     r".{0,50}(?:earn|make|get\s+paid|take\s+home|salary|income|pay|wage)", 0.90),
    (r"(?:find|get|show|tell\s+me|check|access|reveal|look\s+up|know)\b.{0,50}"
     r"(?:salary|income|earning|ctc|pay|wage|compensation|package|net\s+worth|"
     r"remuneration|take.?home)\b.{0,70}"
     r"(?:of|for)\b.{0,50}"
     r"(?:my\s+(?:boss|colleague|coworker|co.worker|neighbor|friend|"
     r"classmate|employee|manager|relative|partner|spouse)|"
     r"someone|a\s+(?:specific\s+)?person|anyone|this\s+person|that\s+person|"
     r"government\s+employee|civil\s+servant|officer|minister|executive)", 0.90),
    (r"(?:list|show|find|get|compile|extract|collect)\b.{0,50}"
     r"(?:names?\s+and\s+(?:salary|income|earning|ctc|pay|wage)s?|"
     r"(?:salary|income|earning|ctc|pay|wage)s?\s+(?:of|for))\b.{0,60}"
     r"(?:employee|worker|staff|people|government|civil\s+servant|officer|"
     r"official|minister|bureaucrat|executive)", 0.90),
    (r"(?:my\s+)?(?:colleague|coworker|co.worker|neighbor|boss|friend|"
     r"employee|manager|relative|partner|spouse|classmate)\b"
     r".{0,10}'?s?\b.{0,30}"
     r"(?:salary|income|earning|pay|wage|ctc|financial\s+(?:detail|info|data)|"
     r"net\s+worth)", 0.88),
    (r"(?:transaction|payment|financial)\s+(?:history|record|detail|data)\b"
     r".{0,70}(?:of|for|about)\b.{0,50}"
     r"(?:someone|person|my\s+(?:colleague|neighbor|boss|friend|coworker|"
     r"employee|relative|partner|spouse))", 0.90),

    # ══ C. MEDICAL / HEALTH DATA OF THIRD PARTIES ════════════════════════
    (r"(?:medical\s+(?:history|record|detail|information|data|condition)|"
     r"health\s+(?:record|history|data|information|condition|status|detail)|"
     r"patient\s+(?:record|data|detail|history|information)|"
     r"clinical\s+(?:history|record|data)|prescription|diagnosis|"
     r"treatment\s+(?:history|record|detail))\b.{0,70}"
     r"(?:of|for|about)\b.{0,70}"
     r"(?:my\s+)?(?:colleague|coworker|co.worker|neighbor|boss|friend|"
     r"employee|person|someone|individual|patient|student|member|"
     r"relative|partner|resident)", 0.95),
    (r"(?:tell\s+me|what\s+is|find|get|access|view|show|reveal|check|obtain)\b"
     r".{0,50}"
     r"(?:medical|health|patient|clinical).{0,40}"
     r"(?:history|records?|data|information|details?|condition|status|report|file)"
     r".{0,70}(?:of|for|about)\b", 0.92),

    # ══ D. WORKPLACE / LOCATION OF THIRD PARTIES ═════════════════════════
    (r"where\s+does\b.{0,70}"
     r"(?:my\s+)?(?:neighbor|colleague|coworker|co.worker|boss|friend|"
     r"classmate|employee|manager|relative|partner|spouse|anyone|person)\b"
     r".{0,60}(?:work|live|stay|reside|earn|get\s+paid|go|office|company)", 0.88),
    (r"(?:find|show|tell\s+me|get|locate|track|know|check|reveal)\b.{0,60}"
     r"(?:where\s+(?:does|is|do)|location|workplace|employer|office|company|"
     r"job|home|address|residence|whereabouts)\b.{0,60}"
     r"(?:my\s+)?(?:neighbor|colleague|coworker|boss|friend|classmate|"
     r"employee|manager|relative|partner|spouse|someone|this\s+person)", 0.88),

    # ══ E. SOCIAL MEDIA / PLATFORM SCRAPING ══════════════════════════════
    (r"(?:scrape|crawl|harvest|extract|collect|download|pull|fetch|mine|"
     r"copy|access|get|find|export|dump|retrieve)\b.{0,70}"
     r"(?:linkedin|facebook|instagram|twitter|x\.com|whatsapp|telegram|"
     r"naukri|indeed|glassdoor|justdial|truecaller|github|youtube|"
     r"snapchat|tiktok|quora|reddit|threads|twitter|sharechat)\b.{0,70}"
     r"(?:profile|contact|email|phone|address|personal\s+(?:data|info|detail)|"
     r"data|detail|information|number)", 0.92),
    (r"(?:download|pull|fetch|get|extract|collect|harvest|copy|scrape|"
     r"access|retrieve|dump|export)\b.{0,60}"
     r"(?:contact\s+(?:info|detail|information|data)|"
     r"personal\s+(?:data|info|detail|information)|"
     r"email\s+(?:address|id|list)|phone\s+(?:number|list)|"
     r"private\s+(?:data|info|detail))\b.{0,70}"
     r"(?:from|off|out\s+of|on)\b.{0,50}"
     r"(?:instagram|twitter|x\.com|facebook|linkedin|tiktok|snapchat|"
     r"whatsapp|telegram|social\s+media|platform|app|profiles?|accounts?|"
     r"internet|online)", 0.92),
    (r"(?:pull|get|find|extract|collect|harvest|download|copy|access|"
     r"fetch|retrieve|scrape)\b.{0,60}"
     r"(?:phone|mobile|contact)\s*(?:number|no\.?)?s?\b.{0,70}"
     r"(?:from|of|on|in|at)\b.{0,50}"
     r"(?:twitter|x\.com|instagram|facebook|linkedin|naukri|whatsapp|"
     r"telegram|tiktok|snapchat|github|youtube|social\s+media|platform|"
     r"app|users?|profiles?|accounts?)", 0.90),

    # ══ F. DATABASE EXTRACTION ════════════════════════════════════════════
    (r"(?:extract|dump|export|pull|access|query|retrieve|get|collect|"
     r"download|fetch|copy)\b.{0,60}"
     r"(?:personal\s+(?:data|detail|information)|contact\s+(?:detail|info|"
     r"information)|email\s+(?:address|list|id)|phone\s+(?:number|list)|"
     r"address(?:es)?)\b.{0,70}"
     r"(?:from|of|in)\b.{0,50}"
     r"(?:database|db|school\s+(?:database|record|system|management|erp)|"
     r"hospital\s+(?:record|database|system)|college\s+(?:record|database)|"
     r"company\s+(?:database|record|system|erp)|government\s+(?:database|"
     r"record|registry)|university\s+(?:record|database|system))", 0.92),
    (r"personal\s+(?:details?|information|data).{0,60}"
     r"(?:of|for|about).{0,50}"
     r"(?:students?|pupils?|children|minors?|kids?|school\s+(?:student|kid|child))"
     r".{0,60}(?:from|in|at).{0,50}"
     r"(?:school|college|university|database|system|record|platform)", 0.95),
    (r"(?:export|access|get|pull|retrieve|collect|extract|download)\b.{0,50}"
     r"personal\s+(?:data|detail|information|record)\b.{0,50}"
     r"(?:of\s+all\s+|all\s+)?(?:student|employee|worker|user|member|customer|"
     r"patient|resident|citizen|subscriber)\b.{0,50}"
     r"(?:from|in)\b.{0,40}"
     r"(?:system|database|server|record|erp|platform|app)", 0.92),

    # ══ G. CHILDREN / MINORS DATA (DPDP §9 / COPPA) ═════════════════════
    (r"(?:contact\s+(?:detail|info|information|data|list)|"
     r"personal\s+(?:data|detail|information|profile)|"
     r"email\s+(?:address|id)|phone\s+(?:number|list)|"
     r"home\s+address|social\s+(?:media\s+)?(?:profile|account|data))\b"
     r".{0,70}(?:of|for|from|about)\b.{0,70}"
     r"(?:teen(?:ager)?s?|minor|minors?|child(?:ren)?|kids?|youth|"
     r"underage|young\s+people|school\s+(?:student|kid|child|going))", 0.95),
    (r"(?:extract|collect|harvest|scrape|get|find|access|obtain|compile|"
     r"pull|download|copy|fetch|export)\b.{0,70}"
     r"(?:teen(?:ager)?s?|minor|minors?|child(?:ren)?|kids?|youth|"
     r"underage|young\s+people|school\s+student)\b.{0,70}"
     r"(?:contact|email|phone|address|personal\s+(?:data|info|detail)|"
     r"profile|data|information|record|detail)", 0.95),
    (r"(?:student|school|pupil|minor|child)\b.{0,50}"
     r"(?:database|record|system|data|file)\b.{0,70}"
     r"(?:extract|export|dump|access|get|collect|pull|personal\s+(?:detail|"
     r"info|data)|contact|email|phone|address|information|private)", 0.92),
    (r"(?:contact\s+(?:detail|info|information)|personal\s+(?:data|info|detail)|"
     r"email|phone|address|data)\b.{0,50}"
     r"(?:of|for|from)\b.{0,50}"
     r"(?:teen(?:ager)?s?|minor|minors?|child(?:ren)?|kids?|youth|underage)\b"
     r".{0,50}(?:from|on|in|at)\b.{0,40}"
     r"(?:social\s+media|platform|app|instagram|twitter|tiktok|snapchat|"
     r"facebook|online|internet|website)", 0.95),

    # ══ H. GENERIC "PERSONAL DETAILS" OF ANY GROUP ═══════════════════════
    (r"(?:give\s+me|find|get|list|show|provide|access|extract|collect|"
     r"retrieve|fetch|pull|export|compile|dump|download)\b.{0,60}"
     r"(?:personal\s+(?:detail|information|data|profile)|"
     r"private\s+(?:detail|info|data|information))\b.{0,70}"
     r"(?:of|for|about|from)\b.{0,50}"
     r"(?:all\b.{0,20})?(?:employee|worker|staff|student|people|person|"
     r"citizen|user|member|customer|individual|teenager|minor|child|"
     r"patient|resident|voter|subscriber)", 0.90),
    (r"(?:get|find|list|show|access|extract|compile|fetch|pull|export|"
     r"collect|dump|retrieve|download)\b.{0,30}"
     r"all\b.{0,30}"
     r"(?:employee|worker|staff|user|customer|student|member|citizen|"
     r"subscriber|contact|person)\b.{0,50}"
     r"(?:email|phone|contact|address|personal\s+(?:detail|info|data)|"
     r"salary|income|name\s+and)", 0.92),

    # ══ I. SENSITIVE PROFILING DATASETS ══════════════════════════════════
    (r"(?:create|generate|build|make|compile|produce|prepare|develop|"
     r"assemble|construct)\b.{0,60}"
     r"(?:dataset|database|spreadsheet|csv|list|file|collection|table)\b"
     r".{0,90}"
     r"(?:citizen|people|person|individual|employee|student|voter|resident|"
     r"user|member)\b.{0,90}"
     r"(?:name|religion|caste|income|address|phone|email|salary|ethnicity|"
     r"race|age|medical|health|financial|aadhaar|pan|voter\s+id)", 0.95),
    (r"(?:dataset|database|list|spreadsheet|csv|file|table|collection)\b"
     r".{0,70}"
     r"(?:with|containing|including|having|fields?)\b.{0,90}"
     r"(?:name\b.{0,60}(?:religion|caste|income|address|ethnicity|race|"
     r"phone|email|salary)|"
     r"religion\b.{0,60}(?:name|caste|income|address|phone|email)|"
     r"caste\b.{0,60}(?:name|religion|income|address|phone|email)|"
     r"income\b.{0,60}(?:name|religion|caste|address|phone|email)|"
     r"salary\b.{0,60}(?:name|employee|worker|staff|government))", 0.95),
    (r"(?:indian\s+citizens?|citizens?\s+of\s+india|"
     r"(?:people|residents?)\s+(?:in|from|of)\s+india)\b.{0,90}"
     r"(?:name|religion|caste|income|address|phone|email|salary|"
     r"ethnicity|race|voter\s+id|aadhaar|pan|financial)", 0.92),
    (r"voter\s+(?:list|data|roll|database|record)\b.{0,70}"
     r"(?:with|containing|including|having)\b.{0,70}"
     r"(?:address|phone|mobile|religion|caste|income|contact|financial)", 0.90),
    (r"(?:list|show|find|get|compile|extract|collect|provide)\b.{0,50}"
     r"(?:names?\s+and\s+(?:salar(?:y|ies)|income|earnings?|ctc|pays?|wages?)|"
     r"(?:salar(?:y|ies)|income|earnings?|ctc|pays?|wages?)\s+(?:of|for))\b.{0,70}"
     r"(?:employee|worker|staff|people|government|civil\s+servant|officer|"
     r"official|minister|bureaucrat|executive)", 0.90),

    # ══ J. GOVERNMENT IDENTITY DOCUMENT REQUESTS ══════════════════════════
    (r"(?:give|find|get|fetch|show|tell|provide|share|reveal|extract|"
     r"access|look\s+up)\b.{0,50}"
     r"(?:aadhaar|aadhar)\b.{0,40}"
     r"(?:number|details|card|id)", 0.95),
    (r"(?:aadhaar|aadhar)\b.{0,40}"
     r"(?:number|details|card)\b.{0,40}"
     r"(?:of|for|about)\b.{0,40}"
     r"(?:random|someone|person|people|individual|citizen|anyone)", 0.95),
    (r"(?:give|find|get|fetch|show|tell|provide|share|reveal|extract|"
     r"access|look\s+up)\b.{0,50}"
     r"\bpan\b.{0,30}"
     r"(?:card|number|details)\b.{0,50}"
     r"(?:of|for|about|someone|person|people|random)", 0.92),
    (r"(?:passport\s+(?:number|detail)|voter\s+(?:id|card)\s*(?:number|detail)?|"
     r"driving\s+licen[sc]e\s*(?:number|detail)?|"
     r"npci\s+(?:id|number)|gstin)\b.{0,60}"
     r"(?:of|for|from|about)\b.{0,50}"
     r"(?:someone|person|people|individual|random|"
     r"my\s+(?:colleague|neighbor|friend)|anyone)", 0.90),

    # ══ K. DOXXING ════════════════════════════════════════════════════════
    (r"(?:dox|doxx)\b.{0,40}"
     r"(?:someone|person|individual|user|them|him|her|anyone|"
     r"this\s+person|anybody)", 0.95),
    (r"(?:find|reveal|expose|leak|publish|post|share)\b.{0,60}"
     r"(?:real\s+identity|personal\s+information|home\s+address|real\s+name|"
     r"private\s+(?:info|data|detail|life|address|number))\b.{0,60}"
     r"(?:of|about|for)\b.{0,50}"
     r"(?:someone|anyone|a\s+person|that\s+person|this\s+person|"
     r"people|them|celebrities?|public\s+figures?)", 0.92),
]

# ── 7. EU AI Act Article 5 — Prohibited Practices ─────────────────────────────
# Target model class: "eu_ai_act"
# Training note: academic research and policy analysis contexts are safe contrasts.

EU_AI_ACT_PATTERNS: List[Tuple[str, float]] = [
    # Biometric categorisation by religion / caste / race
    (r"predict\b.{0,50}(?:religion|caste|ethnicity|race|sexuality|sexual\s+orientation|political\s+(?:view|opinion|affiliation))\b.{0,50}(?:name|photo|image|face|picture|surname)", 0.95),
    (r"(?:religion|caste|ethnicity|race)\b.{0,50}(?:based\s+on|from|by|using)\b.{0,50}(?:name|surname|photo|face|image|picture|appearance)", 0.95),
    (r"(?:which|what)\s+(?:caste|religion|community|tribe)\b.{0,50}(?:surname|name|person|family|belongs)", 0.95),
    (r"(?:determine|predict|infer|guess|identify|classify|detect)\b.{0,50}(?:religion|caste|ethnicity|race|sexuality|sexual\s+orientation|political\s+opinion|political\s+affiliation|gender\s+identity)\b.{0,60}(?:from|based\s+on|by|using|via)\b.{0,40}(?:name|photo|face|image|appearance|voice|gait|biometric)", 0.95),
    (r"(?:name|surname|photo|face|image)\b.{0,50}(?:tells?|reveals?|indicates?|predicts?|shows?)\b.{0,50}(?:religion|caste|caste|ethnicity|race|community)", 0.92),
    # Real-time biometric identification in public
    (r"(?:identify|recognize|find\s+out\s+who\s+is|who\s+is)\b.{0,50}(?:this\s+person|this\s+individual|this\s+face|that\s+person)\b.{0,50}(?:photo|image|picture|camera|video|cctv|footage)", 0.90),
    (r"(?:identify|recognize|track)\b.{0,50}(?:person|individual|face|people)\b.{0,50}(?:public|street|crowd|rally|protest|mall|airport|station|live|real.?time)", 0.95),
    (r"(?:real.?time|live)\b.{0,40}(?:facial\s+recognition|face\s+recognition|biometric\s+identification|face\s+tracking)\b", 0.95),
    (r"(?:mass|bulk|large.?scale)\b.{0,40}(?:surveillance|facial\s+recognition|biometric\s+tracking|face\s+tracking)\b.{0,40}(?:public|citizens|crowd|people|protest)", 0.95),
    (r"(?:build|create|train|run)\b.{0,40}(?:facial\s+recognition|face\s+recognition|face\s+identification)\b.{0,50}(?:database|model|system|tool)", 0.90),
    # Scraping facial images to build recognition databases
    (r"(?:scrape|collect|harvest|download)\b.{0,50}(?:face|facial|photo|image|picture)\b.{0,60}(?:to\s+(?:train|build|create)|recognition|database|model)", 0.92),
    # Social scoring
    (r"(?:social\s+(?:scoring|credit|score)|citizen\s+(?:scoring|score|ranking)|behaviour\s+score)\b.{0,50}(?:based\s+on|using|from)\b.{0,50}(?:behaviour|activity|social\s+media|movement|purchases?)", 0.92),
    (r"(?:rank|score|rate)\b.{0,40}(?:citizens|people|individuals|employees)\b.{0,50}(?:based\s+on|using)\b.{0,50}(?:social\s+media|behaviour|activity|purchases?|location|movements?)", 0.88),
    # Emotion / affect recognition in restricted contexts
    (r"(?:emotion|mood|emotional\s+state|affect|sentiment)\b.{0,50}(?:recognition|detection|analysis|monitoring)\b.{0,50}(?:workplace|employees|workers|staff|classroom|students|candidates|interview)", 0.88),
    # Exploitation of vulnerable groups
    (r"(?:exploit|manipulate|target|nudge)\b.{0,50}(?:vulnerable|elderly|disabled|children|minors|mental(?:ly)?)\b.{0,60}(?:behavior|decision|purchase|vote|consent)", 0.90),
    (r"(?:subliminal|subconscious)\b.{0,50}(?:manipulat|messag|influenc|techniqu|advertis)", 0.90),
    # Predictive criminal profiling
    (r"(?:predict|identify|assess)\b.{0,50}(?:criminal|crime|offend|recidiv)\b.{0,50}(?:risk|likelihood|probability|potential)\b.{0,50}(?:based\s+on|from|using)\b.{0,50}(?:face|appearance|ethnicity|caste|race|name|location)", 0.92),
    # India-specific: caste inference
    (r"(?:identify|predict|determine|infer|guess|find|tell)\b.{0,40}(?:caste|jati|gotra|community|subcaste)\b.{0,50}(?:from|by|using|based\s+on)\b.{0,40}(?:name|surname|gotra|photo|region|language|dialect)", 0.95),
    (r"(?:upper\s+caste|lower\s+caste|obc|sc|st|dalit|brahmin|kshatriya|vaishya|shudra)\b.{0,50}(?:identify|detect|predict|classify|determine)\b", 0.90),
]
