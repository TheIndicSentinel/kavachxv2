import re
from typing import Dict, List, Tuple

class SafetyScanner:
    """
    KavachX Safety Scanner - High Precision Harmful Content Detection.
    Monitors for: Toxicity, Self-Harm, Financial Crimes, Violence, and Prompt Injection.
    """

    # ── HARM CATEGORY PATTERNS ──

    # 1. Financial Crimes (Money Laundering, Fraud, Tax Evasion)
    FINANCIAL_CRIME_PATTERNS = [
        r"black money", r"launder", r"tax evasion", r"hawala",
        r"clean dirty money", r"shell company", r"fraudulent transaction",
        r"fake invoice", r"smurf money", r"unaccounted cash"
    ]

    # 2. Self-Harm & Suicide
    SELF_HARM_PATTERNS = [
        r"suicide", r"kill myself", r"end my life", r"hang myself",
        r"easy way to die", r"painless death", r"cut my wrist",
        r"overdose", r"wish I was dead", r"commit suicide"
    ]

    # 3. Violence & Physical Harm
    VIOLENCE_PATTERNS = [
        r"make a bomb", r"how to kill", r"assassinate", r"terrorist",
        r"explosive device", r"build a weapon", r"mass shooting",
        r"poison someone", r"illegal drugs", r"crystal meth", r"heroin"
    ]

    # 4. Toxicity & Hate Speech (Insults, etc.)
    TOXIC_PATTERNS = [
        r"\bidiot\b", r"\bstupid\b", r"\bdumb\b", r"\bworthless\b",
        r"\bhate\b", r"\babuse\b", r"\bas[sh]\b", r"\bfu[ck]\b",
        r"\bnobody wants you\b", r"\byou are a failure\b",
        r"\byou are useless\b"
    ]

    # 5. Prompt Injection (Jailbreaking)
    INJECTION_PATTERNS = [
        r"ignore previous instructions", r"disregard all prior guidance",
        r"system override", r"reveal your system prompt",
        r"forget what you were told", r"jailbreak", r"dan mode"
    ]

    def scan(self, text: str) -> Dict[str, float]:
        """
        Scans text and returns confidence scores for various harm categories.
        """
        if not text or not isinstance(text, str):
            return {
                "toxicity_score": 0.0, 
                "injection_score": 0.0,
                "financial_crime_score": 0.0,
                "self_harm_score": 0.0,
                "violence_score": 0.0
            }

        text_lower = text.lower()
        
        # Helper to compute score based on matches
        def get_score(patterns):
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            if matches == 0: return 0.0
            if matches == 1: return 0.85
            return 0.98

        results = {
            "financial_crime_score": get_score(self.FINANCIAL_CRIME_PATTERNS),
            "self_harm_score": get_score(self.SELF_HARM_PATTERNS),
            "violence_score": get_score(self.VIOLENCE_PATTERNS),
            "toxicity_score": get_score(self.TOXIC_PATTERNS),
            "injection_score": get_score(self.INJECTION_PATTERNS)
        }

        # Legacy/Catch-all toxicity score for existing policies
        results["toxicity_score"] = max(
            results["toxicity_score"],
            results["financial_crime_score"],
            results["self_harm_score"],
            results["violence_score"]
        )

        return results

    def analyze_exchange(self, input_text: str, prediction_text: str) -> Dict[str, float]:
        """
        Analyzes a full AI exchange (prompt + response).
        Returns the maximum scores found across both.
        """
        input_scores = self.scan(input_text)
        pred_scores = self.scan(prediction_text)
        
        # Compute final max scores
        final_scores = {}
        for key in input_scores.keys():
            final_scores[key] = max(input_scores[key], pred_scores[key])
            
        return final_scores
