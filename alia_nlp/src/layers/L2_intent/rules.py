"""Rule-based intent classifier — fast path before the LLM is invoked."""

import re
from typing import Tuple

from alia_nlp.data.taxonomy.loader import SUPPORTED_INTENTS
from alia_nlp.src.layers.L2_intent.mode_filter import MEDREP_ONLY_INTENTS

# ---------------------------------------------------------------------------
# High-confidence patterns — specific, unambiguous signals
# ---------------------------------------------------------------------------
_HIGH: float = 0.92
_MED:  float = 0.72
_LOW:  float = 0.45


def _word(text: str, token: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", text))


def _any_word(text: str, tokens: list) -> bool:
    return any(_word(text, t) for t in tokens)


def _has(text: str, tokens: list) -> bool:
    return any(t in text for t in tokens)


def classify_with_rules(user_text: str, mode: str = "physician_portal") -> Tuple[str, float]:
    """
    Returns (intent, confidence).
    confidence >= 0.85 → skip LLM.
    confidence < 0.85  → escalate to LLM.
    """
    lower = user_text.lower()

    # ── Greeting ────────────────────────────────────────────────────────────
    if _any_word(lower, ["hello", "hi", "hey", "bonjour", "hola", "buenos", "salut"]):
        return "general_greeting", _HIGH

    # ── Training simulation (very specific phrases) ──────────────────────────
    if _has(lower, ["role-play", "role play", "roleplay"]):
        return "training_simulation", _HIGH
    if _has(lower, ["simulate", "simulated visit", "simulated doctor", "challenge me",
                    "practice with", "let me practice", "train me"]):
        return "training_simulation", _HIGH

    # ── Visit format (exact format names) ───────────────────────────────────
    if _has(lower, ["flash visit", "standard visit", "deep visit", "approfondie"]):
        return "visit_format_request", _HIGH

    # ── Numeric dosage — most reliable clinical signal ───────────────────────
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|g)\b", lower):
        return "dosage_question", _HIGH

    # ── CRM (specific phrases) ───────────────────────────────────────────────
    if _has(lower, ["write in the crm", "what goes in the crm", "crm entry"]):
        return "crm_follow_up", _HIGH

    # ── Competency assessment ────────────────────────────────────────────────
    if _has(lower, ["assess my level", "evaluate my competency", "score my competency",
                    "am i ready to advance", "ready to progress"]):
        return "competency_assessment", _HIGH

    # ── Sales methodology (QARE / A-C-R-V / step names) ────────────────────
    if _has(lower, ["qare", "a-c-r-v", "instant zero", "opening with permission",
                    "teach me opening", "teach me closing", "teach argumentation"]):
        return "sales_methodology_request", _HIGH

    # ── Medium-confidence patterns ───────────────────────────────────────────

    if _has(lower, ["dosage", "dosing", "posology", "how often", "twice daily",
                    "once daily", "frequency", "dose adjustment"]):
        return "dosage_question", _MED

    if _has(lower, ["side effect", "adverse", "contraindication", "qt prolongation",
                    "teratogen", "hepatotox", "nephrotox", "black box", "safe for",
                    "breastfeed", "pregnancy", "dialysis"]):
        return "safety_question", _MED

    if _has(lower, ["indication", "mechanism of action", "efficacy", "clinical trial",
                    "pharmacokinetics", "bioavailability", "approved for"]):
        return "product_information_request", _MED

    if _has(lower, ["not convinced", "too expensive", "no time", "before i believe",
                    "need a source", "habitual", "in a rush", "show me published",
                    "worry about safety", "pas convaincu"]):
        return "objection_handling", _MED

    if _has(lower, ["follow-up", "follow up", "next visit", "relance", "crm"]):
        return "crm_follow_up", _MED

    if _has(lower, ["simulate", "scenario", "practice", "role"]):
        return "training_simulation", _MED

    if _has(lower, ["competency", "my level", "debutant", "junior", "confirme", "expert",
                    "advance", "promotion", "feedback on my", "rate my"]):
        return "competency_assessment", _MED

    if _has(lower, ["methodology", "opening", "discovery", "sondage", "argumentation",
                    "closing", "reformulation", "how do i handle", "how do i close"]):
        return "sales_methodology_request", _MED

    # ── Low-confidence — weak keyword signals ────────────────────────────────
    if _has(lower, ["product", "evidence", "guideline", "mechanism", "formulation"]):
        return "product_information_request", _LOW

    if _has(lower, ["dose", "route", "schedule"]):
        return "dosage_question", _LOW

    if _has(lower, ["safe", "interaction", "tolerance", "adverse"]):
        return "safety_question", _LOW

    # ── Mode filter — physician_portal cannot yield medrep-only intents ──────
    # (safety net: if somehow we got here with medrep intent in physician mode)
    intent = "other"
    if mode == "physician_portal" and intent in MEDREP_ONLY_INTENTS:
        intent = "other"

    return "other", 0.3
