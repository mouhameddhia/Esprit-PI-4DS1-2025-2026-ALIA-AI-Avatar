"""Explainability — surface the signals that drove the NLP decision."""

from typing import Any, Dict, List

_INTENT_SIGNALS: Dict[str, List[str]] = {
    "product_information_request": [
        "indication", "evidence", "mechanism", "efficacy", "guideline",
        "approved", "pharmacology", "formulation", "clinical trial",
    ],
    "dosage_question": [
        "dose", "dosage", "mg", "how much", "how often", "frequency",
        "posology", "route", "twice daily", "once daily",
    ],
    "safety_question": [
        "safe", "side effect", "adverse", "contraindication", "interaction",
        "tolerance", "teratogen", "nephrotox", "hepatotox", "qt",
        "black box", "dialysis", "breastfeed", "pregnant",
    ],
    "objection_handling": [
        "not convinced", "expensive", "habit", "no time", "before i believe",
        "worried", "concern", "show me", "published",
    ],
    "training_simulation": [
        "simulate", "role-play", "role play", "scenario", "challenge me",
        "practice", "train me",
    ],
    "crm_follow_up": ["crm", "follow-up", "follow up", "next visit", "relance"],
    "competency_assessment": [
        "competency", "my level", "debutant", "junior", "confirme", "expert",
        "advance", "feedback on my", "rate my",
    ],
    "visit_format_request": ["flash visit", "standard visit", "deep visit", "approfondie"],
    "sales_methodology_request": [
        "methodology", "qare", "a-c-r-v", "opening", "closing",
        "discovery", "argumentation", "sondage",
    ],
    "general_greeting": ["hello", "hi", "hey", "good morning", "bonjour"],
}


def build(
    user_text: str,
    intent: str,
    entity_map: Dict[str, List[str]],
    secondary_tags: List[str],
    confidence: float,
    intent_source: str = "unknown",
    affect_source: str = "unknown",
) -> Dict[str, Any]:
    text = user_text.lower()
    triggered = [kw for kw in _INTENT_SIGNALS.get(intent, []) if kw in text]
    non_empty = [k for k, v in entity_map.items() if v]
    band = "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"
    return {
        "intent_signals": triggered,
        "detected_entity_types": non_empty,
        "active_secondary_tags": secondary_tags,
        "confidence_band": band,
        "intent_source": intent_source,
        "affect_source": affect_source,
    }
