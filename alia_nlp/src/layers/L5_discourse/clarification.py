"""Clarification trigger — determines if the query needs disambiguation."""

import re
from typing import Dict, List

_DOMAIN_WORDS = frozenset([
    "product", "dose", "dosage", "indication", "safety", "side", "effect",
    "interaction", "contraindication", "patient", "simulation", "objection",
    "visit", "opening", "closing", "flash", "methodology", "crm", "follow",
    "competency", "training", "efficacy", "evidence", "guideline", "mechanism",
    "clinical", "trial", "adverse", "mg", "tablet", "capsule", "molecule",
    "posology", "tolerance", "teratogen", "hepatotox", "nephrotox", "qt",
    "dialysis", "geriatric", "pediatric", "breastfeed", "pregnancy",
])

_AMBIGUITY_PHRASES = [
    "not sure", "i don't know", "any idea", "what do you think",
    "where do i start", "can you help", "help me",
]


def should_clarify(
    user_text: str,
    intent: str,
    confidence: float,
    entity_map: Dict[str, List[str]],
    mode: str = "physician_portal",
    llm_flag: bool | None = None,
) -> bool:
    # Respect the LLM's explicit clarification signal when present
    if isinstance(llm_flag, bool):
        return llm_flag

    tokens = re.findall(r"\b\w+\b", user_text.lower())

    # Physicians ask concise clinical questions — lower token threshold
    min_tokens = 2 if mode == "physician_portal" else 3
    if len(tokens) < min_tokens and not any(t in _DOMAIN_WORDS for t in tokens):
        return True

    has_entity = any(v for v in entity_map.values() if v)
    if intent == "other" and not has_entity:
        return True

    # Confidence threshold varies by mode
    threshold = 0.30 if mode == "physician_portal" else 0.45
    if confidence < threshold:
        return True

    lower = user_text.lower()
    if any(m in lower for m in _AMBIGUITY_PHRASES) and intent in {"other", "general_greeting"}:
        return True

    return False
