"""Intent classification cascade: rules (fast) → LLM (accurate).

If the rule-based classifier is confident (≥ 0.85) the LLM is skipped,
saving latency and cost for the ~60 % of queries with unambiguous intent.
"""

import logging
from typing import Any, Dict, Tuple

from alia_nlp.data.taxonomy.loader import SUPPORTED_INTENTS
from alia_nlp.src.layers.L2_intent.rules import classify_with_rules
from alia_nlp.src.layers.L2_intent.mode_filter import apply_mode_filter, MEDREP_ONLY_INTENTS

logger = logging.getLogger(__name__)

_RULES_CONFIDENCE_THRESHOLD = 0.85

INTENT_ALIASES = {
    "training_objection": "objection_handling",
    "follow_up": "crm_follow_up",
}


def _normalize(intent: Any) -> str:
    if not isinstance(intent, str):
        return "other"
    resolved = INTENT_ALIASES.get(intent, intent)
    return resolved if resolved in SUPPORTED_INTENTS else "other"


def classify(
    user_text: str,
    mode: str,
    parsed_llm: Dict[str, Any],
) -> Tuple[str, float, str]:
    """
    Returns (intent, confidence, source).
    source is "rules" | "llm".

    parsed_llm is the already-parsed JSON from llm.call_structured().
    The caller decides whether to call the LLM based on rules confidence.
    """
    rule_intent, rule_confidence = classify_with_rules(user_text, mode)

    if rule_confidence >= _RULES_CONFIDENCE_THRESHOLD:
        final = apply_mode_filter(rule_intent, mode)
        return final, rule_confidence, "rules"

    # LLM path — use parsed response from caller
    if parsed_llm:
        llm_intent = _normalize(parsed_llm.get("intent", "other"))
        llm_confidence = float(parsed_llm.get("confidence", 0.5))
        llm_confidence = max(0.0, min(1.0, llm_confidence))
        final = apply_mode_filter(llm_intent, mode)
        return final, llm_confidence, "llm"

    # LLM unavailable — fall through to rules
    final = apply_mode_filter(rule_intent, mode)
    return final, rule_confidence, "rules"


def needs_llm(user_text: str, mode: str) -> bool:
    """Pre-check: should we even bother calling the LLM?"""
    _, confidence = classify_with_rules(user_text, mode)
    return confidence < _RULES_CONFIDENCE_THRESHOLD
