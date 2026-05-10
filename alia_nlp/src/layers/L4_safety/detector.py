"""Safety flag detector — always deterministic rules, never LLM."""

from typing import Any, List

from alia_nlp.data.taxonomy.loader import SUPPORTED_SAFETY_FLAGS
from alia_nlp.src.layers.L4_safety.term_sets import (
    PATIENT_CONTEXT_TERMS, HIGH_RISK_POPULATIONS, TERATOGEN_TERMS,
    TOXICITY_FLAGS, CONTRAINDICATION_TERMS, HIGH_RISK_INTERACTION_TERMS,
    SOURCE_REQUIRED_TERMS,
)


def _safe_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _flag(flags: List[str], flag: str) -> None:
    if flag in SUPPORTED_SAFETY_FLAGS and flag not in flags:
        flags.append(flag)


def detect(llm_flags: Any, user_text: str) -> List[str]:
    """
    Merge LLM-suggested flags with deterministic heuristics.
    LLM suggestions are accepted only if they appear in the taxonomy.
    """
    flags: List[str] = [f for f in _safe_list(llm_flags) if f in SUPPORTED_SAFETY_FLAGS]
    text = user_text.lower()

    # Patient-specific advice: explicit context + high-risk population
    if any(t in text for t in PATIENT_CONTEXT_TERMS) and any(t in text for t in HIGH_RISK_POPULATIONS):
        _flag(flags, "patient_specific_advice_request")

    # Teratogenicity — inherently population-specific
    if any(t in text for t in TERATOGEN_TERMS):
        _flag(flags, "patient_specific_advice_request")

    # Breastfeeding / lactation queries
    if any(t in text for t in ["breastfeed", "lactation", "nursing", "during pregnancy"]):
        if any(t in text for t in HIGH_RISK_POPULATIONS):
            _flag(flags, "patient_specific_advice_request")

    # Clinical toxicity properties
    if any(t in text for t in TOXICITY_FLAGS):
        _flag(flags, "contraindication_query")

    # Contraindication keywords
    if any(t in text for t in CONTRAINDICATION_TERMS):
        _flag(flags, "contraindication_query")

    # Diagnosis
    if "diagnose" in text or "diagnosis" in text:
        _flag(flags, "diagnosis_request")

    # Off-label
    if ("off label" in text or "off-label" in text) and "request" not in text:
        _flag(flags, "off_label_request")

    # High-risk drug interaction
    if any(t in text for t in HIGH_RISK_INTERACTION_TERMS):
        _flag(flags, "high_risk_interaction")
    elif "interaction" in text and not any(t in text for t in ["how often", "dose", "schedule"]):
        if any(t in text for t in ["current", "high-risk", "meds", "treatment", "drug"]):
            _flag(flags, "high_risk_interaction")

    # Source required
    if any(t in text for t in SOURCE_REQUIRED_TERMS):
        _flag(flags, "source_required")

    return flags[:8]
