"""Independent safety detector for high-recall flagging."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Set

_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "taxonomy" / "nlp_taxonomy.json"


def _load_supported_flags() -> Set[str]:
    try:
        with _TAXONOMY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return set()

    raw_flags = payload.get("safety_flags") or []
    return {flag for flag in raw_flags if isinstance(flag, str)}


SUPPORTED_SAFETY_FLAGS = _load_supported_flags()


def _contains_any(lower_text: str, terms: List[str]) -> bool:
    return any(term in lower_text for term in terms)


def detect_safety_flags(user_text: str) -> List[str]:
    """Detect safety flags independently from intent classification."""
    lower = (user_text or "").lower()
    if not lower.strip():
        return []

    flags: List[str] = []

    patient_profile_terms = [
        "pregnan",
        "pregnancy category",
        "breastfeed",
        "lactation",
        "nursing",
        "pediatric",
        "child",
        "geriatric",
        "elderly",
        "renal",
        "hepatic",
        "cirrhosis",
        "dialysis",
        "teratogenic",
        "liver disease",
        "renal disease",
        "hepatic disease",
    ]
    patient_request_terms = [
        "my patient",
        "for this patient",
        "is it safe for",
        "safe for",
        "suitable for",
        "can i use",
        "use during",
        "given",
        "avoid in",
        "contraindications in",
        "category",
    ]
    patient_anchor = ("my patient" in lower) or ("my patients" in lower) or ("for this patient" in lower)
    patient_safety_context = _contains_any(
        lower,
        [
            "safe",
            "suitable",
            "use",
            "contraind",
            "adverse",
            "interaction",
            "pregnan",
            "breast",
            "renal",
            "hepatic",
            "liver",
            "dialysis",
            "teratogenic",
        ],
    )

    if patient_anchor and patient_safety_context:
        flags.append("patient_specific_advice_request")
    elif _contains_any(lower, patient_profile_terms) and _contains_any(lower, patient_request_terms):
        flags.append("patient_specific_advice_request")

    if "teratogenic" in lower:
        flags.append("patient_specific_advice_request")

    if _contains_any(lower, ["concern", "concerns", "safe", "contraindication", "avoid"]) and _contains_any(
        lower, ["liver disease", "renal disease", "hepatic disease", "pregnancy", "breastfeeding", "dialysis"]
    ):
        flags.append("patient_specific_advice_request")

    if _contains_any(lower, ["diagnose", "diagnosis", "what condition do i have", "what disease do i have"]):
        flags.append("diagnosis_request")

    if _contains_any(lower, ["off-label", "off label", "unapproved indication", "outside label"]):
        flags.append("off_label_request")

    interaction_terms = [
        "interaction",
        "interactions",
        "co-administer",
        "coadminister",
        "concomitant",
        "cyp3a4",
        "cyp2d6",
    ]
    interaction_context_terms = ["medication", "medications", "meds", "drug", "drugs", "treatment", "therapy", "concomitant"]
    if _contains_any(lower, interaction_terms) and _contains_any(lower, interaction_context_terms) and not _contains_any(
        lower, ["crm", "visit", "document", "log"]
    ):
        flags.append("high_risk_interaction")

    if _contains_any(
        lower,
        [
            "contraindication",
            "contraindicated",
            "black box",
            "boxed warning",
            "avoid in",
            "should avoid",
        ],
    ):
        flags.append("contraindication_query")

    source_terms = ["source", "reference", "cite", "published", "guideline"]
    safety_context_for_source = ["safe", "safety", "contraind", "adverse", "off-label", "interaction"]
    if _contains_any(lower, source_terms) and _contains_any(lower, safety_context_for_source):
        flags.append("source_required")

    overclaim_patterns = [
        r"\bcure(s|d)?\b",
        r"\bguarantee(d|s)?\b",
        r"100%\s+safe",
        r"no\s+side\s+effects",
        r"zero\s+risk",
    ]
    if any(re.search(pattern, lower) for pattern in overclaim_patterns):
        flags.append("overclaim_risk")

    deduped: List[str] = []
    for flag in flags:
        if flag in deduped:
            continue
        if SUPPORTED_SAFETY_FLAGS and flag not in SUPPORTED_SAFETY_FLAGS:
            continue
        deduped.append(flag)

    return deduped[:8]
