"""Mode-differential intent filter.

Intents in MEDREP_ONLY_INTENTS cannot appear in physician_portal mode —
a physician does not do sales role-plays or CRM planning.
"""

from typing import FrozenSet

MEDREP_ONLY_INTENTS: FrozenSet[str] = frozenset({
    "training_simulation",
    "visit_format_request",
    "sales_methodology_request",
    "crm_follow_up",
    "competency_assessment",
})

_CLINICAL_FALLBACK_ORDER = [
    "safety_question",
    "dosage_question",
    "product_information_request",
]


def apply_mode_filter(
    intent: str,
    mode: str,
    is_safety: bool = False,
    is_dosage: bool = False,
    is_product: bool = False,
) -> str:
    if mode != "physician_portal" or intent not in MEDREP_ONLY_INTENTS:
        return intent

    # Re-classify to best clinical match
    if is_safety:
        return "safety_question"
    if is_dosage:
        return "dosage_question"
    if is_product:
        return "product_information_request"
    return "other"
