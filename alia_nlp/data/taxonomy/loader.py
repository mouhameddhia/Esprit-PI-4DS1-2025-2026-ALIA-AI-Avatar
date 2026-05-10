"""Centralized taxonomy loader — single import point for all pipeline constants."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).resolve().parent / "nlp_taxonomy.json"

_FALLBACK: Dict[str, Any] = {
    "intents": [
        "product_information_request", "dosage_question", "safety_question",
        "objection_handling", "training_simulation", "crm_follow_up",
        "competency_assessment", "visit_format_request",
        "sales_methodology_request", "general_greeting", "other",
    ],
    "visit_phases": [],
    "visit_formats": [],
    "objection_types": [],
    "auxiliary_tags": [],
    "safety_flags": ["patient_specific_advice_request", "diagnosis_request"],
    "entity_types": [],
}


def load_taxonomy() -> Dict[str, Any]:
    try:
        with _TAXONOMY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load taxonomy from %s: %s", _TAXONOMY_PATH, exc)
        return dict(_FALLBACK)


TAXONOMY: Dict[str, Any] = load_taxonomy()

SUPPORTED_INTENTS: FrozenSet[str] = frozenset(
    i for i in (TAXONOMY.get("intents") or []) if isinstance(i, str)
)
SUPPORTED_SAFETY_FLAGS: FrozenSet[str] = frozenset(
    f for f in (TAXONOMY.get("safety_flags") or []) if isinstance(f, str)
)
ENTITY_TYPES: List[str] = [
    t for t in (TAXONOMY.get("entity_types") or []) if isinstance(t, str)
]
SUPPORTED_SECONDARY_TAGS: FrozenSet[str] = frozenset({
    *(t for t in (TAXONOMY.get("visit_phases") or []) if isinstance(t, str)),
    *(t for t in (TAXONOMY.get("visit_formats") or []) if isinstance(t, str)),
    *(t for t in (TAXONOMY.get("objection_types") or []) if isinstance(t, str)),
    *(t for t in (TAXONOMY.get("auxiliary_tags") or []) if isinstance(t, str)),
})
