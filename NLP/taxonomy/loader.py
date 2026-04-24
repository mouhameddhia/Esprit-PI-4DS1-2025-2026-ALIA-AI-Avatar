"""Shared NLP taxonomy loader.

Both the pipeline (nlp.py) and the evaluator (evaluator.py) need the same
taxonomy JSON.  This module provides a single authoritative loader so the
file path, error handling, and fallback values are defined only once.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).resolve().parent / "nlp_taxonomy.json"

# Minimal built-in fallback used when the JSON file cannot be read.
_FALLBACK_TAXONOMY: Dict[str, Any] = {
    "intents": [
        "product_information_request",
        "dosage_question",
        "safety_question",
        "objection_handling",
        "training_simulation",
        "crm_follow_up",
        "competency_assessment",
        "visit_format_request",
        "sales_methodology_request",
        "general_greeting",
        "other",
    ],
    "competency_levels": ["Debutant", "Junior", "Confirme", "Expert"],
    "visit_phases": [],
    "visit_formats": [],
    "objection_types": [],
    "safety_flags": ["patient_specific_advice_request", "diagnosis_request"],
    "entity_types": [],
    "promotion_thresholds": {},
}


def load_taxonomy() -> Dict[str, Any]:
    """Load the NLP taxonomy JSON file.

    Returns the parsed taxonomy dict on success, or a copy of the built-in
    fallback on failure so callers never receive None or an exception.
    """
    try:
        with _TAXONOMY_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load NLP taxonomy from %s: %s", _TAXONOMY_PATH, exc)
        return _FALLBACK_TAXONOMY.copy()
