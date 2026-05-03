"""Real-time NLP pipeline — entry point for the backend.

Flow:
  L0 normalize → L1 language → L2 intent (rules → LLM) →
  L3 entities → L4 safety → L5 discourse → L6 scoring → NLPResult
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from alia_nlp.src.schema import NLPResult
from alia_nlp.data.taxonomy.loader import (
    ENTITY_TYPES, SUPPORTED_INTENTS, SUPPORTED_SAFETY_FLAGS, SUPPORTED_SECONDARY_TAGS,
)
from alia_nlp.src.layers.L0_preprocessing.normalizer import normalize
from alia_nlp.src.layers.L0_preprocessing.context import build_user_prompt
from alia_nlp.src.layers.L1_language.detector import detect_language
from alia_nlp.src.layers.L2_intent.classifier import classify, needs_llm
from alia_nlp.src.layers.L2_intent.llm import call_structured
from alia_nlp.src.layers.L3_entities.extractor import extract, flatten
from alia_nlp.src.layers.L4_safety.detector import detect as detect_safety
from alia_nlp.src.layers.L5_discourse.secondary_tags import infer as infer_tags
from alia_nlp.src.layers.L5_discourse.clarification import should_clarify
from alia_nlp.src.layers.L5_discourse.rewriter import rewrite
from alia_nlp.src.layers.L6_scoring.confidence import aggregate
from alia_nlp.src.layers.L6_scoring.explainability import build as build_explainability
from alia_nlp.src.layers.L7_affect import analyze_affect

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _load_prompt(mode: str) -> str:
    filename = "extraction_physician.txt" if mode == "physician_portal" else "extraction_medrep.txt"
    path = _PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to load prompt %s: %s", path, exc)
        return _fallback_prompt(mode)


def _fallback_prompt(mode: str) -> str:
    return (
        "You are an NLP extraction engine for a pharmaceutical assistant. "
        f"Mode: {mode}. "
        "Return ONLY valid JSON with keys: intent, needs_clarification, secondary_tags, "
        "entities, entity_map, topics, objections, action_items, safety_flags, "
        "rewritten_query, confidence."
    )


def _format_prompt(template: str, mode: str) -> str:
    from alia_nlp.data.taxonomy.loader import (
        SUPPORTED_INTENTS, SUPPORTED_SECONDARY_TAGS,
        ENTITY_TYPES, SUPPORTED_SAFETY_FLAGS,
    )
    mode_guidance = {
        "physician_portal": (
            "You are analyzing a query from a HEALTHCARE PROFESSIONAL. "
            "Clinical intents (product_information_request, dosage_question, safety_question) dominate. "
            "Training and simulation intents are unlikely."
        ),
        "medrep_training": (
            "You are analyzing a query from a MEDICAL REPRESENTATIVE practicing sales skills. "
            "Training, methodology, objection_handling, and visit intents are common."
        ),
    }.get(mode, "")
    try:
        return template.format(
            mode_guidance=mode_guidance,
            intents=", ".join(sorted(SUPPORTED_INTENTS)),
            secondary_tags=", ".join(sorted(SUPPORTED_SECONDARY_TAGS)) or "none",
            entity_types=", ".join(ENTITY_TYPES) or "none",
            safety_flags=", ".join(sorted(SUPPORTED_SAFETY_FLAGS)) or "none",
        )
    except KeyError:
        return template


def analyze(
    user_text: str,
    history: Optional[List[Dict[str, Any]]] = None,
    mode: str = "physician_portal",
) -> NLPResult:
    history = history or []
    user_text = normalize(user_text)

    if not user_text:
        return NLPResult.make_fallback()

    # L1 — language detection
    language = detect_language(user_text)

    # L2 — decide if LLM is needed
    parsed_llm: Dict[str, Any] = {}
    if needs_llm(user_text, mode):
        system_prompt = _format_prompt(_load_prompt(mode), mode)
        user_prompt = build_user_prompt(user_text, mode, history)
        parsed_llm = call_structured(system_prompt, user_prompt)

    # L2 — classify intent
    intent, intent_confidence, intent_source = classify(user_text, mode, parsed_llm)

    # L3 — entity extraction
    entity_map = extract(parsed_llm, user_text)
    entities = flatten(entity_map, _safe_list(parsed_llm.get("entities"), 12))

    # L4 — safety flags
    safety_flags = detect_safety(parsed_llm.get("safety_flags"), user_text)

    # L5 — secondary tags, clarification, query rewrite
    secondary_tags = infer_tags(user_text, intent, parsed_llm.get("secondary_tags"))
    needs_clarification = should_clarify(
        user_text, intent, intent_confidence, entity_map, mode,
        llm_flag=parsed_llm.get("needs_clarification") if parsed_llm else None,
    )
    rewritten_query = rewrite(parsed_llm, user_text)

    # L6 — confidence and explainability
    has_entities = any(v for v in entity_map.values() if v)
    confidence = aggregate(intent_confidence, has_entities, intent_source)
    explainability = build_explainability(
        user_text, intent, entity_map, secondary_tags, confidence, intent_source,
    )

    # L7 — affect analysis
    affect = analyze_affect(user_text, parsed_llm, mode)

    return NLPResult(
        intent=intent,
        needs_clarification=needs_clarification,
        confidence=confidence,
        secondary_tags=secondary_tags,
        entities=entities,
        safety_flags=safety_flags,
        topics=_safe_list(parsed_llm.get("topics"), 10),
        objections=_safe_list(parsed_llm.get("objections"), 8),
        action_items=_safe_list(parsed_llm.get("action_items"), 8),
        entity_map=entity_map,
        explainability=explainability,
        affect=affect.to_dict(),
        rewritten_query=rewritten_query,
        language=language,
        intent_source=intent_source,
    )


# Backward-compatible alias used by backend
def analyze_message_nlp(
    user_text: str,
    history: Optional[List[Dict[str, Any]]] = None,
    mode: str = "physician_portal",
) -> Dict[str, Any]:
    try:
        result = analyze(user_text, history=history, mode=mode)
        return result.to_dict()
    except Exception as exc:
        logger.warning("Pipeline error, returning fallback: %s", exc)
        return NLPResult.make_fallback(user_text, language=detect_language(user_text)).to_dict()


def _safe_list(value: Any, limit: int = 10) -> List[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()][:limit]
