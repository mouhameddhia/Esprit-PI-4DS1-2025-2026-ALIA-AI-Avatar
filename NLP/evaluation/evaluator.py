"""Competency evaluation utilities for ALIA conversations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "taxonomy" / "nlp_taxonomy.json"


def _load_taxonomy() -> Dict[str, Any]:
    try:
        with _TAXONOMY_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to load NLP taxonomy: {exc}")
        return {
            "competency_levels": ["Debutant", "Junior", "Confirme", "Expert"],
            "promotion_thresholds": {},
        }


TAXONOMY = _load_taxonomy()


def _contains_any(text: str, phrases: List[str]) -> bool:
    lower = text.lower()
    return any(phrase.lower() in lower for phrase in phrases)


def _score_opening(first_message: str) -> float:
    if not first_message:
        return 0.0
    signals = ["bonjour", "hello", "hi", "je fais tres court", "c'est ok", "permission", "time"]
    return 10.0 if _contains_any(first_message, signals) else 6.0


def _score_discovery(messages: List[Dict[str, Any]]) -> float:
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    question_count = sum((m.get("content") or "").count("?") for m in assistant_messages)
    if question_count >= 5:
        return 10.0
    if question_count >= 3:
        return 8.5
    if question_count >= 2:
        return 7.0
    if question_count >= 1:
        return 5.5
    return 3.0


def _score_synthesis(summary: str) -> float:
    if not summary:
        return 0.0
    if _contains_any(summary, ["si je resume", "si je résume", "en synthese", "synthese"]):
        return 10.0
    return 8.0


def _score_objection_handling(metadata: Dict[str, Any], messages: List[Dict[str, Any]]) -> float:
    objections = metadata.get("objections") or []
    if not objections:
        return 5.0

    assistant_text = " ".join((m.get("content") or "") for m in messages if m.get("role") == "assistant")
    acrv_signals = ["je comprends", "si vous dites", "cela repond", "est-ce que cela", "clarifier", "valider"]
    score = 5.0 + min(5.0, len(objections) * 1.5)
    if _contains_any(assistant_text, acrv_signals):
        score += 1.0
    return min(10.0, score)


def _score_argumentation(summary: str, metadata: Dict[str, Any]) -> float:
    topics = metadata.get("topics") or []
    action_items = metadata.get("action_items") or []
    score = 5.0
    if topics:
        score += min(3.0, len(topics) * 0.75)
    if action_items:
        score += min(2.0, len(action_items) * 0.5)
    if _contains_any(summary, ["benefice", "preuve", "usage", "avantage"]):
        score += 1.0
    return min(10.0, score)


def _score_closing(metadata: Dict[str, Any], summary: str) -> float:
    action_items = metadata.get("action_items") or []
    if not action_items:
        return 4.0
    score = 6.0 + min(4.0, len(action_items) * 1.0)
    if _contains_any(summary, ["repasse", "retour", "essayer", "prochain passage", "suivi"]):
        score += 0.5
    return min(10.0, score)


def _score_crm(summary: str, metadata: Dict[str, Any]) -> float:
    if not summary:
        return 0.0
    topics = metadata.get("topics") or []
    action_items = metadata.get("action_items") or []
    score = 6.0
    if topics:
        score += 1.5
    if action_items:
        score += 1.5
    return min(10.0, score)


def _score_safety(nlp_events: List[Dict[str, Any]], summary: str) -> float:
    flags = []
    for event in nlp_events:
        flags.extend(event.get("safety_flags") or [])
    if flags:
        return 6.0
    if _contains_any(summary, ["je verifie", "je reviens", "source", "officiel", "label"]):
        return 10.0
    return 9.0


def _score_adaptation(metadata: Dict[str, Any], nlp_events: List[Dict[str, Any]]) -> float:
    topics = set(metadata.get("topics") or [])
    entities = set()
    for event in nlp_events:
        for item in event.get("entities") or []:
            entities.add(item)
    score = 5.0
    if len(topics) >= 2:
        score += 2.0
    if len(entities) >= 3:
        score += 2.0
    if any((event.get("intent") == "training_simulation") for event in nlp_events):
        score += 1.0
    return min(10.0, score)


def _score_retrieval_grounding(metadata: Dict[str, Any], nlp_events: List[Dict[str, Any]]) -> float:
    if not nlp_events:
        return 0.0
    rewritten = [event.get("rewritten_query") or "" for event in nlp_events]
    if any(query.strip() for query in rewritten):
        base = 7.0
    else:
        base = 4.0
    if metadata.get("topics"):
        base += 1.5
    if metadata.get("action_items"):
        base += 1.0
    return min(10.0, base)


def _infer_level(overall_score: float, dimensions: Dict[str, float], messages: List[Dict[str, Any]]) -> str:
    total_messages = len(messages)
    objection_score = dimensions.get("objection_handling", 0.0)
    adaptation_score = dimensions.get("adaptation", 0.0)
    safety_score = dimensions.get("safety", 0.0)

    if overall_score >= 9.0 and total_messages >= 8 and safety_score >= 9.0:
        return "Expert"
    if overall_score >= 8.0 and objection_score >= 7.0 and adaptation_score >= 7.0:
        return "Confirme"
    if overall_score >= 7.0:
        return "Junior"
    return "Debutant"


def evaluate_conversation(
    messages: List[Dict[str, Any]],
    summary: str,
    metadata: Dict[str, Any],
    nlp_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    nlp_events = nlp_events or []
    first_message = (messages[0].get("content") if messages else "") or ""

    dimensions = {
        "opening": _score_opening(first_message),
        "discovery": _score_discovery(messages),
        "synthesis": _score_synthesis(summary),
        "objection_handling": _score_objection_handling(metadata, messages),
        "argumentation": _score_argumentation(summary, metadata),
        "closing": _score_closing(metadata, summary),
        "crm": _score_crm(summary, metadata),
        "safety": _score_safety(nlp_events, summary),
        "adaptation": _score_adaptation(metadata, nlp_events),
        "retrieval_grounding": _score_retrieval_grounding(metadata, nlp_events),
    }

    overall_score = round(mean(dimensions.values()), 1) if dimensions else 0.0
    level = _infer_level(overall_score, dimensions, messages)

    strengths = [name for name, score in dimensions.items() if score >= 8.0]
    gaps = [name for name, score in dimensions.items() if score < 7.0]

    notes: List[str] = []
    if dimensions.get("safety", 0.0) < 7.0:
        notes.append("Safety/compliance needs attention")
    if dimensions.get("crm", 0.0) < 7.0:
        notes.append("CRM traceability is weak")
    if dimensions.get("closing", 0.0) < 7.0:
        notes.append("Closing and commitment need reinforcement")

    return {
        "level": level,
        "score": overall_score,
        "dimensions": dimensions,
        "strengths": strengths,
        "gaps": gaps,
        "notes": notes,
        "evaluated_at": datetime.utcnow(),
    }
