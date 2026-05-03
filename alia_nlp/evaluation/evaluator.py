"""Conversation competency evaluator — unchanged logic, updated imports."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "taxonomy" / "nlp_taxonomy.json"


def _load_taxonomy() -> Dict[str, Any]:
    try:
        with _TAXONOMY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Taxonomy load failed: %s", exc)
        return {"competency_levels": ["Debutant", "Junior", "Confirme", "Expert"]}


TAXONOMY = _load_taxonomy()


def _contains_any(text: str, phrases: List[str]) -> bool:
    lower = text.lower()
    return any(p.lower() in lower for p in phrases)


def _score_opening(first_message: str) -> float:
    signals = ["bonjour", "hello", "hi", "je fais tres court", "c'est ok", "permission", "time"]
    return 10.0 if _contains_any(first_message, signals) else 6.0


def _score_discovery(messages: List[Dict[str, Any]]) -> float:
    q = sum((m.get("content") or "").count("?") for m in messages if m.get("role") == "assistant")
    return 10.0 if q >= 5 else 8.5 if q >= 3 else 7.0 if q >= 2 else 5.5 if q >= 1 else 3.0


def _score_synthesis(summary: str) -> float:
    if not summary:
        return 0.0
    return 10.0 if _contains_any(summary, ["si je resume", "si je résume", "en synthese"]) else 8.0


def _score_objection_handling(metadata: Dict, messages: List[Dict]) -> float:
    objections = metadata.get("objections") or []
    if not objections:
        return 5.0
    asst = " ".join((m.get("content") or "") for m in messages if m.get("role") == "assistant")
    score = 5.0 + min(5.0, len(objections) * 1.5)
    if _contains_any(asst, ["je comprends", "si vous dites", "cela repond", "clarifier"]):
        score += 1.0
    return min(10.0, score)


def _score_argumentation(summary: str, metadata: Dict) -> float:
    score = 5.0 + min(3.0, len(metadata.get("topics") or []) * 0.75)
    score += min(2.0, len(metadata.get("action_items") or []) * 0.5)
    if _contains_any(summary, ["benefice", "preuve", "usage", "avantage"]):
        score += 1.0
    return min(10.0, score)


def _score_closing(metadata: Dict, summary: str) -> float:
    items = metadata.get("action_items") or []
    if not items:
        return 4.0
    score = 6.0 + min(4.0, len(items) * 1.0)
    if _contains_any(summary, ["repasse", "retour", "essayer", "prochain passage"]):
        score += 0.5
    return min(10.0, score)


def _score_crm(summary: str, metadata: Dict) -> float:
    if not summary:
        return 0.0
    score = 6.0
    if metadata.get("topics"):   score += 1.5
    if metadata.get("action_items"): score += 1.5
    return min(10.0, score)


def _score_safety(nlp_events: List[Dict], summary: str) -> float:
    flags = [f for ev in nlp_events for f in (ev.get("safety_flags") or [])]
    if flags:
        return 6.0
    return 10.0 if _contains_any(summary, ["je verifie", "source", "officiel", "label"]) else 9.0


def _score_adaptation(metadata: Dict, nlp_events: List[Dict]) -> float:
    topics   = set(metadata.get("topics") or [])
    entities = {e for ev in nlp_events for e in (ev.get("entities") or [])}
    score = 5.0
    if len(topics)   >= 2: score += 2.0
    if len(entities) >= 3: score += 2.0
    if any(ev.get("intent") == "training_simulation" for ev in nlp_events):
        score += 1.0
    return min(10.0, score)


def _score_retrieval_grounding(metadata: Dict, nlp_events: List[Dict]) -> float:
    if not nlp_events:
        return 0.0
    base = 7.0 if any((ev.get("rewritten_query") or "").strip() for ev in nlp_events) else 4.0
    if metadata.get("topics"):       base += 1.5
    if metadata.get("action_items"): base += 1.0
    return min(10.0, base)


def _infer_level(score: float, dims: Dict[str, float], messages: List[Dict]) -> str:
    if score >= 9.0 and len(messages) >= 8 and dims.get("safety", 0) >= 9.0:
        return "Expert"
    if score >= 8.0 and dims.get("objection_handling", 0) >= 7.0 and dims.get("adaptation", 0) >= 7.0:
        return "Confirme"
    if score >= 7.0:
        return "Junior"
    return "Debutant"


def evaluate_conversation(
    messages: List[Dict[str, Any]],
    summary: str,
    metadata: Dict[str, Any],
    nlp_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    nlp_events = nlp_events or []
    first = (messages[0].get("content") if messages else "") or ""

    dims = {
        "opening":            _score_opening(first),
        "discovery":          _score_discovery(messages),
        "synthesis":          _score_synthesis(summary),
        "objection_handling": _score_objection_handling(metadata, messages),
        "argumentation":      _score_argumentation(summary, metadata),
        "closing":            _score_closing(metadata, summary),
        "crm":                _score_crm(summary, metadata),
        "safety":             _score_safety(nlp_events, summary),
        "adaptation":         _score_adaptation(metadata, nlp_events),
        "retrieval_grounding":_score_retrieval_grounding(metadata, nlp_events),
    }

    score = round(mean(dims.values()), 1)
    level = _infer_level(score, dims, messages)

    notes: List[str] = []
    if dims.get("safety", 0) < 7.0:       notes.append("Safety/compliance needs attention")
    if dims.get("crm", 0) < 7.0:          notes.append("CRM traceability is weak")
    if dims.get("closing", 0) < 7.0:      notes.append("Closing and commitment need reinforcement")

    return {
        "level": level, "score": score, "dimensions": dims,
        "strengths": [k for k, v in dims.items() if v >= 8.0],
        "gaps":      [k for k, v in dims.items() if v < 7.0],
        "notes": notes,
        "evaluated_at": datetime.utcnow(),
    }
