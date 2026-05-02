"""Intent and drug-name detection for EN/FR user questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

TOPIC_ALIASES: dict[str, str] = {
    "indication": "indications",
    "indications": "indications",
    "indique": "indications",
    "indiquee": "indications",
    "indiquees": "indications",
    "composition": "composition",
    "ingredient": "composition",
    "ingredients": "composition",
    "dosage": "dosage",
    "dose": "dosage",
    "dosing": "dosage",
    "posologie": "dosage",
    "administration": "administration",
    "administrer": "administration",
    "utilisation": "administration",
    "mode d emploi": "administration",
    "warnings": "warnings",
    "warning": "warnings",
    "precaution": "warnings",
    "precautions": "warnings",
    "contre indication": "warnings",
    "side effects": "side_effects",
    "side effect": "side_effects",
    "effet secondaire": "side_effects",
    "effets secondaires": "side_effects",
    "mechanism": "mechanism_of_action",
    "mecanisme": "mechanism_of_action",
    "mechanism of action": "mechanism_of_action",
}

SUPPORTED_TOPICS = {
    "indications",
    "composition",
    "dosage",
    "administration",
    "warnings",
    "side_effects",
    "mechanism_of_action",
    "age",
}


@dataclass
class DetectedIntent:
    """Detected routing signals for the orchestration layer."""

    question: str
    topic: str | None
    drug_name: str | None


def _normalize(text: str) -> str:
    lowered = text.lower().replace("'", " ")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(cleaned.split())


def _score_match(question: str, candidate: str) -> float:
    q = _normalize(question)
    c = _normalize(candidate)
    if not c:
        return 0.0
    if c in q:
        return 1.0

    # Fuzzy overlap for slight spelling deviations.
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    if not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens) / float(len(c_tokens))
    return overlap if overlap >= 0.6 else 0.0


def _contains_alias(normalized_question: str, alias: str) -> bool:
    tokens = alias.split()
    if not tokens:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in tokens) + r"\b"
    return re.search(pattern, normalized_question) is not None


def detect_topic(question: str) -> str | None:
    """Infer canonical topic from natural language text (EN/FR)."""

    q = _normalize(question)
    for alias, topic in TOPIC_ALIASES.items():
        if _contains_alias(q, alias):
            return topic

    if _contains_alias(q, "age"):
        return "age"
    return None


def detect_drug_name(question: str, candidates: list[str]) -> str | None:
    """Pick best matching drug from known candidates."""

    best_name: str | None = None
    best_score = 0.0
    for candidate in candidates:
        score = _score_match(question, candidate)
        if score > best_score:
            best_name = candidate
            best_score = score
    return best_name


def detect_intent(question: str, drug_candidates: list[str]) -> DetectedIntent:
    """Return detected drug + topic used by controller routing logic."""

    topic = detect_topic(question)
    if topic not in SUPPORTED_TOPICS:
        topic = None
    drug_name = detect_drug_name(question, drug_candidates)
    return DetectedIntent(question=question, topic=topic, drug_name=drug_name)
