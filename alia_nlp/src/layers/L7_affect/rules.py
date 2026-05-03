import re
from typing import Set

from alia_nlp.src.layers.L7_affect.schema import AffectResult

# ── Hedge words (low confidence signal) ────────────────────────────────────
_HEDGES: Set[str] = {
    # English
    "maybe", "perhaps", "i think", "i guess", "i'm not sure", "im not sure",
    "not sure", "kind of", "sort of", "probably", "possibly", "i believe",
    "i suppose", "i'm unsure", "im unsure",
    # French
    "peut-être", "je pense", "je crois", "je suppose", "pas sûr",
    "pas certain", "probablement", "possiblement", "j'imagine",
}

# ── Frustration markers ─────────────────────────────────────────────────────
_FRUSTRATION: Set[str] = {
    # English
    "i don't know", "i dont know", "no idea", "i give up", "i can't do this",
    "i cannot do this", "stuck", "i'm confused", "im confused", "i'm lost",
    "im lost", "help me", "i don't understand", "i dont understand",
    # French
    "je sais pas", "je ne sais pas", "aucune idée", "j'abandonne",
    "je comprends pas", "je ne comprends pas", "je suis perdu", "perdu",
    "aidez-moi", "je suis bloqué",
}

# ── High engagement markers ─────────────────────────────────────────────────
_HIGH_ENGAGEMENT: Set[str] = {
    # English
    "let me try", "i'll try", "simulate", "let's practice", "challenge me",
    "give me a harder", "i want to master", "teach me", "let's go deeper",
    "more difficult", "harder scenario", "push me", "i'm ready",
    # French
    "je vais essayer", "simulons", "pratiquons", "défie-moi",
    "scénario plus difficile", "je veux maîtriser", "allons plus loin",
    "je suis prêt",
}

# ── Urgency markers (physician mode) ───────────────────────────────────────
_URGENCY: Set[str] = {
    "urgent", "urgently", "immediately", "right now", "emergency",
    "critical", "asap", "as soon as possible", "right away", "stat",
    "d'urgence", "immédiatement", "tout de suite", "dès que possible",
}
_ELEVATED: Set[str] = {
    "important", "serious", "concerning", "worried", "worried about",
    "need to know", "need to check", "need to verify", "have to know",
    "sérieux", "inquiet", "préoccupant", "besoin de savoir",
    "besoin de vérifier",
}


def _t(text: str) -> str:
    return text.lower().strip()


def _contains_any(text: str, terms: Set[str]) -> bool:
    t = _t(text)
    return any(term in t for term in terms)


def _hedge_ratio(text: str) -> float:
    t = _t(text)
    token_count = max(len(text.split()), 1)
    hits = sum(1 for h in _HEDGES if h in t)
    return hits / (token_count / 8)


def infer_from_rules(text: str, mode: str) -> AffectResult:
    tokens = text.split()
    token_count = len(tokens)

    # ── Confidence ──────────────────────────────────────────────────────────
    hedge_ratio = _hedge_ratio(text)
    if _contains_any(text, _FRUSTRATION) or hedge_ratio >= 1.5 or token_count < 4:
        rep_confidence = "low"
    elif hedge_ratio >= 0.6:
        rep_confidence = "medium"
    else:
        rep_confidence = "high"

    # ── Frustration ─────────────────────────────────────────────────────────
    frustration_signal = _contains_any(text, _FRUSTRATION)

    # ── Engagement ──────────────────────────────────────────────────────────
    if _contains_any(text, _HIGH_ENGAGEMENT):
        engagement_level = "highly_engaged"
    elif token_count >= 15:
        engagement_level = "active"
    else:
        engagement_level = "passive"

    # ── Urgency (physician only) ─────────────────────────────────────────────
    query_urgency = "routine"
    if mode == "physician_portal":
        if _contains_any(text, _URGENCY):
            query_urgency = "urgent"
        elif _contains_any(text, _ELEVATED):
            query_urgency = "elevated"

    return AffectResult(
        rep_confidence=rep_confidence,
        frustration_signal=frustration_signal,
        engagement_level=engagement_level,
        query_urgency=query_urgency,
        affect_source="rules",
    )
