import re
from typing import Set

from alia_nlp.src.layers.L7_affect.schema import AffectResult

# ── Hedge words (low confidence) ────────────────────────────────────────────
_HEDGES: Set[str] = {
    "maybe", "perhaps", "i think", "i guess", "i'm not sure", "im not sure",
    "not sure", "kind of", "sort of", "probably", "possibly", "i believe",
    "i suppose", "i'm unsure", "im unsure",
    "peut-être", "je pense", "je crois", "je suppose", "pas sûr",
    "pas certain", "probablement", "possiblement", "j'imagine",
}

# ── Frustration (giving up / disengaging) ───────────────────────────────────
_FRUSTRATION: Set[str] = {
    "i don't know", "i dont know", "no idea", "i give up", "i can't do this",
    "i cannot do this", "i'm confused", "im confused", "i'm lost", "im lost",
    "help me", "i don't understand", "i dont understand", "stuck",
    "je sais pas", "je ne sais pas", "aucune idée", "j'abandonne",
    "je comprends pas", "je ne comprends pas", "je suis perdu", "perdu",
    "aidez-moi", "je suis bloqué", "je n'y arrive pas",
}

# ── Stress (pressure / overload — distinct from frustration) ────────────────
_TIME_PRESSURE: Set[str] = {
    "quickly", "fast", "hurry", "right now", "i need to", "asap",
    "right away", "no time", "running out of time", "time is up",
    "vite", "rapidement", "tout de suite", "j'ai besoin", "pas le temps",
    "dépêche", "en urgence", "le plus vite",
}

# ── Urgency (physician mode) ────────────────────────────────────────────────
_URGENCY: Set[str] = {
    "urgent", "urgently", "immediately", "right now", "emergency",
    "critical", "asap", "as soon as possible", "right away", "stat",
    "d'urgence", "immédiatement", "tout de suite", "dès que possible",
}
_ELEVATED: Set[str] = {
    "important", "serious", "concerning", "worried", "need to know",
    "need to check", "need to verify", "have to know",
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
    tokens = max(len(text.split()), 1)
    hits = sum(1 for h in _HEDGES if h in t)
    return hits / (tokens / 8)


def _question_count(text: str) -> int:
    return text.count("?")


def _avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return len(text.split())
    return sum(len(s.split()) for s in sentences) / len(sentences)


def _fragmented(text: str) -> bool:
    return text.count("...") >= 2 or text.count("--") >= 2 or "???" in text


def infer_from_rules(text: str, mode: str) -> AffectResult:
    tokens    = text.split()
    n_tokens  = len(tokens)
    q_count   = _question_count(text)
    hedge_r   = _hedge_ratio(text)
    avg_sent  = _avg_sentence_length(text)

    # ── Confidence ──────────────────────────────────────────────────────────
    is_frustrated = _contains_any(text, _FRUSTRATION)
    if is_frustrated or hedge_r >= 1.5 or n_tokens < 4:
        rep_confidence = "low"
    elif hedge_r >= 0.6:
        rep_confidence = "medium"
    else:
        rep_confidence = "high"

    # ── Frustration ─────────────────────────────────────────────────────────
    frustration_signal = is_frustrated

    # ── Stress ──────────────────────────────────────────────────────────────
    # Multiple stacked questions, time pressure words, or very short fragmented bursts
    stress_signal = (
        q_count >= 3
        or (q_count >= 2 and _contains_any(text, _TIME_PRESSURE))
        or _fragmented(text)
        or (avg_sent < 5 and q_count >= 2 and n_tokens >= 10)
        or (_contains_any(text, _TIME_PRESSURE) and n_tokens >= 8)
    )

    # ── Engagement (binary) ─────────────────────────────────────────────────
    engagement_level = "passive" if n_tokens < 10 else "engaged"

    # ── Urgency (physician only) ─────────────────────────────────────────────
    query_urgency = "routine"
    if mode == "physician_portal":
        if _contains_any(text, _URGENCY) or stress_signal:
            query_urgency = "urgent"
        elif _contains_any(text, _ELEVATED):
            query_urgency = "elevated"

    return AffectResult(
        rep_confidence=rep_confidence,
        frustration_signal=frustration_signal,
        stress_signal=stress_signal,
        engagement_level=engagement_level,
        query_urgency=query_urgency,
        affect_source="rules",
    )
