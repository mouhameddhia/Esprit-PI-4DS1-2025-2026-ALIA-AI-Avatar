"""Lightweight language detection for runtime NLP analytics and routing."""

import re
import unicodedata

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_ARABIZI_RE = re.compile(r"\b[\w']*[23456789][\w']*\b", re.IGNORECASE)

_FRENCH_MARKERS = {
    "bonjour",
    "salut",
    "merci",
    "avec",
    "pour",
    "contre",
    "effets",
    "indication",
    "dose",
    "posologie",
    "docteur",
    "medecin",
    "medecins",
    "patient",
    "patients",
    "bonjour",
    "efficacite",
    "tolerance",
    "pouvez",
    "quels",
    "quelle",
    "visite",
    "cliniques",
    "repondre",
    "question",
    "donnez",
    "minutes",
    "traitement",
    "effets",
    "secondaires",
    "securite",
    "innocuite",
    "efficacite",
    "contreindication",
    "interactions",
}

_ENGLISH_MARKERS = {
    "hello",
    "thanks",
    "please",
    "dose",
    "dosing",
    "safety",
    "contraindication",
    "interaction",
    "efficacy",
    "doctor",
    "patient",
    "training",
    "objection",
    "visit",
    "tolerability",
    "clinical",
    "evidence",
    "trial",
    "recommended",
    "recommended",
    "dosage",
}

_ARABIZI_MARKERS = {
    "marhaba",
    "marhba",
    "salam",
    "shukran",
    "chokran",
    "docteur",
    "doktour",
    "dose",
    "jar3a",
    "jar3ah",
    "3la",
    "3andak",
    "mashi",
    "bghit",
    "chno",
    "salam",
}

_FRENCH_FUNCTION_WORDS = {
    "le",
    "la",
    "les",
    "de",
    "des",
    "du",
    "un",
    "une",
    "et",
    "ou",
    "pour",
    "avec",
    "sur",
    "dans",
    "que",
    "qui",
    "vous",
    "nous",
    "je",
    "est",
    "sont",
    "pas",
    "plus",
}

_ENGLISH_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "on",
    "in",
    "to",
    "from",
    "you",
    "we",
    "is",
    "are",
    "not",
    "can",
    "should",
    "what",
    "how",
}

_FRENCH_ACCENTS = set("àâçéèêëîïôùûüÿœæ")


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9']+", text.lower())


def _normalized_tokens(text: str) -> list[str]:
    tokens = _tokenize(text)
    out: list[str] = []
    for token in tokens:
        cleaned = _strip_diacritics(token).replace("-", "").strip("'")
        if cleaned:
            out.append(cleaned)
    return out


def _score_markers(tokens: list[str], markers: set[str]) -> int:
    return sum(1 for token in tokens if token in markers)


def _contains_arabizi(text: str, tokens: list[str]) -> bool:
    if _ARABIZI_RE.search(text):
        return True
    return any(token in _ARABIZI_MARKERS for token in tokens)


def detect_language(text: str) -> str:
    """Return coarse language label: en, fr, ar, or unknown."""
    value = (text or "").strip()
    if not value:
        return "unknown"

    if _ARABIC_CHAR_RE.search(value):
        return "ar"

    lowered = value.lower()
    tokens = _normalized_tokens(lowered)

    french_score = _score_markers(tokens, _FRENCH_MARKERS)
    english_score = _score_markers(tokens, _ENGLISH_MARKERS)
    arabizi_score = _score_markers(tokens, _ARABIZI_MARKERS)

    french_score += _score_markers(tokens, _FRENCH_FUNCTION_WORDS)
    english_score += _score_markers(tokens, _ENGLISH_FUNCTION_WORDS)

    accent_hits = sum(1 for char in lowered if char in _FRENCH_ACCENTS)
    french_score += min(2, accent_hits)

    if _contains_arabizi(lowered, tokens):
        arabizi_score += 2

    if arabizi_score >= max(2, french_score + 1, english_score + 1):
        return "ar"

    if french_score >= max(2, english_score + 1):
        return "fr"
    if english_score >= max(2, french_score + 1):
        return "en"

    if len(tokens) <= 2:
        return "unknown"

    if french_score > english_score:
        return "fr"
    if english_score > french_score:
        return "en"

    # Conservative fallback for ambiguous Latin-script messages.
    if tokens:
        return "unknown"
    return "unknown"
