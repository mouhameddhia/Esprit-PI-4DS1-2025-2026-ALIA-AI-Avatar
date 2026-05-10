"""Language detection supporting English, French, Spanish, and Arabic."""

import re
from typing import FrozenSet

# Arabic is detected by Unicode script (U+0600–U+06FF) — more reliable
# than word lists because Arabic words don't tokenise well with Latin regex.
_ARABIC_RANGE = ("؀", "ۿ")

_FR_MARKERS: FrozenSet[str] = frozenset([
    "le", "la", "les", "un", "une", "des", "je", "tu", "il", "elle",
    "nous", "vous", "ils", "elles", "pas", "que", "qui", "ou", "et",
    "en", "sur", "bonjour", "merci", "comment", "pourquoi", "pour",
    "avec", "dans", "mais", "donc", "car", "aussi", "bien", "très",
    "plus", "est", "sont", "médecin", "produit", "visite", "objectif",
    "réponse", "convaincu", "cher", "relance", "sondage",
])

_EN_MARKERS: FrozenSet[str] = frozenset([
    "the", "is", "are", "was", "were", "have", "has", "had", "how",
    "what", "why", "when", "where", "which", "hello", "thank", "please",
    "this", "that", "with", "from", "they", "their", "about", "would",
    "could", "should", "will", "can", "not", "and", "but", "for",
    "you", "your", "doctor", "patient", "product", "dose", "visit",
])

_ES_MARKERS: FrozenSet[str] = frozenset([
    "el", "la", "los", "las", "un", "una", "unos", "unas", "es", "son",
    "está", "están", "de", "del", "en", "con", "que", "qué", "cómo",
    "por", "para", "como", "pero", "también", "además", "sin", "sobre",
    "médico", "producto", "visita", "dosis", "seguridad", "paciente",
    "eficacia", "indicación", "medicamento", "tratamiento", "hola",
    "gracias", "buenas",
])


def _has_arabic_script(text: str) -> bool:
    lo, hi = _ARABIC_RANGE
    return any(lo <= ch <= hi for ch in text)


def detect_language(text: str) -> str:
    """Return ISO 639-1 code: 'ar' | 'fr' | 'en' | 'es' | 'unknown'."""
    if _has_arabic_script(text):
        return "ar"

    tokens = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())
    if not tokens:
        return "unknown"

    fr = sum(1 for t in tokens if t in _FR_MARKERS)
    en = sum(1 for t in tokens if t in _EN_MARKERS)
    es = sum(1 for t in tokens if t in _ES_MARKERS)

    best = max(fr, en, es)
    if best == 0:
        return "unknown"
    if fr == best:
        return "fr"
    if en == best:
        return "en"
    return "es"
