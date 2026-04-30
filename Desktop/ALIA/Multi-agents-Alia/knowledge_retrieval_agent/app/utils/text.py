"""Text normalization utilities for scientific and pharmaceutical retrieval."""

from __future__ import annotations

import re


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "if", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "with", "without", "which", "what", "when", "where",
    "who", "whom", "why", "how", "can", "could", "may", "might", "should", "would",
}

_TOKEN_PATTERN = re.compile(r"[\w]+(?:/[\w]+)*(?:-[\w]+)*(?:[α-ωΑ-Ω]+)?", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize unicode punctuation and whitespace while preserving terminology."""

    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "×": "x",
        "µ": "u",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def tokenize_scientific_text(text: str) -> list[str]:
    """Tokenize scientific text with light normalization and stopword removal."""

    normalized = normalize_text(text.lower())
    tokens = _TOKEN_PATTERN.findall(normalized)
    return [token for token in tokens if token not in _STOPWORDS]