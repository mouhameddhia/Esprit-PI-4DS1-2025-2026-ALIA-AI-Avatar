"""Text normalization — runs before any classification."""

import re
import unicodedata


_MAX_CHARS = 2000


def normalize(text: str) -> str:
    """Unicode-normalize, collapse whitespace, hard-truncate."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_CHARS]
