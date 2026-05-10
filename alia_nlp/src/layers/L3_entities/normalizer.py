"""Entity map validation, normalisation, and merging."""

from typing import Any, Dict, List

from alia_nlp.data.taxonomy.loader import ENTITY_TYPES


def _safe_list(value: Any, limit: int = 10) -> List[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()][:limit]


def normalize_entity_map(raw: Any) -> Dict[str, List[str]]:
    """Validate keys against taxonomy; discard unknown keys."""
    out: Dict[str, List[str]] = {t: [] for t in ENTITY_TYPES}
    if not isinstance(raw, dict):
        return out
    for key, values in raw.items():
        if isinstance(key, str) and key in out:
            out[key] = _safe_list(values, limit=10)
    return out


def merge_maps(
    primary: Dict[str, List[str]],
    secondary: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Union primary and secondary, deduplicate, cap at 12 per type."""
    merged: Dict[str, List[str]] = {t: [] for t in ENTITY_TYPES}
    for t in ENTITY_TYPES:
        combined = (primary.get(t) or []) + (secondary.get(t) or [])
        seen: list = []
        for v in combined:
            if v and v not in seen:
                seen.append(v)
        merged[t] = seen[:12]
    return merged
