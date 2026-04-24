"""Entity value normalization helpers shared by runtime and evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

_NORMALIZATION_PATH = Path(__file__).resolve().with_name("entity_normalization.json")


def _load_normalization_map() -> Dict[str, Dict[str, str]]:
    try:
        payload = json.loads(_NORMALIZATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for entity_type, canonical_map in payload.items():
        if not isinstance(entity_type, str) or not isinstance(canonical_map, dict):
            continue

        normalized_entity_type = entity_type.strip().lower()
        out[normalized_entity_type] = {}

        for canonical, aliases in canonical_map.items():
            if not isinstance(canonical, str):
                continue
            canonical_norm = canonical.strip().lower()
            if not canonical_norm:
                continue

            out[normalized_entity_type][canonical_norm] = canonical_norm
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip():
                        out[normalized_entity_type][alias.strip().lower()] = canonical_norm

    return out


ENTITY_NORMALIZATION = _load_normalization_map()


def normalize_entity_value(entity_type: str, value: str) -> str:
    """Map entity value variants to canonical form for stable matching."""
    normalized_value = (value or "").strip().lower()
    if not normalized_value:
        return ""

    entity_key = (entity_type or "").strip().lower()
    mappings = ENTITY_NORMALIZATION.get(entity_key, {})
    return mappings.get(normalized_value, normalized_value)
