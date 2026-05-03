"""Precision, recall, and accuracy calculations."""

from typing import Any, Set


def _safe_set(values: Any) -> Set[str]:
    if not isinstance(values, list):
        return set()
    return {v for v in values if isinstance(v, str)}


def _entity_pairs(entity_map: Any) -> Set[str]:
    if not isinstance(entity_map, dict):
        return set()
    pairs: Set[str] = set()
    for key, values in entity_map.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        for v in values:
            if isinstance(v, str) and v.strip():
                pairs.add(f"{key}::{v.strip().lower()}")
    return pairs


def precision_recall(tp: int, fp: int, fn: int):
    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall    = (tp / (tp + fn)) if (tp + fn) else 1.0
    return round(precision, 4), round(recall, 4)
