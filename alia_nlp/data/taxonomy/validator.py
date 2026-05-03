"""Validation helpers for NLP taxonomy-aligned labels and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set


def load_taxonomy(taxonomy_path: Path | None = None) -> Dict[str, Any]:
    if taxonomy_path is None:
        taxonomy_path = Path(__file__).resolve().with_name("nlp_taxonomy.json")
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def allowed_secondary_tags(taxonomy: Dict[str, Any]) -> Set[str]:
    tags: Set[str] = set()
    for key in ("visit_phases", "visit_formats", "objection_types", "auxiliary_tags"):
        values = taxonomy.get(key) or []
        tags.update(value for value in values if isinstance(value, str))
    return tags


def _as_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _entity_types(taxonomy: Dict[str, Any]) -> Set[str]:
    return {item for item in (taxonomy.get("entity_types") or []) if isinstance(item, str)}


def validate_dataset_row_labels(row: Dict[str, Any], taxonomy: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    intents = {item for item in (taxonomy.get("intents") or []) if isinstance(item, str)}
    intent = row.get("expected_intent")
    if isinstance(intent, str) and intent not in intents:
        issues.append(f"invalid expected_intent: {intent}")

    safety_flags = {item for item in (taxonomy.get("safety_flags") or []) if isinstance(item, str)}
    for flag in _as_list(row.get("expected_safety_flags", [])):
        if flag not in safety_flags:
            issues.append(f"invalid expected_safety_flag: {flag}")

    secondary_tags = allowed_secondary_tags(taxonomy)
    for tag in _as_list(row.get("expected_secondary_tags", [])):
        if tag not in secondary_tags:
            issues.append(f"invalid expected_secondary_tag: {tag}")

    entity_types = _entity_types(taxonomy)
    expected_entity_map = row.get("expected_entity_map", {})
    if expected_entity_map and not isinstance(expected_entity_map, dict):
        issues.append("expected_entity_map must be an object")
    elif isinstance(expected_entity_map, dict):
        for key, value in expected_entity_map.items():
            if key not in entity_types:
                issues.append(f"invalid expected_entity_map key: {key}")
                continue
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                issues.append(f"expected_entity_map[{key}] must be a string list")

    return issues


def validate_nlp_output(output: Dict[str, Any], taxonomy: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    intent = output.get("intent")
    intents = {item for item in (taxonomy.get("intents") or []) if isinstance(item, str)}
    if not isinstance(intent, str) or intent not in intents:
        issues.append("invalid output intent")

    for key in ("entities", "topics", "objections", "action_items", "safety_flags", "secondary_tags"):
        value = output.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            issues.append(f"{key} must be a string list")

    safety_flags = {item for item in (taxonomy.get("safety_flags") or []) if isinstance(item, str)}
    for flag in _as_list(output.get("safety_flags", [])):
        if flag not in safety_flags:
            issues.append(f"unsupported safety flag: {flag}")

    secondary = allowed_secondary_tags(taxonomy)
    for tag in _as_list(output.get("secondary_tags", [])):
        if tag not in secondary:
            issues.append(f"unsupported secondary tag: {tag}")

    entity_types = _entity_types(taxonomy)
    entity_map = output.get("entity_map", {})
    if not isinstance(entity_map, dict):
        issues.append("entity_map must be an object")
    else:
        for key, value in entity_map.items():
            if key not in entity_types:
                issues.append(f"unsupported entity_map key: {key}")
                continue
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                issues.append(f"entity_map[{key}] must be a string list")

    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)):
        issues.append("confidence must be numeric")
    else:
        if confidence < 0 or confidence > 1:
            issues.append("confidence must be between 0 and 1")

    rewritten_query = output.get("rewritten_query")
    if not isinstance(rewritten_query, str):
        issues.append("rewritten_query must be a string")

    return issues
