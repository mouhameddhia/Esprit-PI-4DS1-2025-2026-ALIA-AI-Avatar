"""Merge LLM entity_map with rule-based extraction."""

from typing import Any, Dict, List

from alia_nlp.data.taxonomy.loader import ENTITY_TYPES
from alia_nlp.src.layers.L3_entities.normalizer import normalize_entity_map, merge_maps


def extract(
    parsed_llm: Dict[str, Any],
    user_text: str,
) -> Dict[str, List[str]]:
    """Merge LLM output with rule-based map, rules acting as complement."""
    from alia_nlp.src.layers.L3_entities.rules import extract_rules

    llm_map = normalize_entity_map(parsed_llm.get("entity_map"))
    rule_map = extract_rules(user_text)
    return merge_maps(llm_map, rule_map)


def flatten(entity_map: Dict[str, List[str]], extra: List[str]) -> List[str]:
    seen: list = list(extra)
    for values in entity_map.values():
        for v in values:
            if v and v not in seen:
                seen.append(v)
    return seen[:20]
