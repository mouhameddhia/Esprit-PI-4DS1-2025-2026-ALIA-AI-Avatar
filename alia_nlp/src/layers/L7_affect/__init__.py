from typing import Any, Dict

from alia_nlp.src.layers.L7_affect.extractor import extract_from_llm
from alia_nlp.src.layers.L7_affect.rules import infer_from_rules
from alia_nlp.src.layers.L7_affect.schema import AffectResult


def analyze_affect(text: str, parsed_llm: Dict[str, Any], mode: str) -> AffectResult:
    """Return AffectResult from LLM output when available, rules otherwise."""
    if parsed_llm and isinstance(parsed_llm.get("affect"), dict):
        return extract_from_llm(parsed_llm, mode)
    return infer_from_rules(text, mode)
