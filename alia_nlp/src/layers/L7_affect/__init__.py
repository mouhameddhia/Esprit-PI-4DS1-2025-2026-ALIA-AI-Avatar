from typing import Any, Dict

from alia_nlp.src.layers.L7_affect.classifier import predict as model_predict
from alia_nlp.src.layers.L7_affect.extractor import extract_from_llm
from alia_nlp.src.layers.L7_affect.rules import infer_from_rules
from alia_nlp.src.layers.L7_affect.schema import AffectResult


def analyze_affect(
    text: str,
    parsed_llm: Dict[str, Any],
    mode: str,
    prev_turn: str = "",
) -> AffectResult:
    """
    Priority: fine-tuned model → LLM output → heuristic rules.
    prev_turn: last message from history — passed to the model for multi-turn context.
    """
    result = model_predict(text, mode, prev_turn=prev_turn)
    if result is not None:
        return result

    if parsed_llm and isinstance(parsed_llm.get("affect"), dict):
        return extract_from_llm(parsed_llm, mode)

    return infer_from_rules(text, mode)
