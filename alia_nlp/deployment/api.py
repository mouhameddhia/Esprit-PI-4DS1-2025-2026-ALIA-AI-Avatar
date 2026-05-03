"""FastAPI integration layer — thin wrapper around src/pipeline/online.py.

This module is the only entry point the backend should import from
when using the NLP pipeline. It never imports from training/, evaluation/,
or preprocessing/.
"""

from typing import Any, Dict, List, Optional

from alia_nlp.src.pipeline.online import analyze_message_nlp
from alia_nlp.src.pipeline.hybrid import analyze_hybrid, is_enabled as hybrid_enabled
from alia_nlp.data.taxonomy.loader import SUPPORTED_INTENTS
from alia_nlp.evaluation.evaluator import evaluate_conversation


def analyze(
    user_text: str,
    history: Optional[List[Dict[str, Any]]] = None,
    mode: str = "physician_portal",
    adapter_prediction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Single entry point for all NLP analysis in the backend.
    Routes to hybrid pipeline when enabled and adapter prediction is provided.
    """
    if hybrid_enabled() and adapter_prediction:
        return analyze_hybrid(
            user_text, history=history, mode=mode,
            adapter_prediction=adapter_prediction,
        )
    return analyze_message_nlp(user_text, history=history, mode=mode)


__all__ = [
    "analyze",
    "evaluate_conversation",
    "SUPPORTED_INTENTS",
]
