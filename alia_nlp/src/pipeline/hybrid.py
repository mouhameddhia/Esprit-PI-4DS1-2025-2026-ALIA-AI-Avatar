"""Hybrid pipeline — LoRA adapter path (activated via USE_HYBRID_ADAPTER=1).

Merges fine-tuned adapter predictions with the online pipeline baseline.
Merge strategy:
  - intent: trust baseline unless baseline == "other", then use adapter
  - safety_flags, secondary_tags, entity_map: union (OR)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from alia_nlp.src.pipeline.online import analyze_message_nlp as baseline_analyze

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    return os.getenv("ALIA_USE_HYBRID_ADAPTER", "0").lower() in {"1", "true", "yes"}


def _union_list(a: List[str], b: List[str]) -> List[str]:
    seen: list = list(a)
    for v in b:
        if v not in seen:
            seen.append(v)
    return seen


def _union_entity_map(
    a: Dict[str, List[str]], b: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    keys = set(a) | set(b)
    return {k: _union_list(a.get(k, []), b.get(k, [])) for k in keys}


def analyze_hybrid(
    user_text: str,
    history: Optional[List[Dict[str, Any]]] = None,
    mode: str = "physician_portal",
    adapter_prediction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    baseline = baseline_analyze(user_text, history=history, mode=mode)

    if not adapter_prediction:
        return baseline

    adapter_intent = adapter_prediction.get("intent", "other")
    merged_intent = baseline["intent"] if baseline["intent"] != "other" else adapter_intent

    return {
        **baseline,
        "intent": merged_intent,
        "safety_flags": _union_list(
            adapter_prediction.get("safety_flags", []),
            baseline.get("safety_flags", []),
        ),
        "secondary_tags": _union_list(
            adapter_prediction.get("secondary_tags", []),
            baseline.get("secondary_tags", []),
        ),
        "entity_map": _union_entity_map(
            adapter_prediction.get("entity_map", {}),
            baseline.get("entity_map", {}),
        ),
        "confidence": adapter_prediction.get("confidence", baseline.get("confidence", 0.0)),
        "intent_source": "hybrid",
    }
