from typing import Any, Dict

from alia_nlp.src.layers.L7_affect.schema import AffectResult

_VALID_CONFIDENCE = {"low", "medium", "high"}
_VALID_ENGAGEMENT = {"passive", "engaged"}
_VALID_URGENCY    = {"routine", "elevated", "urgent"}


def extract_from_llm(parsed_llm: Dict[str, Any], mode: str) -> AffectResult:
    raw = parsed_llm.get("affect", {})
    if not isinstance(raw, dict):
        return AffectResult()

    confidence = raw.get("rep_confidence", "medium")
    if confidence not in _VALID_CONFIDENCE:
        confidence = "medium"

    frustration = bool(raw.get("frustration_signal", False))
    stress      = bool(raw.get("stress_signal", False))

    engagement = raw.get("engagement_level", "engaged")
    # accept legacy 3-class values from old prompts
    if engagement in ("active", "highly_engaged"):
        engagement = "engaged"
    elif engagement not in _VALID_ENGAGEMENT:
        engagement = "engaged"

    urgency = raw.get("query_urgency", "routine")
    if urgency not in _VALID_URGENCY:
        urgency = "routine"

    return AffectResult(
        rep_confidence=confidence,
        frustration_signal=frustration,
        stress_signal=stress,
        engagement_level=engagement,
        query_urgency=urgency,
        affect_source="llm",
    )
