"""Confidence aggregation across pipeline layers."""


def aggregate(
    intent_confidence: float,
    has_entities: bool,
    intent_source: str,
) -> float:
    """
    Blend intent confidence with signal quality to produce a final score.
    Rules: small bonus for matched entities; small penalty for fallback source.
    """
    score = intent_confidence
    if has_entities:
        score = min(1.0, score + 0.05)
    if intent_source == "fallback":
        score = max(0.0, score - 0.10)
    return round(score, 4)
