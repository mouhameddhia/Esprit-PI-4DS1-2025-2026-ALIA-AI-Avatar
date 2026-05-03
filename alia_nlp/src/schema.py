"""Typed output contract for the NLP pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class NLPResult:
    # Core classification
    intent: str
    needs_clarification: bool
    confidence: float

    # Lists
    secondary_tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    # Maps
    entity_map: Dict[str, List[str]] = field(default_factory=dict)
    explainability: Dict[str, Any] = field(default_factory=dict)

    # Affect — communicative state analysis (L7)
    affect: Dict[str, Any] = field(default_factory=dict)

    # Scalars
    rewritten_query: str = ""
    language: str = "unknown"
    taxonomy_version: str = "v1"

    # Audit — which backend produced the intent
    intent_source: str = "unknown"  # "rules" | "llm" | "fallback"

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict for existing backend consumers."""
        return asdict(self)

    @classmethod
    def make_fallback(cls, user_text: str = "", language: str = "unknown") -> "NLPResult":
        return cls(
            intent="other",
            needs_clarification=True,
            confidence=0.4,
            rewritten_query=user_text,
            language=language,
            intent_source="fallback",
        )
