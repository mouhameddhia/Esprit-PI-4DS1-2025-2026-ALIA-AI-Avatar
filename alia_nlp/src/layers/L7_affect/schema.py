from dataclasses import asdict, dataclass


@dataclass
class AffectResult:
    rep_confidence: str = "medium"     # "low" | "medium" | "high"
    frustration_signal: bool = False
    engagement_level: str = "active"   # "passive" | "active" | "highly_engaged"
    query_urgency: str = "routine"     # "routine" | "elevated" | "urgent"
    affect_source: str = "rules"       # "rules" | "llm" | "model"

    def to_dict(self) -> dict:
        return asdict(self)
