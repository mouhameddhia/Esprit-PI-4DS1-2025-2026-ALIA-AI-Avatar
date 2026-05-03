"""Shadow monitoring — compare LLM intent vs. rule-based fallback on live traffic."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MAX_DIVERGENCE = 0.08  # 8 % tolerance


def compute_divergence(events: List[Dict[str, Any]]) -> float:
    """Fraction of events where LLM intent != fallback intent."""
    if not events:
        return 0.0
    diverged = sum(
        1 for e in events
        if e.get("llm_intent") and e.get("fallback_intent")
        and e["llm_intent"] != e["fallback_intent"]
    )
    return diverged / len(events)


def snapshot(events: List[Dict[str, Any]], output_dir: str = "alia_nlp/evaluation/results") -> Dict:
    divergence = compute_divergence(events)
    result = {
        "sampled_at": datetime.utcnow().isoformat(),
        "event_count": len(events),
        "divergence": round(divergence, 4),
        "status": "ALERT" if divergence > _MAX_DIVERGENCE else "OK",
    }
    out_path = Path(output_dir) / f"shadow_{datetime.utcnow().strftime('%Y%m%d')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Shadow snapshot: divergence=%.2f%% status=%s", divergence * 100, result["status"])
    return result
