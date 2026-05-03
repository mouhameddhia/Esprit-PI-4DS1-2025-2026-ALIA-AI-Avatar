"""
Representative response scoring service.

This is a stub implementation that returns neutral placeholder scores.
Replace the body of RepScoringService.score_response() with real model
inference once a scoring artifact is available (set REP_SCORER_ARTIFACT_DIR
in .env to point to it).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_instance: "RepScoringService | None" = None


class RepScoringService:
    """Scores a medical-rep response on clarity, persuasion, and confidence."""

    def score_response(self, content: str) -> Dict[str, Any]:
        """
        Returns a dict matching RepScoreResponse fields.
        Real implementation would run content through a trained classifier.
        """
        # Stub: neutral mid-range scores so the endpoint is functional
        return {
            "clarity_score": 0.5,
            "persuasion_score": 0.5,
            "confidence_score": 0.5,
            "model_source": "stub",
        }


def get_rep_scoring_service() -> RepScoringService:
    """Singleton factory used by FastAPI dependency injection."""
    global _instance
    if _instance is None:
        _instance = RepScoringService()
    return _instance
