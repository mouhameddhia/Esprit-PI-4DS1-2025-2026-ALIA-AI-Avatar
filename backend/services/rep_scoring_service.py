"""Service layer for representative response scoring."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..utils.rep_scoring import RepResponseScorer


class RepScoringService:
    """Thin service wrapper to keep route handlers decoupled from model internals."""

    def __init__(self) -> None:
        self._scorer = RepResponseScorer()

    def score_response(self, content: str) -> dict[str, Any]:
        result = self._scorer.score_text(content)
        return result.to_dict()

    def is_ready(self) -> bool:
        return self._scorer.is_ready()


@lru_cache(maxsize=1)
def get_rep_scoring_service() -> RepScoringService:
    return RepScoringService()
