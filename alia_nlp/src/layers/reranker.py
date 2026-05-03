"""Passthrough reranker — placeholder until a scored reranker is implemented."""

from typing import Any, List


def rerank_candidates(candidates: List[Any], query: str) -> List[Any]:
    """Return candidates unchanged. Replace with a scored reranker when ready."""
    return candidates
