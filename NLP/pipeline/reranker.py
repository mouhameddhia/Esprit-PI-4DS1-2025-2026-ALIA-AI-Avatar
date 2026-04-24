"""Lightweight hybrid reranker for retrieval candidates."""

from __future__ import annotations

import re
from typing import Any, Dict, List


DEFAULT_WEIGHTS: Dict[str, float] = {
    "base_score": 0.65,
    "overlap": 0.20,
    "jaccard": 0.15,
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return (inter / union) if union else 0.0


def _overlap_recall(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    return len(qset & dset) / max(1, len(qset))


def _candidate_text(candidate: Dict[str, Any]) -> str:
    # Prefer explicit text when present (used by evaluation harness).
    text = candidate.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    fields = [
        metadata.get("name", ""),
        metadata.get("source_name", ""),
        metadata.get("title", ""),
        metadata.get("section_title", ""),
        metadata.get("category", ""),
        metadata.get("chunk_text", ""),
    ]
    return " ".join(part for part in fields if isinstance(part, str) and part.strip()).strip()


def _normalize_weights(weights: Dict[str, Any] | None) -> Dict[str, float]:
    payload = weights if isinstance(weights, dict) else {}
    base = float(payload.get("base_score", DEFAULT_WEIGHTS["base_score"]))
    overlap = float(payload.get("overlap", DEFAULT_WEIGHTS["overlap"]))
    jaccard = float(payload.get("jaccard", DEFAULT_WEIGHTS["jaccard"]))

    total = base + overlap + jaccard
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)

    return {
        "base_score": base / total,
        "overlap": overlap / total,
        "jaccard": jaccard / total,
    }


def score_candidate(query: str, candidate: Dict[str, Any], weights: Dict[str, Any] | None = None) -> float:
    """Compute a blended rerank score using vector score + lexical relevance."""
    base_score = float(candidate.get("score", 0.0) or 0.0)
    text = _candidate_text(candidate)

    query_tokens = _tokenize(query)
    text_tokens = _tokenize(text)

    jaccard = _jaccard(query_tokens, text_tokens)
    overlap = _overlap_recall(query_tokens, text_tokens)

    normalized = _normalize_weights(weights)
    # Preserve vector semantics but boost lexical grounding.
    combined = (
        (normalized["base_score"] * base_score)
        + (normalized["overlap"] * overlap)
        + (normalized["jaccard"] * jaccard)
    )

    return round(combined, 6)


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int | None = None,
    *,
    model: Dict[str, Any] | None = None,
    weights: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return candidates sorted by hybrid rerank score."""
    scored: List[Dict[str, Any]] = []

    effective_weights = weights
    if effective_weights is None and isinstance(model, dict):
        maybe = model.get("weights")
        if isinstance(maybe, dict):
            effective_weights = maybe

    for candidate in candidates:
        enriched = dict(candidate)
        enriched["rerank_score"] = score_candidate(query=query, candidate=candidate, weights=effective_weights)
        scored.append(enriched)

    scored.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)

    if top_k is not None and top_k > 0:
        return scored[:top_k]
    return scored
