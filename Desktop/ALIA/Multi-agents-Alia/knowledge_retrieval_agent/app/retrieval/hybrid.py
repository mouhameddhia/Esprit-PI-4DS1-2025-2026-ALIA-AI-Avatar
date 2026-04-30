"""Hybrid retrieval score fusion (dense + sparse + metadata)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.models import ScoredDocument


class HybridRetriever:
    """Combine dense and sparse signals into a final ranking score."""

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        """Store weighted scoring coefficients."""

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _metadata_quality(metadata: dict) -> float:
        """Estimate metadata quality for scientific trustworthiness."""

        score = 0.0

        source_tier = metadata.get("source_tier")
        if source_tier == 1:
            score += 0.5
        elif source_tier == 2:
            score += 0.35
        elif source_tier == 3:
            score += 0.2
        elif source_tier == 4:
            score += 0.05

        if metadata.get("source"):
            score += 0.15
        if metadata.get("page"):
            score += 0.1
        if metadata.get("domain") == "pharma":
            score += 0.1

        doc_date = metadata.get("doc_date")
        if isinstance(doc_date, str) and doc_date:
            try:
                parsed = datetime.fromisoformat(doc_date).replace(tzinfo=timezone.utc)
                age_days = max(0.0, (datetime.now(timezone.utc) - parsed).days)
                if age_days <= 365:
                    score += 0.15
                elif age_days <= 1825:
                    score += 0.1
                elif age_days <= 3650:
                    score += 0.05
            except ValueError:
                score += 0.0

        return min(score, 1.0)

    def combine(
        self,
        dense_docs: list[ScoredDocument],
        sparse_docs: list[ScoredDocument],
        top_k: int,
    ) -> list[ScoredDocument]:
        """Merge retrieval sets and compute final weighted scores.

        Final Score = alpha * Dense + beta * BM25 + gamma * Metadata
        """

        merged: dict[str, ScoredDocument] = {}

        for doc in dense_docs:
            merged[doc.doc_id] = doc

        for doc in sparse_docs:
            existing = merged.get(doc.doc_id)
            if existing is None:
                merged[doc.doc_id] = doc
            else:
                existing.bm25_score = doc.bm25_score

        for doc in merged.values():
            doc.metadata_score = self._metadata_quality(doc.metadata)
            doc.final_score = (
                self.alpha * doc.dense_score
                + self.beta * doc.bm25_score
                + self.gamma * doc.metadata_score
            )

        ranked = sorted(merged.values(), key=lambda d: d.final_score, reverse=True)
        return ranked[:top_k]
