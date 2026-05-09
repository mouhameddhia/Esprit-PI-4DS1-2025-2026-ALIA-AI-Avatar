"""Corrective RAG (CRAG) validation and fallback decision logic."""

from __future__ import annotations

from app.schemas.models import RetrievalDiagnostics, RetrievalQuality, ScoredDocument


class CRAGValidator:
    """Assess retrieval confidence and decide if fallback is required."""

    def __init__(
        self,
        min_top_score: float,
        min_mean_score: float,
        min_source_count: int,
        fallback_enabled: bool,
    ) -> None:
        """Initialize CRAG thresholds and fallback controls."""

        self.min_top_score = min_top_score
        self.min_mean_score = min_mean_score
        self.min_source_count = min_source_count
        self.fallback_enabled = fallback_enabled

    def evaluate(self, docs: list[ScoredDocument]) -> RetrievalDiagnostics:
        """Return retrieval diagnostics with confidence and failure reasons."""

        diagnostics = RetrievalDiagnostics()
        if not docs:
            diagnostics.low_confidence = True
            diagnostics.quality = RetrievalQuality.INCORRECT
            diagnostics.reasons.append("no_documents_retrieved")
            diagnostics.confidence_score = 0.0
            return diagnostics

        top_score = docs[0].final_score
        mean_score = sum(doc.final_score for doc in docs) / len(docs)
        unique_sources = {doc.metadata.get("source") for doc in docs if doc.metadata.get("source")}

        confidence = (0.5 * top_score) + (0.4 * mean_score) + (0.1 * min(len(unique_sources) / 3.0, 1.0))
        diagnostics.confidence_score = confidence

        if confidence > 0.7:
            diagnostics.quality = RetrievalQuality.CORRECT
            diagnostics.low_confidence = False
        elif confidence > 0.4:
            diagnostics.quality = RetrievalQuality.AMBIGUOUS
            diagnostics.low_confidence = True
        else:
            diagnostics.quality = RetrievalQuality.INCORRECT
            diagnostics.low_confidence = True

        if top_score < self.min_top_score:
            diagnostics.reasons.append("top_score_below_threshold")
        if mean_score < self.min_mean_score:
            diagnostics.reasons.append("mean_score_below_threshold")
        if len(unique_sources) < self.min_source_count:
            diagnostics.reasons.append("insufficient_source_diversity")

        return diagnostics

    def should_expand(self, diagnostics: RetrievalDiagnostics) -> bool:
        """Return True when query rewriting and merged retrieval should run."""

        return diagnostics.quality == RetrievalQuality.AMBIGUOUS

    def should_fallback(self, diagnostics: RetrievalDiagnostics) -> bool:
        """Return True when fallback retrieval should run."""

        return self.fallback_enabled and diagnostics.quality == RetrievalQuality.INCORRECT
