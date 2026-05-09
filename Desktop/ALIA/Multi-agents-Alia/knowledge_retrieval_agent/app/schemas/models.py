"""Shared Pydantic models used across the RAG pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Canonical representation of a chunked source document."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredDocument(BaseModel):
    """Retrieved document with individual and combined retrieval scores."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    dense_score: float = 0.0
    bm25_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    rerank_score: float = 0.0


class RetrievalQuality(str, Enum):
    """Three-state retrieval quality used by CRAG."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class RetrievalDiagnostics(BaseModel):
    """Debug and quality signals for observability and CRAG decisions."""

    low_confidence: bool = False
    quality: RetrievalQuality = RetrievalQuality.INCORRECT
    reasons: list[str] = Field(default_factory=list)
    fallback_triggered: bool = False
    query_variants: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0


class EvidenceSnippet(BaseModel):
    """Compact evidence block to expose the exact supporting chunk text."""

    doc_id: str
    text: str
    source: str = "unknown_source"
    page: int | str = "n/a"


class PipelineResponse(BaseModel):
    """Final pipeline output containing answer, citations, and diagnostics."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    supporting_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    retrieved_docs: list[ScoredDocument] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    answer_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = Field(default="")
    conflict_notes: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class GeneratedAnswer(BaseModel):
    """Structured LLM output for grounded answer generation."""

    answer: str
    citation_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = Field(default="")
    conflict_notes: list[str] = Field(default_factory=list)
