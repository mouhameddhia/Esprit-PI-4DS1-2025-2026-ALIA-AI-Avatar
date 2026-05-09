"""Adapter around existing knowledge_retrieval_agent RAG pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.models import PipelineResponse

class RAGAdapter:
    """Thin compatibility wrapper that leaves the original RAG code untouched."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._pipeline = None
        self._init_error: str | None = None

    def _ensure_pipeline(self):
        if self._init_error:
            return None
        if self._pipeline is not None:
            return self._pipeline

        rag_root = self.workspace_root / "knowledge_retrieval_agent"
        if str(rag_root) not in sys.path:
            sys.path.insert(0, str(rag_root))

        try:
            from app.config import get_settings
            from app.pipelines.rag_pipeline import RAGPipeline

            self._pipeline = RAGPipeline(get_settings())
            return self._pipeline
        except Exception as exc:
            # Keep app responsive if optional RAG dependencies are unavailable.
            self._init_error = str(exc)
            return None

    def query(self, question: str, response_language: str = "en") -> PipelineResponse | None:
        """Execute fallback retrieval and return full RAG pipeline output."""

        pipeline = self._ensure_pipeline()
        if pipeline is None:
            return None
        response = pipeline.run(question, response_language=response_language)

        if not response.retrieved_docs:
            return None

        return response
