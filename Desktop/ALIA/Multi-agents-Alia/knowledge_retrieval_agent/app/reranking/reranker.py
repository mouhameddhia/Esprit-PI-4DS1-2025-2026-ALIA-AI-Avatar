"""Cross-encoder reranker for top retrieved chunks."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.schemas.models import ScoredDocument


class CrossEncoderReranker:
    """Rerank candidate chunks using pairwise query-document scoring."""

    def __init__(self, model_name: str) -> None:
        """Initialize with a cross-encoder model name."""

        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        """Lazily load the reranker model."""

        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: list[ScoredDocument], top_k: int) -> list[ScoredDocument]:
        """Return top-k reranked documents."""

        if not docs:
            return []

        model = self._load()
        pairs = [[query, doc.text] for doc in docs]
        scores = model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc.rerank_score = float(score)

        ranked = sorted(docs, key=lambda item: item.rerank_score, reverse=True)
        return ranked[:top_k]
