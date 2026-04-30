"""Sparse BM25 retriever for lexical signal."""

from __future__ import annotations

import pickle
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional retrieval dependency
    BM25Okapi = None

from app.schemas.models import ScoredDocument
from app.utils.text import tokenize_scientific_text


class SparseRetriever:
    """Retrieve chunks using BM25 scoring."""

    def __init__(self, vector_store_dir: str) -> None:
        """Initialize retriever with BM25 artifact file."""

        self.bm25_path = Path(vector_store_dir) / "bm25.pkl"
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    def _load(self) -> bool:
        """Load serialized BM25 corpus and initialize index lazily."""

        if self._bm25 is not None:
            return True
        if BM25Okapi is None or not self.bm25_path.exists():
            return False

        with self.bm25_path.open("rb") as f:
            payload = pickle.load(f)

        self._doc_ids = payload["doc_ids"]
        self._texts = payload["texts"]
        self._metadata = payload["metadata"]
        bm25 = payload.get("bm25")
        if isinstance(bm25, BM25Okapi):
            self._bm25 = bm25
        else:
            tokenized_texts = payload["tokenized_texts"]
            self._bm25 = BM25Okapi(tokenized_texts)
        return True

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return BM25 retrieval results with normalized lexical scores."""

        if not self._load() or self._bm25 is None:
            return []

        query_tokens = tokenize_scientific_text(query)
        scores = self._bm25.get_scores(query_tokens)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        indexed_scores = indexed_scores[:top_k]

        if not indexed_scores:
            return []

        max_score = max(score for _, score in indexed_scores) or 1.0

        results: list[ScoredDocument] = []
        for idx, score in indexed_scores:
            results.append(
                ScoredDocument(
                    doc_id=self._doc_ids[idx],
                    text=self._texts[idx],
                    metadata=self._metadata[idx],
                    bm25_score=float(max(0.0, score / max_score)),
                )
            )
        return results
