"""Dense retriever backed by FAISS embeddings index."""

from __future__ import annotations

from pathlib import Path

try:
    from langchain_community.vectorstores.faiss import FAISS
except ImportError:  # pragma: no cover - optional retrieval dependency
    FAISS = None

from app.ingestion.embedder import get_embedding_model
from app.schemas.models import ScoredDocument


class DenseRetriever:
    """Retrieve semantically similar chunks from FAISS."""

    def __init__(self, vector_store_dir: str, embedding_model: str) -> None:
        """Initialize retriever with local FAISS artifact path."""

        self.faiss_path = Path(vector_store_dir) / "faiss"
        self.embedding_model = embedding_model
        self._db: FAISS | None = None

    def _load(self) -> FAISS | None:
        """Load FAISS index lazily."""

        if self._db is not None:
            return self._db
        if FAISS is None or not self.faiss_path.exists():
            return None

        embeddings = get_embedding_model(self.embedding_model)
        self._db = FAISS.load_local(
            str(self.faiss_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return self._db

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return dense retrieval results with normalized similarity scores."""

        db = self._load()
        if db is None:
            return []

        docs = db.similarity_search_with_score(query, k=top_k)
        if not docs:
            return []

        # FAISS returns distance-like values; lower is better.
        distances = [score for _, score in docs]
        min_d, max_d = min(distances), max(distances)

        results: list[ScoredDocument] = []
        for doc, distance in docs:
            if max_d > min_d:
                dense_score = 1.0 - ((distance - min_d) / (max_d - min_d))
            else:
                dense_score = 1.0

            doc_id = str(doc.metadata.get("doc_id", doc.page_content[:30]))
            results.append(
                ScoredDocument(
                    doc_id=doc_id,
                    text=doc.page_content,
                    metadata=dict(doc.metadata),
                    dense_score=float(max(0.0, min(1.0, dense_score))),
                )
            )
        return results
