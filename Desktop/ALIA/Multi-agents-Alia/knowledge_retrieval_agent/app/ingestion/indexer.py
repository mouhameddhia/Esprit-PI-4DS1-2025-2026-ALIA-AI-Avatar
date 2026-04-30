"""Build and persist dense and sparse retrieval indexes."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
import shutil
import time

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional indexing dependency
    BM25Okapi = None

try:
    from langchain_community.vectorstores.faiss import FAISS
except ImportError:  # pragma: no cover - optional indexing dependency
    FAISS = None

from app.ingestion.embedder import get_embedding_model
from app.schemas.models import DocumentChunk
from app.utils.logger import get_logger
from app.utils.text import tokenize_scientific_text

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional memory dependency
    psutil = None


class Indexer:
    """Create FAISS and BM25 artifacts for the retrieval stack."""

    def __init__(
        self,
        vector_store_dir: str,
        embedding_model: str,
        embedding_batch_size: int = 64,
        embedding_min_batch_size: int = 8,
        embedding_max_tokens: int = 256,
        min_chunk_chars: int = 40,
        max_non_printable_ratio: float = 0.2,
    ) -> None:
        """Initialize indexer with storage location and embedding model."""

        self.vector_store_dir = Path(vector_store_dir)
        self.embedding_model = embedding_model
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("knowledge_retrieval_agent.indexer")

        self.embedding_batch_size = max(1, embedding_batch_size)
        self.embedding_min_batch_size = max(1, embedding_min_batch_size)
        self.embedding_max_tokens = max(8, embedding_max_tokens)
        self.min_chunk_chars = max(1, min_chunk_chars)
        self.max_non_printable_ratio = max(0.0, min(1.0, max_non_printable_ratio))
        self.failures_log_path = self.vector_store_dir / "embedding_failures.jsonl"

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format byte counts into a readable unit string."""

        value = float(num_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if value < 1024.0 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{num_bytes} B"

    @staticmethod
    def _estimate_required_space_bytes(chunks: list[DocumentChunk], embedding_dim: int) -> int:
        """Estimate space needed for FAISS index and docstore artifacts."""

        vector_bytes = len(chunks) * embedding_dim * 4
        text_bytes = sum(len(chunk.text.encode("utf-8", errors="ignore")) for chunk in chunks)
        metadata_bytes = max(1, len(chunks)) * 256
        # Add 35% safety margin and 64MB fixed overhead for temporary writes.
        base = vector_bytes + text_bytes + metadata_bytes
        return int(base * 1.35) + (64 * 1024 * 1024)

    def _assert_disk_space(self, required_bytes: int) -> None:
        """Raise a clear error when free disk space is insufficient."""

        free_bytes = shutil.disk_usage(self.vector_store_dir).free
        if free_bytes >= required_bytes:
            return

        raise RuntimeError(
            "Insufficient disk space for FAISS save. "
            f"Required approx: {self._format_bytes(required_bytes)}, "
            f"available: {self._format_bytes(free_bytes)} in '{self.vector_store_dir}'. "
            "Free disk space or point vector_store_dir to a larger drive, then rerun ingest."
        )

    @staticmethod
    def _non_printable_ratio(text: str) -> float:
        """Return the ratio of non-printable characters in a text."""

        if not text:
            return 1.0
        total = len(text)
        non_printable = sum(1 for char in text if not (char.isprintable() or char in "\n\r\t"))
        return non_printable / total

    def _sanitize_text(self, text: str) -> tuple[str | None, str | None]:
        """Normalize and validate text before embedding."""

        cleaned = text.strip()
        if len(cleaned) < self.min_chunk_chars:
            return None, "too_short"

        if self._non_printable_ratio(cleaned) > self.max_non_printable_ratio:
            return None, "non_printable_noise"

        tokens = tokenize_scientific_text(cleaned)
        if len(tokens) < 4:
            return None, "low_information"

        if len(tokens) > self.embedding_max_tokens:
            cleaned = " ".join(tokens[: self.embedding_max_tokens])

        return cleaned, None

    def _prepare_chunks(self, chunks: list[DocumentChunk]) -> tuple[list[DocumentChunk], list[dict[str, str]]]:
        """Filter and sanitize chunks before indexing."""

        prepared: list[DocumentChunk] = []
        rejected: list[dict[str, str]] = []

        for chunk in chunks:
            sanitized_text, reason = self._sanitize_text(chunk.text)
            if sanitized_text is None:
                rejected.append({"doc_id": chunk.doc_id, "reason": reason or "unknown"})
                continue
            prepared.append(
                DocumentChunk(
                    doc_id=chunk.doc_id,
                    text=sanitized_text,
                    metadata={**chunk.metadata},
                )
            )

        return prepared, rejected

    def _append_failures(self, failures: list[dict]) -> None:
        """Append batch/chunk failures to a JSONL log."""

        if not failures:
            return
        with self.failures_log_path.open("a", encoding="utf-8") as handle:
            for item in failures:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    def _resolve_batch_size(self) -> int:
        """Pick a batch size based on available system memory."""

        if psutil is None:
            return self.embedding_batch_size

        available = psutil.virtual_memory().available
        gb = available / (1024**3)
        if gb < 2:
            return self.embedding_min_batch_size
        if gb < 4:
            return max(self.embedding_min_batch_size, self.embedding_batch_size // 2)
        if gb > 16:
            return self.embedding_batch_size * 2
        return self.embedding_batch_size

    def _embed_in_batches(
        self,
        chunks: list[DocumentChunk],
        embeddings,
    ) -> tuple[list[str], list[list[float]], list[dict]]:
        """Embed chunks in resilient batches with progress and failure capture."""

        batch_size = self._resolve_batch_size()
        texts: list[str] = [chunk.text for chunk in chunks]
        metadatas: list[dict] = [{"doc_id": chunk.doc_id, **chunk.metadata} for chunk in chunks]

        embedded_texts: list[str] = []
        vectors: list[list[float]] = []
        embedded_metadata: list[dict] = []
        failed_batches: list[dict] = []

        iterator = range(0, len(texts), batch_size)
        total_batches = math.ceil(len(texts) / batch_size)
        start_time = time.perf_counter()

        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc="Embedding batches",
                unit="batch",
            )

        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            batch_metadata = metadatas[start : start + batch_size]
            try:
                batch_vectors = embeddings.embed_documents(batch_texts)
                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError("embedding_count_mismatch")
                embedded_texts.extend(batch_texts)
                vectors.extend(batch_vectors)
                embedded_metadata.extend(batch_metadata)
            except Exception as exc:
                failed_batches.append(
                    {
                        "stage": "embedding_batch",
                        "start_index": start,
                        "batch_size": len(batch_texts),
                        "error": type(exc).__name__,
                        "doc_ids": [str(meta.get("doc_id", "unknown")) for meta in batch_metadata],
                    }
                )

        elapsed = max(0.001, time.perf_counter() - start_time)
        throughput = len(embedded_texts) / elapsed
        self.logger.info(
            "embedding_metrics",
            extra={
                "extra": {
                    "input_chunks": len(chunks),
                    "embedded_chunks": len(embedded_texts),
                    "failed_batches": len(failed_batches),
                    "batch_size": batch_size,
                    "duration_seconds": round(elapsed, 3),
                    "chunks_per_second": round(throughput, 3),
                }
            },
        )
        self._append_failures(failed_batches)

        return embedded_texts, vectors, embedded_metadata

    def build_dense_index(self, chunks: list[DocumentChunk]) -> None:
        """Persist a FAISS index over chunk embeddings."""

        if not chunks:
            return

        if FAISS is None:
            raise RuntimeError("langchain_community is not installed")

        embeddings = get_embedding_model(self.embedding_model)
        # Probe embedding dimension once, then fail fast if disk is too small.
        embedding_dim = len(embeddings.embed_query("dimension probe"))
        required_bytes = self._estimate_required_space_bytes(chunks, embedding_dim)
        self._assert_disk_space(required_bytes)

        prepared_chunks, rejected = self._prepare_chunks(chunks)
        if rejected:
            self._append_failures([{"stage": "chunk_filter", **item} for item in rejected])
            self.logger.info(
                "chunk_filter_metrics",
                extra={"extra": {"input_chunks": len(chunks), "rejected_chunks": len(rejected)}},
            )

        if not prepared_chunks:
            raise RuntimeError("No valid chunks remained after quality filtering")

        texts, vectors, metadatas = self._embed_in_batches(prepared_chunks, embeddings)
        if not texts:
            raise RuntimeError("All embedding batches failed. Check embedding_failures.jsonl for details")

        text_embeddings = list(zip(texts, vectors))
        db = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embeddings, metadatas=metadatas)
        db.save_local(str(self.vector_store_dir / "faiss"))

    def build_sparse_index(self, chunks: list[DocumentChunk]) -> None:
        """Persist tokenized corpus for BM25 retrieval."""

        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 is not installed")

        tokenized_texts = [tokenize_scientific_text(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_texts)
        payload = {
            "doc_ids": [chunk.doc_id for chunk in chunks],
            "texts": [chunk.text for chunk in chunks],
            "metadata": [chunk.metadata for chunk in chunks],
            "tokenized_texts": tokenized_texts,
            "bm25": bm25,
        }
        with (self.vector_store_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(payload, f)

    def build_all(self, chunks: list[DocumentChunk]) -> None:
        """Build both dense and sparse indexes in one call."""

        self.build_dense_index(chunks)
        self.build_sparse_index(chunks)
