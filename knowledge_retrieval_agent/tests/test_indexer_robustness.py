"""Tests for robust embedding safeguards in indexer."""

from __future__ import annotations

import json

from app.ingestion.indexer import Indexer
from app.schemas.models import DocumentChunk


class FakeEmbeddings:
    """Simple embedding stub that fails on trigger text."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any("FAIL" in text for text in texts):
            raise RuntimeError("simulated_batch_failure")
        return [[float(len(text))] for text in texts]


def test_sanitize_text_filters_noise_and_truncates_tokens(tmp_path):
    """Short/noisy chunks are rejected and long chunks are token-capped."""

    indexer = Indexer(
        vector_store_dir=str(tmp_path),
        embedding_model="stub",
        embedding_max_tokens=8,
        min_chunk_chars=10,
        max_non_printable_ratio=0.1,
    )

    cleaned, reason = indexer._sanitize_text("tiny")
    assert cleaned is None
    assert reason == "too_short"

    noisy = "Valid text \x00\x01\x02\x03\x04 with enough length"
    cleaned, reason = indexer._sanitize_text(noisy)
    assert cleaned is None
    assert reason == "non_printable_noise"

    long_text = "token1 token2 token3 token4 token5 token6 token7 token8 token9 token10"
    cleaned, reason = indexer._sanitize_text(long_text)
    assert reason is None
    assert cleaned == "token1 token2 token3 token4 token5 token6 token7 token8"


def test_embed_batches_continue_on_failure_and_log(tmp_path):
    """Failed embedding batches are logged while successful batches continue."""

    indexer = Indexer(
        vector_store_dir=str(tmp_path),
        embedding_model="stub",
        embedding_batch_size=1,
        min_chunk_chars=5,
    )

    chunks = [
        DocumentChunk(doc_id="a", text="This is fine", metadata={}),
        DocumentChunk(doc_id="b", text="This chunk will FAIL now", metadata={}),
        DocumentChunk(doc_id="c", text="This is also fine", metadata={}),
    ]

    texts, vectors, metadatas = indexer._embed_in_batches(chunks, FakeEmbeddings())

    assert len(texts) == 2
    assert len(vectors) == 2
    assert len(metadatas) == 2
    assert {m["doc_id"] for m in metadatas} == {"a", "c"}

    log_path = tmp_path / "embedding_failures.jsonl"
    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(record.get("stage") == "embedding_batch" for record in records)


def test_resolve_batch_size_uses_available_memory(tmp_path, monkeypatch):
    """Adaptive batch size scales down on constrained memory."""

    from app.ingestion import indexer as indexer_module

    class _Mem:
        available = int(1.5 * (1024**3))

    class _Psutil:
        @staticmethod
        def virtual_memory():
            return _Mem()

    monkeypatch.setattr(indexer_module, "psutil", _Psutil)

    indexer = Indexer(
        vector_store_dir=str(tmp_path),
        embedding_model="stub",
        embedding_batch_size=64,
        embedding_min_batch_size=8,
    )
    assert indexer._resolve_batch_size() == 8
