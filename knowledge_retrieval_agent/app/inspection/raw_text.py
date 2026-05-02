"""Helpers for persisting and searching raw extracted text."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.models import DocumentChunk
from app.utils.text import tokenize_scientific_text


def save_raw_documents(path: Path, documents: list[DocumentChunk]) -> None:
    """Persist raw extracted documents as JSONL for inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            payload = {
                "doc_id": document.doc_id,
                "text": document.text,
                "metadata": document.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_raw_documents(path: Path) -> list[DocumentChunk]:
    """Load raw extracted documents from a JSONL cache file."""

    if not path.exists():
        return []

    documents: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            documents.append(
                DocumentChunk(
                    doc_id=str(payload.get("doc_id", "unknown_doc")),
                    text=str(payload.get("text", "")),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
    return documents


def search_raw_documents(documents: list[DocumentChunk], query: str, top_k: int = 5) -> list[DocumentChunk]:
    """Return top-k raw documents ranked by lexical overlap with the query."""

    query_tokens = tokenize_scientific_text(query)
    if not query_tokens:
        return documents[:top_k]

    scored: list[tuple[int, int, DocumentChunk]] = []
    for idx, document in enumerate(documents):
        source = str(document.metadata.get("source", ""))
        searchable = f"{document.text}\n{document.doc_id}\n{source}"
        doc_tokens = tokenize_scientific_text(searchable)
        if not doc_tokens:
            continue
        overlap = sum(1 for token in query_tokens if token in doc_tokens)
        if overlap > 0:
            scored.append((overlap, -idx, document))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:top_k]]
