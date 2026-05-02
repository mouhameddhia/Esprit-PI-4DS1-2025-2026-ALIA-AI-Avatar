"""Tests for raw extraction inspection helpers."""

from pathlib import Path

from app.inspection.raw_text import load_raw_documents, save_raw_documents, search_raw_documents
from app.schemas.models import DocumentChunk


def test_raw_inspection_save_load_and_search(tmp_path) -> None:
    """Raw inspection cache should persist documents and support lexical search."""

    docs = [
        DocumentChunk(doc_id="a", text="BACTOL composition includes chlorhexidine", metadata={"source": "a.txt", "page": 1}),
        DocumentChunk(doc_id="b", text="Unrelated cleaning details", metadata={"source": "b.txt", "page": 1}),
    ]

    cache_path = tmp_path / "raw_extracted.jsonl"
    save_raw_documents(cache_path, docs)
    loaded = load_raw_documents(cache_path)

    assert len(loaded) == 2

    matches = search_raw_documents(loaded, query="bactol composition", top_k=1)
    assert len(matches) == 1
    assert matches[0].doc_id == "a"


def test_raw_inspection_search_matches_source_and_doc_id(tmp_path) -> None:
    """Search should also match metadata/source identifiers for debugging extraction."""

    docs = [
        DocumentChunk(doc_id="Gamme BACTOL_p1", text="PK binary header", metadata={"source": "Gamme BACTOL.pptx", "page": 1}),
        DocumentChunk(doc_id="other_doc", text="unrelated", metadata={"source": "other.txt", "page": 1}),
    ]
    matches = search_raw_documents(docs, query="bactol", top_k=2)
    assert len(matches) == 1
    assert matches[0].doc_id == "Gamme BACTOL_p1"
