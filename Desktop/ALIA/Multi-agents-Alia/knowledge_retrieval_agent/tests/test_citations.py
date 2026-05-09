"""Tests for citation extraction in the answer generator."""

from app.generation.generator import AnswerGenerator
from app.schemas.models import ScoredDocument


def test_extract_used_citations_matches_answer_markers() -> None:
    """Only documents referenced by inline markers should be returned as citations."""

    docs = [
        ScoredDocument(doc_id="1", text="doc 1", metadata={"source": "a.pdf", "page": 1}),
        ScoredDocument(doc_id="2", text="doc 2", metadata={"source": "b.pdf", "page": 2}),
        ScoredDocument(doc_id="3", text="doc 3", metadata={"source": "c.pdf", "page": 3}),
    ]

    citations = AnswerGenerator._extract_used_citations("Result from [1] and [3].", docs)

    assert citations == ["a.pdf#page=1", "c.pdf#page=3"]