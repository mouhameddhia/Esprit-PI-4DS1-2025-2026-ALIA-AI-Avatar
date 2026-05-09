"""Tests for three-state CRAG quality classification."""

from app.correction.crag import CRAGValidator
from app.schemas.models import RetrievalQuality, ScoredDocument


def test_crag_classifies_correct_ambiguous_and_incorrect() -> None:
    """CRAG should separate correct, ambiguous, and incorrect retrieval states."""

    validator = CRAGValidator(
        min_top_score=0.35,
        min_mean_score=0.25,
        min_source_count=1,
        fallback_enabled=True,
    )

    correct_docs = [ScoredDocument(doc_id="a", text="x", metadata={"source": "s1"}, final_score=0.9)]
    ambiguous_docs = [ScoredDocument(doc_id="a", text="x", metadata={"source": "s1"}, final_score=0.55)]
    incorrect_docs = [ScoredDocument(doc_id="a", text="x", metadata={"source": "s1"}, final_score=0.2)]

    assert validator.evaluate(correct_docs).quality == RetrievalQuality.CORRECT
    assert validator.evaluate(ambiguous_docs).quality == RetrievalQuality.AMBIGUOUS
    assert validator.evaluate(incorrect_docs).quality == RetrievalQuality.INCORRECT