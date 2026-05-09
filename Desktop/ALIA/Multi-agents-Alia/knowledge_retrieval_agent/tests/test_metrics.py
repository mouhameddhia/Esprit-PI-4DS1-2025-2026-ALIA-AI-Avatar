"""Tests for evaluation metrics behavior."""

from app.evaluation import metrics


def test_normalize_scores_returns_unit_range() -> None:
    """Score normalization should preserve ordering on a [0, 1] scale."""

    assert metrics.normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_graded_ndcg_uses_relevance_grades() -> None:
    """Graded nDCG should reward highly relevant documents more than marginal ones."""

    relevance_scores = {"doc_a": 3.0, "doc_b": 1.0, "doc_c": 0.0}
    best_order = metrics.ndcg_at_k(["doc_a", "doc_b", "doc_c"], set(), 3, relevance_scores)
    worse_order = metrics.ndcg_at_k(["doc_b", "doc_a", "doc_c"], set(), 3, relevance_scores)

    assert best_order > worse_order


def test_evaluate_batch_aggregates_metrics(monkeypatch) -> None:
    """Batch evaluation should aggregate per-query metric outputs."""

    monkeypatch.setattr(metrics, "faithfulness_nli", lambda answer, contexts: 0.8)
    monkeypatch.setattr(metrics, "hallucination_rate_nli", lambda answer, contexts: 0.2)
    monkeypatch.setattr(metrics, "answer_relevance", lambda query, answer, model_name=metrics.RELEVANCE_MODEL_NAME: 0.9)
    monkeypatch.setattr(metrics, "context_relevance", lambda query, contexts, model_name=metrics.RELEVANCE_MODEL_NAME: 0.7)

    sample = metrics.EvaluationSample(
        query="What is the half-life?",
        retrieved=["doc_a", "doc_b"],
        relevant={"doc_a"},
        answer="The half-life is 4 hours.",
        contexts=["The half-life is 4 hours."],
        dense_scores=[0.2, 0.8],
        bm25_scores=[1.0, 3.0],
        latency_ms=12.5,
    )

    result = metrics.evaluate_batch([sample], k=1)

    assert len(result.per_sample) == 1
    assert result.summary["faithfulness"].mean == 0.8
    assert result.summary["hallucination_rate"].mean == 0.2
    assert result.summary["answer_relevance"].mean == 0.9
    assert result.summary["context_relevance"].mean == 0.7
    assert result.summary["retrieval_confidence"].mean == 0.5