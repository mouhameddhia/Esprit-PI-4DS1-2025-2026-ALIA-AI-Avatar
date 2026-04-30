"""Evaluation metrics for retrieval and generation quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from collections.abc import Iterable, Sequence
import math
from statistics import mean, pstdev

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
RELEVANCE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class EvaluationSample:
    """Single query evaluation payload."""

    query: str
    retrieved: list[str]
    relevant: set[str]
    answer: str
    contexts: list[str] = field(default_factory=list)
    relevance_scores: dict[str, float] = field(default_factory=dict)
    dense_scores: list[float] = field(default_factory=list)
    bm25_scores: list[float] = field(default_factory=list)
    latency_ms: float | None = None


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate metric statistics across a batch."""

    mean: float
    std: float
    count: int


@dataclass
class EvaluationResult:
    """Batch-level evaluation output with per-sample and aggregate metrics."""

    per_sample: list[dict[str, float]]
    summary: dict[str, MetricSummary]


def _as_list(values: Iterable[float]) -> list[float]:
    """Materialize an iterable of numeric values into a list of floats."""

    return [float(value) for value in values]


def _sigmoid(value: float) -> float:
    """Convert an unconstrained score into a 0-1 range."""

    return 1.0 / (1.0 + math.exp(-value))


def _safe_mean(values: Sequence[float]) -> float:
    """Return a mean that defaults to 0 for empty input."""

    return float(mean(values)) if values else 0.0


def normalize_scores(scores: Iterable[float]) -> list[float]:
    """Min-max normalize a list of scores to the [0, 1] interval.

    When all values are identical, return a neutral score of 0.5 for each item.
    """

    values = _as_list(scores)
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.5 for _ in values]

    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Precision@K."""

    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@K."""

    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """Compute graded nDCG@K.

    If relevance_scores is omitted, fall back to binary relevance.
    """

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    def gain(item: str) -> float:
        if relevance_scores is not None:
            return max(0.0, float(relevance_scores.get(item, 0.0)))
        return 1.0 if item in relevant else 0.0

    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        rel = gain(item)
        dcg += (2.0**rel - 1.0) / math.log2(rank + 1)

    ideal_relevances = sorted(
        (gain(item) for item in retrieved if gain(item) > 0.0),
        reverse=True,
    )[:k]
    if not ideal_relevances:
        return 0.0

    idcg = 0.0
    for rank, rel in enumerate(ideal_relevances, start=1):
        idcg += (2.0**rel - 1.0) / math.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Compute reciprocal rank for a single query."""

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Backward-compatible alias for reciprocal rank on a single query."""

    return reciprocal_rank(retrieved, relevant)


def mean_reciprocal_rank(query_rankings: Sequence[list[str]], relevant_sets: Sequence[set[str]]) -> float:
    """Compute mean reciprocal rank across a batch of queries."""

    if not query_rankings or not relevant_sets:
        return 0.0

    scores = [reciprocal_rank(ranking, relevant) for ranking, relevant in zip(query_rankings, relevant_sets)]
    return _safe_mean(scores)


@lru_cache(maxsize=1)
def _get_nli_model() -> object:
    """Load the NLI cross-encoder once per process."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(NLI_MODEL_NAME)


@lru_cache(maxsize=4)
def _get_relevance_model(model_name: str = RELEVANCE_MODEL_NAME) -> object:
    """Load a cross-encoder for semantic relevance scoring once per model name."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _row_to_probability(row: Sequence[float]) -> float:
    """Convert a model output row into a probability-like score."""

    values = [float(value) for value in row]
    if not values:
        return 0.0
    if len(values) == 1:
        return _sigmoid(values[0])

    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    denominator = sum(exp_values)
    if denominator <= 0:
        return 0.0

    entailment_index = 2 if len(values) > 2 else len(values) - 1
    return exp_values[entailment_index] / denominator


def _extract_entailment_scores(raw_scores: object) -> list[float]:
    """Extract entailment probabilities from a CrossEncoder output."""

    if raw_scores is None:
        return []

    if isinstance(raw_scores, (int, float)):
        return [_sigmoid(float(raw_scores))]

    rows = list(raw_scores) if isinstance(raw_scores, Iterable) else []
    if not rows:
        return []

    first_row = rows[0]
    if isinstance(first_row, Iterable) and not isinstance(first_row, (str, bytes)):
        return [_row_to_probability(row) for row in rows]

    return [_sigmoid(float(score)) for score in rows]


def faithfulness_nli(answer: str, contexts: list[str]) -> float:
    """Score faithfulness via NLI entailment.

    Each context is treated as a premise for the answer hypothesis.
    The final score is the maximum entailment probability across contexts.
    """

    if not answer.strip() or not contexts:
        return 0.0

    pairs = [(context, answer) for context in contexts if context.strip()]
    if not pairs:
        return 0.0

    scores = _get_nli_model().predict(pairs)
    entailment_scores = _extract_entailment_scores(scores)
    return max(entailment_scores) if entailment_scores else 0.0


def hallucination_rate_nli(answer: str, contexts: list[str]) -> float:
    """Estimate hallucination as the complement of NLI faithfulness."""

    return 1.0 - faithfulness_nli(answer, contexts)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Backward-compatible alias for NLI-based faithfulness."""

    return faithfulness_nli(answer, contexts)


def hallucination_rate(answer: str, contexts: list[str]) -> float:
    """Backward-compatible alias for NLI-based hallucination rate."""

    return hallucination_rate_nli(answer, contexts)


def answer_relevance(query: str, answer: str, model_name: str = RELEVANCE_MODEL_NAME) -> float:
    """Measure whether the answer addresses the query using a cross-encoder.

    Returns a normalized score in the [0, 1] range.
    """

    if not query.strip() or not answer.strip():
        return 0.0

    score = _get_relevance_model(model_name).predict([(query, answer)])
    values = _as_list(score if isinstance(score, Iterable) and not isinstance(score, (str, bytes)) else [score])
    return _sigmoid(values[0]) if values else 0.0


def context_relevance(query: str, contexts: list[str], model_name: str = RELEVANCE_MODEL_NAME) -> float:
    """Measure whether the retrieved chunks are relevant to the query."""

    if not query.strip() or not contexts:
        return 0.0

    pairs = [(query, context) for context in contexts if context.strip()]
    if not pairs:
        return 0.0

    raw_scores = _get_relevance_model(model_name).predict(pairs)
    values = _as_list(raw_scores if isinstance(raw_scores, Iterable) and not isinstance(raw_scores, (str, bytes)) else [raw_scores])
    probabilities = [_sigmoid(value) for value in values]
    return max(probabilities) if probabilities else 0.0


def retrieval_confidence(*score_groups: Iterable[float]) -> float:
    """Aggregate retrieval confidence after normalizing each score group.

    This prevents dense and sparse scores from being averaged on incompatible scales.
    """

    normalized_values: list[float] = []
    for score_group in score_groups:
        normalized_values.extend(normalize_scores(score_group))
    return _safe_mean(normalized_values)


def latency_ms(start_ms: float, end_ms: float) -> float:
    """Return latency from two millisecond timestamps."""

    return max(0.0, end_ms - start_ms)


def fallback_trigger_rate(total_queries: int, fallback_count: int) -> float:
    """Compute fallback trigger ratio."""

    if total_queries <= 0:
        return 0.0
    return fallback_count / total_queries


def _aggregate(values: list[float]) -> MetricSummary:
    """Summarize a list of metric values."""

    if not values:
        return MetricSummary(mean=0.0, std=0.0, count=0)
    if len(values) == 1:
        return MetricSummary(mean=values[0], std=0.0, count=1)
    return MetricSummary(mean=_safe_mean(values), std=float(pstdev(values)), count=len(values))


def evaluate_batch(
    samples: Sequence[EvaluationSample],
    k: int = 10,
    relevance_model_name: str = RELEVANCE_MODEL_NAME,
) -> EvaluationResult:
    """Evaluate a batch of queries and return per-sample and aggregate metrics."""

    per_sample: list[dict[str, float]] = []
    metric_buckets: dict[str, list[float]] = {
        "precision_at_k": [],
        "recall_at_k": [],
        "ndcg_at_k": [],
        "reciprocal_rank": [],
        "answer_relevance": [],
        "context_relevance": [],
        "faithfulness": [],
        "hallucination_rate": [],
        "retrieval_confidence": [],
        "latency_ms": [],
    }

    for sample in samples:
        precision = precision_at_k(sample.retrieved, sample.relevant, k)
        recall = recall_at_k(sample.retrieved, sample.relevant, k)
        ndcg = ndcg_at_k(sample.retrieved, sample.relevant, k, sample.relevance_scores or None)
        rr = reciprocal_rank(sample.retrieved, sample.relevant)
        answer_rel = answer_relevance(sample.query, sample.answer, model_name=relevance_model_name)
        context_rel = context_relevance(sample.query, sample.contexts, model_name=relevance_model_name)
        faith = faithfulness_nli(sample.answer, sample.contexts)
        hallucination = hallucination_rate_nli(sample.answer, sample.contexts)
        retrieval_conf = retrieval_confidence(sample.dense_scores, sample.bm25_scores)
        latency = float(sample.latency_ms) if sample.latency_ms is not None else 0.0

        sample_metrics = {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "ndcg_at_k": ndcg,
            "reciprocal_rank": rr,
            "answer_relevance": answer_rel,
            "context_relevance": context_rel,
            "faithfulness": faith,
            "hallucination_rate": hallucination,
            "retrieval_confidence": retrieval_conf,
            "latency_ms": latency,
        }
        per_sample.append(sample_metrics)

        for metric_name, metric_value in sample_metrics.items():
            metric_buckets[metric_name].append(metric_value)

    summary = {metric_name: _aggregate(values) for metric_name, values in metric_buckets.items()}
    return EvaluationResult(per_sample=per_sample, summary=summary)
