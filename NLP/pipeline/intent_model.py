"""Lightweight trainable intent model (multinomial Naive Bayes)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\u00c0-\u00ff']+", (text or "").lower())


def _expand_ngrams(tokens: List[str], ngram_min: int, ngram_max: int) -> List[str]:
    if not tokens:
        return []

    ngram_min = max(1, int(ngram_min))
    ngram_max = max(ngram_min, int(ngram_max))

    features: List[str] = []
    size = len(tokens)
    for n in range(ngram_min, ngram_max + 1):
        if n == 1:
            features.extend(tokens)
            continue
        for i in range(0, size - n + 1):
            features.append("__ng_" + "_".join(tokens[i : i + n]))
    return features


def train_intent_model(
    rows: Iterable[Dict[str, Any]],
    *,
    ngram_min: int = 1,
    ngram_max: int = 2,
    class_prior_mode: str = "uniform",
    class_balance: bool = True,
) -> Dict[str, Any]:
    rows = list(rows)

    class_counts: Counter[str] = Counter()
    token_counts: Dict[str, Dict[str, float]] = defaultdict(dict)
    token_totals: Dict[str, float] = defaultdict(float)
    vocab: set[str] = set()

    seen = 0
    used = 0

    for row in rows:
        seen += 1
        text = row.get("text")
        label = row.get("expected_intent")
        mode = row.get("mode", "unknown")
        if not isinstance(text, str) or not text.strip() or not isinstance(label, str) or not label.strip():
            continue

        tokens = _expand_ngrams(_tokenize(text), ngram_min=ngram_min, ngram_max=ngram_max)
        if isinstance(mode, str) and mode.strip():
            tokens.append(f"__mode_{mode.strip().lower()}")

        label = label.strip()
        class_counts[label] += 1
        used += 1

        for tok in tokens:
            vocab.add(tok)

    labels = sorted(class_counts.keys())
    total_docs = sum(class_counts.values())
    vocab_size = max(1, len(vocab))

    if total_docs == 0:
        return {
            "model_type": "multinomial_nb",
            "labels": [],
            "vocab_size": 0,
            "class_counts": {},
            "token_totals": {},
            "token_counts": {},
            "class_log_prior": {},
            "meta": {"rows_seen": seen, "rows_used": used},
        }

    if class_balance:
        max_count = max(class_counts.values())
        class_weights = {label: (max_count / max(1, class_counts[label])) for label in labels}
    else:
        class_weights = {label: 1.0 for label in labels}

    # Second pass to apply class-weighted counts.
    for row in rows:
        text = row.get("text")
        label = row.get("expected_intent")
        mode = row.get("mode", "unknown")
        if not isinstance(text, str) or not text.strip() or not isinstance(label, str) or not label.strip():
            continue

        label = label.strip()
        weight = float(class_weights.get(label, 1.0))
        tokens = _expand_ngrams(_tokenize(text), ngram_min=ngram_min, ngram_max=ngram_max)
        if isinstance(mode, str) and mode.strip():
            tokens.append(f"__mode_{mode.strip().lower()}")

        for tok in tokens:
            token_counts[label][tok] = float(token_counts[label].get(tok, 0.0)) + weight
            token_totals[label] += weight

    if class_prior_mode == "uniform":
        class_log_prior = {label: math.log(1.0 / max(1, len(labels))) for label in labels}
    else:
        class_log_prior = {
            label: math.log(class_counts[label] / total_docs)
            for label in labels
        }

    return {
        "model_type": "multinomial_nb",
        "labels": labels,
        "vocab_size": vocab_size,
        "class_counts": dict(class_counts),
        "token_totals": {label: float(token_totals[label]) for label in labels},
        "token_counts": {label: dict(token_counts[label]) for label in labels},
        "class_log_prior": class_log_prior,
        "feature_config": {
            "ngram_min": int(ngram_min),
            "ngram_max": int(ngram_max),
            "class_prior_mode": class_prior_mode,
            "class_balance": bool(class_balance),
        },
        "meta": {
            "rows_seen": seen,
            "rows_used": used,
        },
    }


def predict_intent(model: Dict[str, Any], text: str, mode: str = "unknown") -> Tuple[str, float, Dict[str, float]]:
    labels_any = model.get("labels")
    labels = labels_any if isinstance(labels_any, list) else []
    if not labels:
        return "other", 0.0, {}

    vocab_size = int(model.get("vocab_size", 1) or 1)
    token_totals_any = model.get("token_totals")
    token_totals = token_totals_any if isinstance(token_totals_any, dict) else {}
    token_counts_any = model.get("token_counts")
    token_counts = token_counts_any if isinstance(token_counts_any, dict) else {}
    class_log_prior_any = model.get("class_log_prior")
    class_log_prior = class_log_prior_any if isinstance(class_log_prior_any, dict) else {}

    feature_config_any = model.get("feature_config")
    feature_config = feature_config_any if isinstance(feature_config_any, dict) else {}
    ngram_min = int(feature_config.get("ngram_min", 1) or 1)
    ngram_max = int(feature_config.get("ngram_max", 1) or 1)

    tokens = _expand_ngrams(_tokenize(text), ngram_min=ngram_min, ngram_max=ngram_max)
    if isinstance(mode, str) and mode.strip():
        tokens.append(f"__mode_{mode.strip().lower()}")

    tf = Counter(tokens)
    log_scores: Dict[str, float] = {}

    for label in labels:
        score = float(class_log_prior.get(label, -100.0))
        total = float(token_totals.get(label, 0.0))
        denom = total + vocab_size
        counts_any = token_counts.get(label)
        counts = counts_any if isinstance(counts_any, dict) else {}

        for tok, freq in tf.items():
            c = float(counts.get(tok, 0.0))
            score += freq * math.log((c + 1.0) / denom)

        log_scores[label] = score

    max_score = max(log_scores.values())
    exps = {k: math.exp(v - max_score) for k, v in log_scores.items()}
    total_exp = sum(exps.values()) or 1.0
    probs = {k: (v / total_exp) for k, v in exps.items()}

    best_label = max(probs.items(), key=lambda item: item[1])[0]
    best_conf = float(probs[best_label])
    return best_label, best_conf, probs
