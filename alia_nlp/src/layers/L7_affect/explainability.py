"""
LIME and SHAP explainability for text-based affect predictions.

Public API
----------
explain_text(text, mode, method="both", top_k=10) -> dict
    Unified entry point — returns SHAP and/or LIME token attributions.

explain_text_shap(text, mode, top_k) -> dict
    Token-level SHAP values via partition explainer (model-agnostic).

explain_text_lime(text, mode, top_k, num_samples) -> dict
    Word-level LIME attributions for confidence, frustration, stress heads.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

_CONF_CLASSES = ["low", "medium", "high"]
_ALL_TARGETS  = ["confidence_high", "frustration", "stress"]


# ---------------------------------------------------------------------------
# Shared prediction wrapper
# ---------------------------------------------------------------------------

def _make_predict_fn(mode: str, target: str):
    """
    Return a (texts -> probability_matrix) callable for a single affect target.

    target options: "confidence_high" (col 2), "frustration" (col 5), "stress" (col 6)
    """
    from .classifier import predict_proba

    col = {"confidence_high": 2, "frustration": 5, "stress": 6}.get(target, 2)

    def _fn(texts):
        proba = predict_proba(list(texts), mode)
        p = proba[:, col].reshape(-1, 1)
        return np.hstack([1 - p, p])   # binary [negative, positive]

    return _fn


# ---------------------------------------------------------------------------
# Text SHAP
# ---------------------------------------------------------------------------

def explain_text_shap(
    text: str,
    mode: str,
    top_k: int = 10,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """
    Token-level SHAP attributions using the partition (mask-based) explainer.

    Works entirely at the word/subword level without access to model internals.
    Returns importance scores for each requested affect dimension.
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap package not installed — run: pip install shap", "method": "shap"}

    from .classifier import get_tokenizer

    tokenizer = get_tokenizer()
    if tokenizer is None:
        return {"error": "Affect classifier not loaded — no model at alia_nlp/models/affect_classifier/", "method": "shap"}

    targets = targets or _ALL_TARGETS
    results: dict[str, Any] = {"method": "shap", "text": text, "dimensions": {}}

    for target in targets:
        fn = _make_predict_fn(mode, target)
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(fn, masker, output_names=["negative", "positive"])

        try:
            sv = explainer([text], max_evals=200, silent=True)
        except Exception as exc:
            results["dimensions"][target] = {"error": str(exc)}
            continue

        tokens = sv.data[0]
        # Column 1 = "positive" class importance
        values = sv.values[0][:, 1]
        base   = float(sv.base_values[0][1])

        pairs = sorted(
            [{"token": str(t), "importance": round(float(v), 4)} for t, v in zip(tokens, values)],
            key=lambda x: abs(x["importance"]),
            reverse=True,
        )
        results["dimensions"][target] = {
            "tokens":     pairs[:top_k],
            "base_value": round(base, 4),
        }

    return results


# ---------------------------------------------------------------------------
# Text LIME
# ---------------------------------------------------------------------------

def explain_text_lime(
    text: str,
    mode: str,
    top_k: int = 10,
    num_samples: int = 200,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """
    Word-level LIME attributions.

    Perturbs the input by randomly removing words and observes how each
    affect dimension changes. Faster than SHAP for long texts.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return {"error": "lime package not installed — run: pip install lime", "method": "lime"}

    targets = targets or _ALL_TARGETS
    results: dict[str, Any] = {"method": "lime", "text": text, "dimensions": {}}

    for target in targets:
        fn = _make_predict_fn(mode, target)
        explainer = LimeTextExplainer(class_names=["negative", "positive"])

        try:
            exp = explainer.explain_instance(
                text, fn,
                num_features=top_k,
                num_samples=num_samples,
                labels=[1],
            )
        except Exception as exc:
            results["dimensions"][target] = {"error": str(exc)}
            continue

        results["dimensions"][target] = {
            "tokens": [
                {"token": t, "importance": round(float(v), 4)}
                for t, v in exp.as_list(label=1)
            ],
            "prediction_prob": round(float(exp.predict_proba[1]), 4),
        }

    return results


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def explain_text(
    text: str,
    mode: str,
    method: Literal["shap", "lime", "both"] = "both",
    top_k: int = 10,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run SHAP, LIME, or both on *text* and return a combined explanation dict.

    Parameters
    ----------
    text    : the message to explain
    mode    : "medrep_training" | "physician_portal"
    method  : which explainer(s) to run
    top_k   : number of tokens to return per dimension
    targets : subset of ["confidence_high", "frustration", "stress"]
    """
    out: dict[str, Any] = {"text": text, "mode": mode}

    if method in ("shap", "both"):
        out["shap"] = explain_text_shap(text, mode, top_k=top_k, targets=targets)
    if method in ("lime", "both"):
        out["lime"] = explain_text_lime(text, mode, top_k=top_k, targets=targets)

    return out
