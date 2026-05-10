"""
LIME and SHAP explainability for audio-based affect (SER).

Strategy: segment-masking
  The audio waveform is split into N equal time segments.
  Each segment is treated as a binary feature: present (1) or silenced (0).
  LIME/SHAP measure which segments most influence the emotion confidence.

The text transcription (if provided) is explained separately using the
text-based LIME/SHAP from L7_affect.explainability, giving word-level insight
into what the speaker said that drove the prediction.

Public API
----------
explain_audio(audio_np, sr, transcription, method, n_segments, top_k) -> dict
explain_audio_lime(audio_np, sr, n_segments, num_samples, top_k)       -> dict
explain_audio_shap(audio_np, sr, n_segments, top_k)                    -> dict
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ser_confidence(audio_np: np.ndarray, sr: int) -> float:
    """Return emotion confidence for *audio_np* (0.0 on failure)."""
    from backend.routes.audio import _classify_emotion_array
    result = _classify_emotion_array(audio_np, sr)
    return result["confidence"] if result else 0.0


def _build_segment_info(audio_np: np.ndarray, sr: int, n_segments: int) -> list[dict]:
    seg_len = max(1, len(audio_np) // n_segments)
    return [
        {
            "index":      i,
            "time_start": round(i * seg_len / sr, 2),
            "time_end":   round(min((i + 1) * seg_len, len(audio_np)) / sr, 2),
        }
        for i in range(n_segments)
    ]


def _apply_mask(audio_np: np.ndarray, mask: np.ndarray, seg_len: int) -> np.ndarray:
    masked = audio_np.copy().astype(np.float32)
    for i, keep in enumerate(mask):
        if keep < 0.5:
            s = i * seg_len
            e = min(s + seg_len, len(masked))
            masked[s:e] = 0.0
    return masked


# ---------------------------------------------------------------------------
# Audio LIME
# ---------------------------------------------------------------------------

def explain_audio_lime(
    audio_np: np.ndarray,
    sr: int,
    n_segments: int = 10,
    num_samples: int = 100,
    top_k: int = 6,
) -> dict[str, Any]:
    """
    Segment-level LIME: randomly silences subsets of time windows and fits
    a linear model to estimate each segment's contribution to the
    emotion confidence score.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return {"error": "lime not installed — run: pip install lime", "method": "lime_audio"}

    seg_len  = max(1, len(audio_np) // n_segments)
    seg_info = _build_segment_info(audio_np, sr, n_segments)
    feature_names = [f"seg_{s['index']} ({s['time_start']}s–{s['time_end']}s)" for s in seg_info]

    def _predict(masks: np.ndarray) -> np.ndarray:
        rows = []
        for mask in masks:
            masked  = _apply_mask(audio_np, mask, seg_len)
            conf    = _ser_confidence(masked, sr)
            rows.append([1.0 - conf, conf])
        return np.array(rows, dtype=float)

    background = np.zeros((1, n_segments))
    explainer  = LimeTabularExplainer(
        background,
        feature_names=feature_names,
        class_names=["low_emotion", "high_emotion"],
        discretize_continuous=False,
    )

    try:
        exp = explainer.explain_instance(
            np.ones(n_segments), _predict,
            num_features=top_k, num_samples=num_samples, labels=[1],
        )
    except Exception as exc:
        logger.error("Audio LIME failed: %s", exc)
        return {"method": "lime_audio", "error": str(exc)}

    segments = [
        {"feature": f, "importance": round(float(v), 4)}
        for f, v in exp.as_list(label=1)
    ]
    return {
        "method":           "lime_audio",
        "n_segments":       n_segments,
        "duration_seconds": round(len(audio_np) / sr, 2),
        "segments":         segments[:top_k],
        "prediction_prob":  round(float(exp.predict_proba[1]), 4),
    }


# ---------------------------------------------------------------------------
# Audio SHAP
# ---------------------------------------------------------------------------

def explain_audio_shap(
    audio_np: np.ndarray,
    sr: int,
    n_segments: int = 10,
    top_k: int = 6,
    nsamples: int = 80,
) -> dict[str, Any]:
    """
    Segment-level SHAP using KernelExplainer (model-agnostic).
    Background = all-zeros mask (full silence); test = all-ones (full audio).
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed — run: pip install shap", "method": "shap_audio"}

    seg_len  = max(1, len(audio_np) // n_segments)
    seg_info = _build_segment_info(audio_np, sr, n_segments)

    def _predict(masks: np.ndarray) -> np.ndarray:
        return np.array(
            [[_ser_confidence(_apply_mask(audio_np, m, seg_len), sr)] for m in masks],
            dtype=float,
        )

    background = np.zeros((1, n_segments))
    explainer  = shap.KernelExplainer(_predict, background)

    try:
        sv = explainer.shap_values(
            np.ones((1, n_segments)), nsamples=nsamples, l1_reg=f"num_features({top_k})", silent=True
        )
    except Exception as exc:
        logger.error("Audio SHAP failed: %s", exc)
        return {"method": "shap_audio", "error": str(exc)}

    values = (sv[0][0] if isinstance(sv, list) else sv[0]).tolist()
    segments = sorted(
        [
            {**seg_info[i], "importance": round(float(values[i]), 4)}
            for i in range(n_segments)
        ],
        key=lambda x: abs(x["importance"]),
        reverse=True,
    )
    return {
        "method":           "shap_audio",
        "n_segments":       n_segments,
        "duration_seconds": round(len(audio_np) / sr, 2),
        "segments":         segments[:top_k],
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def explain_audio(
    audio_np: np.ndarray,
    sr: int,
    transcription: str = "",
    method: Literal["shap", "lime", "both"] = "both",
    n_segments: int = 10,
    top_k: int = 6,
) -> dict[str, Any]:
    """
    Run SHAP and/or LIME on audio segments, plus text-level explanation
    on the transcription (if provided).

    Returns a unified explanation dict suitable for the AffectPanel.
    """
    out: dict[str, Any] = {"duration_seconds": round(len(audio_np) / sr, 2)}

    if method in ("lime", "both"):
        out["lime"] = explain_audio_lime(audio_np, sr, n_segments=n_segments, top_k=top_k)
    if method in ("shap", "both"):
        out["shap"] = explain_audio_shap(audio_np, sr, n_segments=n_segments, top_k=top_k)

    # Text-level explanation on the transcription
    if transcription.strip():
        from alia_nlp.src.layers.L7_affect.explainability import explain_text
        out["text_explanation"] = explain_text(
            transcription, mode="physician_portal", method=method, top_k=top_k
        )

    return out
