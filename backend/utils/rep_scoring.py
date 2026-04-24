"""Secondary NLP scoring for representative responses.

This module computes three quality scores from rep text:
- clarity_score (1-10)
- persuasion_score (1-10)
- confidence_score (1-10)

Design:
- Uses transformer embeddings through backend.embeddings.EmbeddingEncoder.
- Uses sklearn models (regression + classification).
- Loads persisted artifacts when available.
- Falls back to weakly supervised bootstrap models if artifacts are missing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from ..embeddings import EmbeddingEncoder

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1] / "model_artifacts" / "rep_scorer"
)


@dataclass
class RepScoreResult:
    clarity_score: float
    persuasion_score: float
    confidence_score: float
    model_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "clarity_score": self.clarity_score,
            "persuasion_score": self.persuasion_score,
            "confidence_score": self.confidence_score,
            "model_source": self.model_source,
        }


class RepResponseScorer:
    """Scores rep text with transformer embeddings + sklearn models."""

    def __init__(
        self,
        embedding_encoder: EmbeddingEncoder | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.encoder = embedding_encoder or EmbeddingEncoder()
        self.artifact_dir = Path(
            artifact_dir
            or os.getenv("REP_SCORER_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
        )

        self._clarity_model: Ridge | None = None
        self._persuasion_model: Ridge | None = None
        self._confidence_model: LogisticRegression | None = None

        self.model_source = "bootstrap"
        self._is_ready = self.encoder.is_ready()

        if not self._is_ready:
            logger.warning("RepResponseScorer initialized without ready embedding encoder")
            return

        if not self._load_models():
            self._train_bootstrap_models()

    def is_ready(self) -> bool:
        return self._is_ready and self._clarity_model is not None and self._persuasion_model is not None and self._confidence_model is not None

    def score_text(self, rep_text: str) -> RepScoreResult:
        text = (rep_text or "").strip()
        if not text:
            return RepScoreResult(
                clarity_score=1.0,
                persuasion_score=1.0,
                confidence_score=1.0,
                model_source=self.model_source,
            )

        if not self.is_ready():
            return RepScoreResult(
                clarity_score=1.0,
                persuasion_score=1.0,
                confidence_score=1.0,
                model_source="unavailable",
            )

        features = self._features_for_texts([text])

        clarity_pred = float(self._clarity_model.predict(features)[0])
        persuasion_pred = float(self._persuasion_model.predict(features)[0])

        # Confidence is modeled as 10-way classification and converted back to 1..10.
        confidence_class = int(self._confidence_model.predict(features)[0])

        return RepScoreResult(
            clarity_score=_clamp_score(clarity_pred),
            persuasion_score=_clamp_score(persuasion_pred),
            confidence_score=_clamp_score(float(confidence_class)),
            model_source=self.model_source,
        )

    def _load_models(self) -> bool:
        try:
            clarity_path = self.artifact_dir / "clarity_regressor.joblib"
            persuasion_path = self.artifact_dir / "persuasion_regressor.joblib"
            confidence_path = self.artifact_dir / "confidence_classifier.joblib"

            if not (clarity_path.exists() and persuasion_path.exists() and confidence_path.exists()):
                return False

            self._clarity_model = joblib.load(clarity_path)
            self._persuasion_model = joblib.load(persuasion_path)
            self._confidence_model = joblib.load(confidence_path)
            self.model_source = "artifact"
            logger.info("Loaded rep scorer models from %s", self.artifact_dir)
            return True
        except Exception as exc:
            logger.warning("Failed to load rep scorer artifacts: %s", exc)
            return False

    def _features_for_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.encoder.encode_batch(texts)
        dense = np.asarray(embeddings, dtype=np.float32)

        handcrafted = np.asarray([
            _handcrafted_features(text) for text in texts
        ], dtype=np.float32)

        return np.concatenate([dense, handcrafted], axis=1)

    def _train_bootstrap_models(self) -> None:
        samples = _bootstrap_samples()
        texts = [row["text"] for row in samples]

        x = self._features_for_texts(texts)
        y_clarity = np.asarray([row["clarity"] for row in samples], dtype=np.float32)
        y_persuasion = np.asarray([row["persuasion"] for row in samples], dtype=np.float32)
        y_confidence = np.asarray([row["confidence"] for row in samples], dtype=np.int32)

        self._clarity_model = Ridge(alpha=1.0, random_state=42)
        self._clarity_model.fit(x, y_clarity)

        self._persuasion_model = Ridge(alpha=1.0, random_state=42)
        self._persuasion_model.fit(x, y_persuasion)

        self._confidence_model = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        self._confidence_model.fit(x, y_confidence)

        self.model_source = "bootstrap"
        logger.info("Initialized bootstrap rep scorer models")


def _clamp_score(value: float) -> float:
    return round(float(max(1.0, min(10.0, value))), 2)


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _count_markers(text_lower: str, markers: Iterable[str]) -> int:
    return sum(1 for marker in markers if marker in text_lower)


def _handcrafted_features(text: str) -> list[float]:
    lowered = text.lower().strip()
    words = [w for w in lowered.split() if w]
    word_count = len(words)
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))

    evidence_markers = [
        "study",
        "trial",
        "data",
        "guideline",
        "meta-analysis",
        "published",
        "evidence",
    ]
    persuasion_markers = [
        "benefit",
        "improve",
        "reduce",
        "outcome",
        "value",
        "supports",
        "recommend",
    ]
    confidence_markers = [
        "i'm confident",
        "clearly",
        "strongly",
        "demonstrates",
        "proven",
        "robust",
    ]
    hedge_markers = [
        "maybe",
        "perhaps",
        "might",
        "possibly",
        "i think",
        "not sure",
    ]

    avg_word_len = _safe_div(sum(len(w) for w in words), max(1, word_count))

    return [
        float(word_count),
        float(sentence_count),
        float(avg_word_len),
        float(_count_markers(lowered, evidence_markers)),
        float(_count_markers(lowered, persuasion_markers)),
        float(_count_markers(lowered, confidence_markers)),
        float(_count_markers(lowered, hedge_markers)),
        float(text.count(",")),
    ]


def _bootstrap_samples() -> list[dict[str, Any]]:
    # Weak supervision seed set to ensure the module runs out of the box.
    # Replace with trained artifacts for production behavior.
    return [
        {
            "text": "CardioGuard reduced systolic pressure by 12 mmHg in a phase III trial, with consistent benefit across high-risk cohorts.",
            "clarity": 8.8,
            "persuasion": 8.4,
            "confidence": 9,
        },
        {
            "text": "I think this product might help some patients, but I would need to check the data.",
            "clarity": 5.0,
            "persuasion": 4.2,
            "confidence": 3,
        },
        {
            "text": "The key benefit is once-daily dosing with strong adherence outcomes and fewer drop-offs versus baseline.",
            "clarity": 8.0,
            "persuasion": 8.1,
            "confidence": 8,
        },
        {
            "text": "There are safety concerns, and perhaps we can revisit later.",
            "clarity": 5.4,
            "persuasion": 3.1,
            "confidence": 2,
        },
        {
            "text": "In a published study, neuro events were reduced while maintaining favorable tolerability over 24 weeks.",
            "clarity": 8.5,
            "persuasion": 7.9,
            "confidence": 8,
        },
        {
            "text": "The mechanism is probably okay, but I am not sure how much it changes outcomes.",
            "clarity": 5.3,
            "persuasion": 4.0,
            "confidence": 3,
        },
        {
            "text": "This recommendation is supported by multicenter evidence and aligns with recent guideline updates.",
            "clarity": 8.7,
            "persuasion": 8.3,
            "confidence": 9,
        },
        {
            "text": "It could be useful depending on the patient.",
            "clarity": 4.4,
            "persuasion": 3.6,
            "confidence": 2,
        },
        {
            "text": "To summarize: better control, manageable safety profile, and measurable quality-of-life improvements.",
            "clarity": 8.2,
            "persuasion": 8.0,
            "confidence": 8,
        },
        {
            "text": "Maybe it helps, maybe not.",
            "clarity": 2.8,
            "persuasion": 1.8,
            "confidence": 1,
        },
        {
            "text": "Compared with standard therapy, we observed lower hospitalization risk and faster symptom stabilization.",
            "clarity": 8.6,
            "persuasion": 8.5,
            "confidence": 9,
        },
        {
            "text": "I think we can discuss this later after I verify a few points.",
            "clarity": 5.0,
            "persuasion": 3.4,
            "confidence": 3,
        },
        {
            "text": "The evidence demonstrates durable efficacy and supports adoption in appropriate adult populations.",
            "clarity": 8.5,
            "persuasion": 8.6,
            "confidence": 9,
        },
        {
            "text": "There might be benefit but I cannot say clearly right now.",
            "clarity": 4.5,
            "persuasion": 3.2,
            "confidence": 2,
        },
        {
            "text": "We recommend this option because the trial data show meaningful reductions in exacerbations.",
            "clarity": 8.4,
            "persuasion": 8.7,
            "confidence": 9,
        },
    ]
