"""Train secondary NLP scoring models for rep responses.

Expected input JSONL format (one record per line):
{
  "text": "rep utterance",
  "clarity_score": 1..10,
  "persuasion_score": 1..10,
  "confidence_score": 1..10
}

Usage:
  python backend/scripts/train_rep_response_scorer.py \
      --input backend/data/rep_scoring_train.jsonl \
      --out backend/model_artifacts/rep_scorer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.embeddings import EmbeddingEncoder
from backend.utils.rep_scoring import _handcrafted_features


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _clamp_score(score: Any) -> float:
    try:
        value = float(score)
    except Exception:
        value = 1.0
    return max(1.0, min(10.0, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL training data")
    parser.add_argument("--out", required=True, help="Output artifact directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(input_path)
    if not rows:
        raise RuntimeError("No training rows found")

    texts = [str(row.get("text", "")).strip() for row in rows]
    if not all(texts):
        raise RuntimeError("All rows must include non-empty text")

    encoder = EmbeddingEncoder()
    if not encoder.is_ready():
        raise RuntimeError("Embedding encoder not ready")

    dense = np.asarray(encoder.encode_batch(texts), dtype=np.float32)
    handcrafted = np.asarray([_handcrafted_features(text) for text in texts], dtype=np.float32)
    features = np.concatenate([dense, handcrafted], axis=1)

    y_clarity = np.asarray([_clamp_score(row.get("clarity_score", 1.0)) for row in rows], dtype=np.float32)
    y_persuasion = np.asarray([_clamp_score(row.get("persuasion_score", 1.0)) for row in rows], dtype=np.float32)
    y_confidence = np.asarray([int(round(_clamp_score(row.get("confidence_score", 1.0)))) for row in rows], dtype=np.int32)

    clarity_model = Ridge(alpha=1.0, random_state=42)
    clarity_model.fit(features, y_clarity)

    persuasion_model = Ridge(alpha=1.0, random_state=42)
    persuasion_model.fit(features, y_persuasion)

    confidence_model = LogisticRegression(
        max_iter=3000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    confidence_model.fit(features, y_confidence)

    joblib.dump(clarity_model, out_dir / "clarity_regressor.joblib")
    joblib.dump(persuasion_model, out_dir / "persuasion_regressor.joblib")
    joblib.dump(confidence_model, out_dir / "confidence_classifier.joblib")

    print(f"Saved artifacts to: {out_dir}")
    print(f"Rows used: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
