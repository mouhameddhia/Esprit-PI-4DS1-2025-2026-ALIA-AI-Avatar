#!/usr/bin/env python3
"""Train a lightweight baseline intent model for Phase 4 shadow rollout."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.intent_model import predict_intent, train_intent_model


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _compute_accuracy(model: Dict[str, Any], rows: List[Dict[str, Any]]) -> float:
    total = 0
    correct = 0
    for row in rows:
        text = row.get("text")
        expected = row.get("expected_intent")
        mode = row.get("mode", "unknown")
        if not isinstance(text, str) or not isinstance(expected, str):
            continue
        pred, _, _ = predict_intent(model, text=text, mode=str(mode))
        total += 1
        if pred == expected:
            correct += 1
    return (correct / total) if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline intent model")
    parser.add_argument(
        "--dataset-jsonl",
        default="NLP/datasets/eval_intent_safety_v4.jsonl",
        help="Training dataset path",
    )
    parser.add_argument(
        "--val-jsonl",
        default="",
        help="Optional validation dataset path",
    )
    parser.add_argument(
        "--output-json",
        default="NLP/evaluation/results/ci_intent_model_v1.json",
        help="Output model artifact path",
    )
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument(
        "--class-prior-mode",
        default="uniform",
        choices=["uniform", "empirical"],
    )
    parser.add_argument("--disable-class-balance", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    rows = _read_jsonl(dataset_path)
    val_rows: List[Dict[str, Any]] = []
    val_path = None
    if args.val_jsonl:
        val_path = Path(args.val_jsonl).expanduser().resolve()
        if not val_path.exists():
            print(f"Validation dataset not found: {val_path}")
            return 1
        val_rows = _read_jsonl(val_path)

    model = train_intent_model(
        rows,
        ngram_min=max(1, args.ngram_min),
        ngram_max=max(max(1, args.ngram_min), args.ngram_max),
        class_prior_mode=args.class_prior_mode,
        class_balance=not args.disable_class_balance,
    )
    train_accuracy = _compute_accuracy(model, rows)
    val_accuracy = _compute_accuracy(model, val_rows) if val_rows else None

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "intent_nb_v2",
        "dataset": str(dataset_path),
        "train_dataset": str(dataset_path),
        "val_dataset": str(val_path) if val_path else None,
        "training_config": {
            "ngram_min": max(1, args.ngram_min),
            "ngram_max": max(max(1, args.ngram_min), args.ngram_max),
            "class_prior_mode": args.class_prior_mode,
            "class_balance": not args.disable_class_balance,
        },
        "train_metrics": {
            "samples": len(rows),
            "train_accuracy": round(train_accuracy, 4),
        },
        "val_metrics": {
            "samples": len(val_rows),
            "val_accuracy": round(val_accuracy, 4) if isinstance(val_accuracy, float) else None,
        },
        "model": model,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Samples: {len(rows)}")
    print(f"Train accuracy: {train_accuracy:.2%}")
    if isinstance(val_accuracy, float):
        print(f"Val samples: {len(val_rows)}")
        print(f"Val accuracy: {val_accuracy:.2%}")
    print(f"Artifact saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
