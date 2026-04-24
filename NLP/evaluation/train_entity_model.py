#!/usr/bin/env python3
"""Train lightweight entity extraction model for Phase 4 shadow rollout."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.entity_model import predict_entity_map, train_entity_model
from NLP.taxonomy.entity_normalization import normalize_entity_value


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


def _pairs(entity_map: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(entity_map, dict):
        return out
    for key, values in entity_map.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = normalize_entity_value(key, value)
            if normalized:
                out.add(f"{key}::{normalized}")
    return out


def _entity_metrics(model: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, float | int]:
    tp = 0
    fp = 0
    fn = 0
    count = 0
    for row in rows:
        text = row.get("text")
        expected = row.get("expected_entity_map")
        if not isinstance(text, str) or not isinstance(expected, dict):
            continue
        count += 1
        pred = predict_entity_map(model, text)
        pred_pairs = _pairs(pred)
        exp_pairs = _pairs(expected)
        tp += len(pred_pairs & exp_pairs)
        fp += len(pred_pairs - exp_pairs)
        fn += len(exp_pairs - pred_pairs)

    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) else 1.0
    return {
        "samples": count,
        "entity_precision": round(precision, 4),
        "entity_recall": round(recall, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline entity extraction model")
    parser.add_argument("--dataset-jsonl", required=True, help="Training dataset path")
    parser.add_argument("--val-jsonl", default="", help="Optional validation dataset path")
    parser.add_argument("--min-value-support", type=int, default=1)
    parser.add_argument("--max-values-per-entity", type=int, default=200)
    parser.add_argument("--strategy", choices=["hybrid_lexicon", "lexicon"], default="hybrid_lexicon")
    parser.add_argument("--output-json", default="NLP/evaluation/results/ci_entity_model_v1.json")
    args = parser.parse_args()

    train_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not train_path.exists():
        print(f"Dataset not found: {train_path}")
        return 1

    train_rows = _read_jsonl(train_path)
    val_rows: List[Dict[str, Any]] = []
    val_path = None
    if args.val_jsonl:
        val_path = Path(args.val_jsonl).expanduser().resolve()
        if not val_path.exists():
            print(f"Validation dataset not found: {val_path}")
            return 1
        val_rows = _read_jsonl(val_path)

    model = train_entity_model(
        train_rows,
        min_value_support=max(1, args.min_value_support),
        max_values_per_entity=max(20, args.max_values_per_entity),
        strategy=args.strategy,
    )

    train_metrics = _entity_metrics(model, train_rows)
    val_metrics = _entity_metrics(model, val_rows) if val_rows else {"samples": 0, "entity_precision": None, "entity_recall": None}

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "entity_lex_v1",
        "train_dataset": str(train_path),
        "val_dataset": str(val_path) if val_path else None,
        "training_config": {
            "min_value_support": max(1, args.min_value_support),
            "max_values_per_entity": max(20, args.max_values_per_entity),
            "strategy": args.strategy,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model": model,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Train samples: {train_metrics['samples']}")
    print(f"Train entity precision: {train_metrics['entity_precision']:.2%}")
    print(f"Train entity recall: {train_metrics['entity_recall']:.2%}")
    if isinstance(val_metrics.get("entity_recall"), float):
        print(f"Val samples: {val_metrics['samples']}")
        print(f"Val entity recall: {val_metrics['entity_recall']:.2%}")
    print(f"Artifact saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())