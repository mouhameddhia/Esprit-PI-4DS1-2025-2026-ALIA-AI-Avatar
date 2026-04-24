#!/usr/bin/env python3
"""Evaluate trained entity model in shadow mode against production NLP entity output."""

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

from NLP.pipeline.entity_model import predict_entity_map
from NLP.pipeline.nlp import analyze_message_nlp
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


def _load_model_artifact(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_thresholds(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else payload
    out: Dict[str, float] = {}
    for key in (
        "min_shadow_entity_recall",
        "max_divergence",
        "max_train_test_gap",
        "max_recall_regression",
    ):
        val = thresholds.get(key) if isinstance(thresholds, dict) else None
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


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


def evaluate(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    shadow_tp = 0
    shadow_fp = 0
    shadow_fn = 0
    prod_tp = 0
    prod_fp = 0
    prod_fn = 0
    divergence_rows = 0

    failures: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        text = row.get("text")
        expected_map = row.get("expected_entity_map")
        mode = str(row.get("mode", "physician_portal"))
        if not isinstance(text, str) or not isinstance(expected_map, dict):
            continue

        total += 1
        expected_pairs = _pairs(expected_map)

        shadow_map = predict_entity_map(model, text)
        shadow_pairs = _pairs(shadow_map)

        prod_map = analyze_message_nlp(user_text=text, mode=mode, history=[]).get("entity_map", {})
        prod_pairs = _pairs(prod_map)

        shadow_tp += len(shadow_pairs & expected_pairs)
        shadow_fp += len(shadow_pairs - expected_pairs)
        shadow_fn += len(expected_pairs - shadow_pairs)

        prod_tp += len(prod_pairs & expected_pairs)
        prod_fp += len(prod_pairs - expected_pairs)
        prod_fn += len(expected_pairs - prod_pairs)

        if shadow_pairs != prod_pairs:
            divergence_rows += 1

        if shadow_pairs != expected_pairs:
            failures.append(
                {
                    "row": idx,
                    "text": text,
                    "expected_pairs": sorted(expected_pairs),
                    "shadow_pairs": sorted(shadow_pairs),
                    "production_pairs": sorted(prod_pairs),
                }
            )

    shadow_precision = (shadow_tp / (shadow_tp + shadow_fp)) if (shadow_tp + shadow_fp) else 1.0
    shadow_recall = (shadow_tp / (shadow_tp + shadow_fn)) if (shadow_tp + shadow_fn) else 1.0
    production_recall = (prod_tp / (prod_tp + prod_fn)) if (prod_tp + prod_fn) else 1.0
    divergence_rate = (divergence_rows / total) if total else 0.0

    return {
        "count": total,
        "shadow_entity_precision": round(shadow_precision, 4),
        "shadow_entity_recall": round(shadow_recall, 4),
        "production_entity_recall": round(production_recall, 4),
        "shadow_recall_delta": round(shadow_recall - production_recall, 4),
        "divergence_rate": round(divergence_rate, 4),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run trained entity shadow evaluation")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--model-json", default="NLP/evaluation/results/ci_entity_model_v1.json")
    parser.add_argument("--thresholds-json", default="")
    parser.add_argument("--min-shadow-entity-recall", type=float, default=0.35)
    parser.add_argument("--max-divergence", type=float, default=0.60)
    parser.add_argument("--max-train-test-gap", type=float, default=0.35)
    parser.add_argument("--max-recall-regression", type=float, default=0.30)
    parser.add_argument("--output-json", default="NLP/evaluation/results/ci_trained_entity_shadow_eval.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    model_path = Path(args.model_json).expanduser().resolve()

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1
    if not model_path.exists():
        print(f"Model artifact not found: {model_path}")
        return 1

    if args.thresholds_json:
        thresholds_path = Path(args.thresholds_json).expanduser().resolve()
        if not thresholds_path.exists():
            print(f"Thresholds config not found: {thresholds_path}")
            return 1
        loaded = _load_thresholds(thresholds_path)
        if "min_shadow_entity_recall" in loaded:
            args.min_shadow_entity_recall = loaded["min_shadow_entity_recall"]
        if "max_divergence" in loaded:
            args.max_divergence = loaded["max_divergence"]
        if "max_train_test_gap" in loaded:
            args.max_train_test_gap = loaded["max_train_test_gap"]
        if "max_recall_regression" in loaded:
            args.max_recall_regression = loaded["max_recall_regression"]

    rows = _read_jsonl(dataset_path)
    model_artifact = _load_model_artifact(model_path)
    model = model_artifact.get("model", {}) if isinstance(model_artifact.get("model"), dict) else {}
    result = evaluate(rows, model)

    train_recall = None
    train_metrics_any = model_artifact.get("train_metrics")
    train_metrics: Dict[str, Any] = train_metrics_any if isinstance(train_metrics_any, dict) else {}
    train_recall_raw = train_metrics.get("entity_recall")
    if isinstance(train_recall_raw, (int, float)):
        train_recall = float(train_recall_raw)

    train_test_gap = None
    if isinstance(train_recall, float):
        train_test_gap = train_recall - float(result["shadow_entity_recall"])

    recall_regression = float(result["production_entity_recall"]) - float(result["shadow_entity_recall"])

    checks = {
        "shadow_entity_recall": float(result["shadow_entity_recall"]) >= args.min_shadow_entity_recall,
        "divergence": float(result["divergence_rate"]) <= args.max_divergence,
        "recall_regression": recall_regression <= args.max_recall_regression,
    }
    if isinstance(train_test_gap, float):
        checks["train_test_gap"] = train_test_gap <= args.max_train_test_gap

    failed_checks = [name for name, ok in checks.items() if not ok]

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "model_json": str(model_path),
        "result": result,
        "generalization": {
            "train_entity_recall": round(train_recall, 4) if isinstance(train_recall, float) else None,
            "test_entity_recall": result["shadow_entity_recall"],
            "train_test_gap": round(train_test_gap, 4) if isinstance(train_test_gap, float) else None,
            "recall_regression_vs_production": round(recall_regression, 4),
        },
        "thresholds": {
            "min_shadow_entity_recall": args.min_shadow_entity_recall,
            "max_divergence": args.max_divergence,
            "max_train_test_gap": args.max_train_test_gap,
            "max_recall_regression": args.max_recall_regression,
        },
        "quality_gate": "pass" if not failed_checks else "fail",
        "failed_checks": failed_checks,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Samples: {result['count']}")
    print(f"Shadow entity recall: {result['shadow_entity_recall']:.2%}")
    print(f"Production entity recall: {result['production_entity_recall']:.2%}")
    print(f"Divergence rate: {result['divergence_rate']:.2%}")
    if isinstance(train_test_gap, float):
        print(f"Train-test gap: {train_test_gap:+.2%}")
    print(f"Recall regression vs production: {recall_regression:+.2%}")
    print(f"Artifact saved to: {output_path}")

    if failed_checks:
        print("QUALITY GATE: FAIL")
        for item in failed_checks:
            print(f"- {item}")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())