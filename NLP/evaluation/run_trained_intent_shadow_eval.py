#!/usr/bin/env python3
"""Evaluate trained intent model in shadow mode against production NLP outputs."""

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

from NLP.pipeline.intent_model import predict_intent
from NLP.pipeline.nlp import analyze_message_nlp


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
    for key in ("min_shadow_accuracy", "max_divergence", "max_train_test_gap"):
        val = thresholds.get(key) if isinstance(thresholds, dict) else None
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


def evaluate(dataset_rows: List[Dict[str, Any]], model: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    model_correct = 0
    prod_correct = 0
    divergence = 0
    failures: List[Dict[str, Any]] = []

    for idx, row in enumerate(dataset_rows, start=1):
        text = row.get("text")
        expected = row.get("expected_intent")
        mode = row.get("mode", "physician_portal")

        if not isinstance(text, str) or not text.strip() or not isinstance(expected, str) or not expected.strip():
            continue

        total += 1
        shadow_pred, shadow_conf, _ = predict_intent(model=model, text=text, mode=str(mode))
        prod_pred = str(analyze_message_nlp(user_text=text, mode=str(mode), history=[]).get("intent", "other"))

        if shadow_pred == expected:
            model_correct += 1
        if prod_pred == expected:
            prod_correct += 1
        if shadow_pred != prod_pred:
            divergence += 1

        if shadow_pred != expected:
            failures.append(
                {
                    "row": idx,
                    "text": text,
                    "expected": expected,
                    "shadow_pred": shadow_pred,
                    "shadow_confidence": round(shadow_conf, 4),
                    "production_pred": prod_pred,
                }
            )

    shadow_accuracy = (model_correct / total) if total else 0.0
    prod_accuracy = (prod_correct / total) if total else 0.0
    divergence_rate = (divergence / total) if total else 0.0

    return {
        "count": total,
        "shadow_intent_accuracy": round(shadow_accuracy, 4),
        "production_intent_accuracy": round(prod_accuracy, 4),
        "shadow_accuracy_delta": round(shadow_accuracy - prod_accuracy, 4),
        "divergence_rate": round(divergence_rate, 4),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run trained intent shadow evaluation")
    parser.add_argument("--dataset-jsonl", default="NLP/datasets/eval_intent_safety_v4.jsonl")
    parser.add_argument("--model-json", default="NLP/evaluation/results/ci_intent_model_v1.json")
    parser.add_argument("--thresholds-json", default="")
    parser.add_argument("--min-shadow-accuracy", type=float, default=0.85)
    parser.add_argument("--max-divergence", type=float, default=0.40)
    parser.add_argument("--max-train-test-gap", type=float, default=0.10)
    parser.add_argument("--output-json", default="NLP/evaluation/results/ci_trained_intent_shadow_eval.json")
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
        if "min_shadow_accuracy" in loaded:
            args.min_shadow_accuracy = loaded["min_shadow_accuracy"]
        if "max_divergence" in loaded:
            args.max_divergence = loaded["max_divergence"]
        if "max_train_test_gap" in loaded:
            args.max_train_test_gap = loaded["max_train_test_gap"]

    rows = _read_jsonl(dataset_path)
    model_artifact = _load_model_artifact(model_path)
    model = model_artifact.get("model", {}) if isinstance(model_artifact.get("model"), dict) else {}
    result = evaluate(rows, model)

    train_acc = None
    train_metrics = model_artifact.get("train_metrics") if isinstance(model_artifact.get("train_metrics"), dict) else {}
    if isinstance(train_metrics.get("train_accuracy"), (int, float)):
        train_acc = float(train_metrics.get("train_accuracy"))

    train_test_gap = None
    if isinstance(train_acc, float):
        train_test_gap = train_acc - float(result["shadow_intent_accuracy"])

    checks = {
        "shadow_accuracy": result["shadow_intent_accuracy"] >= args.min_shadow_accuracy,
        "divergence": result["divergence_rate"] <= args.max_divergence,
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
            "train_accuracy": round(train_acc, 4) if isinstance(train_acc, float) else None,
            "test_accuracy": result["shadow_intent_accuracy"],
            "train_test_gap": round(train_test_gap, 4) if isinstance(train_test_gap, float) else None,
        },
        "thresholds": {
            "min_shadow_accuracy": args.min_shadow_accuracy,
            "max_divergence": args.max_divergence,
            "max_train_test_gap": args.max_train_test_gap,
        },
        "quality_gate": "pass" if not failed_checks else "fail",
        "failed_checks": failed_checks,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Samples: {result['count']}")
    print(f"Shadow accuracy: {result['shadow_intent_accuracy']:.2%}")
    print(f"Production accuracy: {result['production_intent_accuracy']:.2%}")
    print(f"Shadow delta: {result['shadow_accuracy_delta']:+.2%}")
    print(f"Divergence rate: {result['divergence_rate']:.2%}")
    if isinstance(train_test_gap, float):
        print(f"Train-test gap: {train_test_gap:+.2%}")
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
