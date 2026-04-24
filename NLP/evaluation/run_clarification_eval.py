#!/usr/bin/env python3
"""Evaluate low-confidence clarification behavior on labeled ambiguity data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.nlp import analyze_message_nlp


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            rows.append(row)
    return rows


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def evaluate_dataset(path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        return {
            "count": 0,
            "clarification_precision": 0.0,
            "clarification_recall": 0.0,
            "clarification_f1": 0.0,
            "clarification_accuracy": 0.0,
            "clear_intent_accuracy": 0.0,
            "failures": [],
            "dataset_issues": [],
        }

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    clear_correct = 0
    clear_total = 0

    failures: List[Dict[str, Any]] = []
    dataset_issues: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            dataset_issues.append({"row": idx, "issues": ["missing text"]})
            continue

        mode = row.get("mode", "physician_portal")
        if not isinstance(mode, str) or not mode.strip():
            mode = "physician_portal"

        expected_clarification = _to_bool(row.get("expected_clarification", False))
        expected_intent = row.get("expected_intent")
        if expected_intent is not None and not isinstance(expected_intent, str):
            dataset_issues.append({"row": idx, "issues": ["expected_intent must be string when provided"]})
            expected_intent = None

        actual = analyze_message_nlp(user_text=text.strip(), mode=mode, history=[])
        actual_action_items = actual.get("action_items", [])
        actual_intent = str(actual.get("intent", "other"))
        actual_clarification = (
            isinstance(actual_action_items, list)
            and "needs_intent_clarification" in actual_action_items
        )

        if expected_clarification and actual_clarification:
            tp += 1
        elif not expected_clarification and actual_clarification:
            fp += 1
        elif expected_clarification and not actual_clarification:
            fn += 1
        else:
            tn += 1

        if expected_intent and not expected_clarification:
            clear_total += 1
            if actual_intent == expected_intent:
                clear_correct += 1

        if actual_clarification != expected_clarification:
            failures.append(
                {
                    "row": idx,
                    "type": "clarification",
                    "text": text,
                    "expected": expected_clarification,
                    "actual": actual_clarification,
                    "actual_intent": actual_intent,
                }
            )
        elif expected_intent and not expected_clarification and actual_intent != expected_intent:
            failures.append(
                {
                    "row": idx,
                    "type": "clear_intent",
                    "text": text,
                    "expected": expected_intent,
                    "actual": actual_intent,
                }
            )

    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) else 0.0
    clear_intent_accuracy = (clear_correct / clear_total) if clear_total else 1.0

    return {
        "count": len(rows),
        "clarification_precision": round(precision, 4),
        "clarification_recall": round(recall, 4),
        "clarification_f1": round(f1, 4),
        "clarification_accuracy": round(accuracy, 4),
        "clear_intent_accuracy": round(clear_intent_accuracy, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "failures": failures,
        "dataset_issues": dataset_issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate clarification behavior for low-confidence NLP cases")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "eval_intent_clarification_v1.jsonl"),
        help="Path to clarification evaluation dataset",
    )
    parser.add_argument("--min-clarification-recall", type=float, default=0.80)
    parser.add_argument("--min-clarification-precision", type=float, default=0.70)
    parser.add_argument("--min-clear-intent-accuracy", type=float, default=0.85)
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "eval_clarification_latest.json"),
        help="Path to write evaluation artifact",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    result = evaluate_dataset(dataset_path)
    print(f"Samples: {result['count']}")
    print(f"Clarification precision: {result['clarification_precision']:.2%}")
    print(f"Clarification recall: {result['clarification_recall']:.2%}")
    print(f"Clarification F1: {result['clarification_f1']:.2%}")
    print(f"Clarification accuracy: {result['clarification_accuracy']:.2%}")
    print(f"Clear-intent accuracy: {result['clear_intent_accuracy']:.2%}")

    checks = {
        "clarification_recall": result["clarification_recall"] >= args.min_clarification_recall,
        "clarification_precision": result["clarification_precision"] >= args.min_clarification_precision,
        "clear_intent_accuracy": result["clear_intent_accuracy"] >= args.min_clear_intent_accuracy,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    artifact = {
        "dataset": str(dataset_path),
        "result": result,
        "thresholds": {
            "min_clarification_recall": args.min_clarification_recall,
            "min_clarification_precision": args.min_clarification_precision,
            "min_clear_intent_accuracy": args.min_clear_intent_accuracy,
        },
        "quality_gate": "fail" if failed_checks else "pass",
        "failed_checks": failed_checks,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artifact saved to: {output_path}")

    if failed_checks:
        print("QUALITY GATE: FAIL")
        for name in failed_checks:
            print(f"- {name}")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())