"""Run basic NLP evaluation on labeled JSONL datasets."""

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
from NLP.taxonomy.validator import (
    load_taxonomy,
    validate_dataset_row_labels,
    validate_nlp_output,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {v for v in values if isinstance(v, str)}


def _entity_pairs(entity_map: Any) -> set[str]:
    if not isinstance(entity_map, dict):
        return set()
    pairs: set[str] = set()
    for key, values in entity_map.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                pairs.add(f"{key}::{value.strip().lower()}")
    return pairs


def evaluate_dataset(dataset_path: Path) -> Dict[str, Any]:
    taxonomy = load_taxonomy()
    rows = _read_jsonl(dataset_path)
    if not rows:
        return {
            "count": 0,
            "intent_accuracy": 0.0,
            "safety_precision": 0.0,
            "safety_recall": 0.0,
            "secondary_tags_precision": 0.0,
            "secondary_tags_recall": 0.0,
            "entity_map_precision": 0.0,
            "entity_map_recall": 0.0,
            "dataset_issues": [],
            "output_issues": [],
        }

    correct_intent = 0

    tp = 0
    fp = 0
    fn = 0

    tags_tp = 0
    tags_fp = 0
    tags_fn = 0

    entity_tp = 0
    entity_fp = 0
    entity_fn = 0

    failures: List[Dict[str, Any]] = []
    dataset_issues: List[Dict[str, Any]] = []
    output_issues: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        row_issues = validate_dataset_row_labels(row, taxonomy)
        if row_issues:
            dataset_issues.append({"row": idx, "issues": row_issues})

        text = str(row.get("text", ""))
        mode = str(row.get("mode", "physician_portal"))
        expected_intent = str(row.get("expected_intent", "other"))
        expected_flags = _safe_set(row.get("expected_safety_flags", []))
        expected_tags = _safe_set(row.get("expected_secondary_tags", []))
        expected_entity_pairs = _entity_pairs(row.get("expected_entity_map", {}))

        actual = analyze_message_nlp(user_text=text, mode=mode, history=[])
        validation_issues = validate_nlp_output(actual, taxonomy)
        if validation_issues:
            output_issues.append({"row": idx, "issues": validation_issues, "text": text})

        actual_intent = str(actual.get("intent", "other"))
        actual_flags = _safe_set(actual.get("safety_flags", []))
        actual_tags = _safe_set(actual.get("secondary_tags", []))
        actual_entity_pairs = _entity_pairs(actual.get("entity_map", {}))

        if actual_intent == expected_intent:
            correct_intent += 1
        else:
            failures.append(
                {
                    "row": idx,
                    "type": "intent",
                    "text": text,
                    "expected": expected_intent,
                    "actual": actual_intent,
                }
            )

        tp += len(actual_flags & expected_flags)
        fp += len(actual_flags - expected_flags)
        fn += len(expected_flags - actual_flags)

        tags_tp += len(actual_tags & expected_tags)
        tags_fp += len(actual_tags - expected_tags)
        tags_fn += len(expected_tags - actual_tags)

        entity_tp += len(actual_entity_pairs & expected_entity_pairs)
        entity_fp += len(actual_entity_pairs - expected_entity_pairs)
        entity_fn += len(expected_entity_pairs - actual_entity_pairs)

        if actual_flags != expected_flags:
            failures.append(
                {
                    "row": idx,
                    "type": "safety_flags",
                    "text": text,
                    "expected": sorted(expected_flags),
                    "actual": sorted(actual_flags),
                }
            )

    intent_accuracy = correct_intent / len(rows)
    safety_precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    safety_recall = (tp / (tp + fn)) if (tp + fn) else 1.0
    secondary_tags_precision = (tags_tp / (tags_tp + tags_fp)) if (tags_tp + tags_fp) else 1.0
    secondary_tags_recall = (tags_tp / (tags_tp + tags_fn)) if (tags_tp + tags_fn) else 1.0
    entity_map_precision = (entity_tp / (entity_tp + entity_fp)) if (entity_tp + entity_fp) else 1.0
    entity_map_recall = (entity_tp / (entity_tp + entity_fn)) if (entity_tp + entity_fn) else 1.0

    return {
        "count": len(rows),
        "intent_accuracy": round(intent_accuracy, 4),
        "safety_precision": round(safety_precision, 4),
        "safety_recall": round(safety_recall, 4),
        "secondary_tags_precision": round(secondary_tags_precision, 4),
        "secondary_tags_recall": round(secondary_tags_recall, 4),
        "entity_map_precision": round(entity_map_precision, 4),
        "entity_map_recall": round(entity_map_recall, 4),
        "dataset_issues": dataset_issues,
        "output_issues": output_issues,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NLP pipeline on labeled JSONL data")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "eval_intent_safety_template.jsonl"),
        help="Path to JSONL dataset",
    )
    parser.add_argument("--min-intent-accuracy", type=float, default=0.90)
    parser.add_argument("--min-safety-recall", type=float, default=0.95)
    parser.add_argument("--min-secondary-tags-recall", type=float, default=0.60)
    parser.add_argument("--min-entity-map-recall", type=float, default=0.50)
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "last_eval.json"),
        help="Path to save evaluation result JSON artifact",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    result = evaluate_dataset(dataset_path)

    print(f"Samples: {result['count']}")
    print(f"Intent accuracy: {result['intent_accuracy']:.2%}")
    print(f"Safety precision: {result['safety_precision']:.2%}")
    print(f"Safety recall: {result['safety_recall']:.2%}")
    print(f"Secondary tags precision: {result['secondary_tags_precision']:.2%}")
    print(f"Secondary tags recall: {result['secondary_tags_recall']:.2%}")
    print(f"Entity-map precision: {result['entity_map_precision']:.2%}")
    print(f"Entity-map recall: {result['entity_map_recall']:.2%}")

    if result.get("dataset_issues"):
        print(f"Dataset validation issues: {len(result['dataset_issues'])}")
    if result.get("output_issues"):
        print(f"Model output schema issues: {len(result['output_issues'])}")

    failures = result.get("failures") or []
    if failures:
        print("\nTop mismatches:")
        for item in failures[:10]:
            print(json.dumps(item, ensure_ascii=False))

    checks = {
        "intent_accuracy": (result["intent_accuracy"], args.min_intent_accuracy),
        "safety_recall": (result["safety_recall"], args.min_safety_recall),
        "secondary_tags_recall": (
            result["secondary_tags_recall"],
            args.min_secondary_tags_recall,
        ),
        "entity_map_recall": (result["entity_map_recall"], args.min_entity_map_recall),
    }

    failed = [name for name, (actual, threshold) in checks.items() if actual < threshold]

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "dataset": str(dataset_path),
        "result": result,
        "thresholds": {
            "min_intent_accuracy": args.min_intent_accuracy,
            "min_safety_recall": args.min_safety_recall,
            "min_secondary_tags_recall": args.min_secondary_tags_recall,
            "min_entity_map_recall": args.min_entity_map_recall,
        },
        "quality_gate": "fail" if failed else "pass",
        "failed_checks": failed,
    }
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artifact saved to: {output_path}")

    if failed:
        print("\nQUALITY GATE: FAIL")
        for name in failed:
            actual, threshold = checks[name]
            print(f"- {name}: {actual:.4f} < {threshold:.4f}")
        return 2

    print("\nQUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
