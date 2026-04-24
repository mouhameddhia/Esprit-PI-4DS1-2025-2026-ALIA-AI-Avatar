"""Run basic NLP evaluation on labeled JSONL datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.nlp import analyze_message_nlp
from NLP.taxonomy.entity_normalization import normalize_entity_value
from NLP.taxonomy.validator import (
    load_taxonomy,
    validate_dataset_row_labels,
    validate_nlp_output,
)


THRESHOLD_PROFILES: Dict[str, Dict[str, float]] = {
    "production": {
        "min_intent_accuracy": 0.90,
        "min_safety_recall": 0.95,
        "min_secondary_tags_recall": 0.60,
        "min_entity_map_recall": 0.50,
    },
    "public_hardening": {
        "min_intent_accuracy": 0.50,
        "min_safety_recall": 0.95,
        "min_secondary_tags_recall": 0.30,
        "min_entity_map_recall": 0.10,
    },
}


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
                normalized = normalize_entity_value(key, value)
                if normalized:
                    pairs.add(f"{key}::{normalized}")
    return pairs


def _entity_map_values(entity_map: Any) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = defaultdict(set)
    if not isinstance(entity_map, dict):
        return {}

    for key, values in entity_map.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                normalized = normalize_entity_value(key, value)
                if normalized:
                    out[key].add(normalized)
    return dict(out)


def _tokenize_entity_value(value: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", value.lower()) if tok}


def _is_partial_entity_match(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    if expected in actual or actual in expected:
        return True

    expected_tokens = _tokenize_entity_value(expected)
    actual_tokens = _tokenize_entity_value(actual)
    if not expected_tokens or not actual_tokens:
        return False

    overlap = len(expected_tokens & actual_tokens)
    min_tokens = min(len(expected_tokens), len(actual_tokens))
    return min_tokens > 0 and (overlap / min_tokens) >= 0.6


def _partial_entity_counts(
    expected_values: Dict[str, set[str]],
    actual_values: Dict[str, set[str]],
) -> tuple[int, int, int]:
    tp = 0
    fp = 0
    fn = 0

    entity_types = set(expected_values) | set(actual_values)
    for entity_type in entity_types:
        expected_set = set(expected_values.get(entity_type, set()))
        actual_pool = set(actual_values.get(entity_type, set()))

        matched = 0
        for expected_item in expected_set:
            candidate = next(
                (actual_item for actual_item in actual_pool if _is_partial_entity_match(expected_item, actual_item)),
                None,
            )
            if candidate is not None:
                matched += 1
                actual_pool.remove(candidate)

        tp += matched
        fn += max(0, len(expected_set) - matched)
        fp += max(0, len(actual_values.get(entity_type, set())) - matched)

    return tp, fp, fn


def evaluate_dataset(dataset_path: Path) -> Dict[str, Any]:
    taxonomy = load_taxonomy()
    rows = _read_jsonl(dataset_path)
    supported_safety_flags = {
        item for item in (taxonomy.get("safety_flags") or []) if isinstance(item, str)
    }
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
            "safety_flag_metrics": {},
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

    partial_entity_tp = 0
    partial_entity_fp = 0
    partial_entity_fn = 0

    failures: List[Dict[str, Any]] = []
    dataset_issues: List[Dict[str, Any]] = []
    output_issues: List[Dict[str, Any]] = []
    intent_confusions: Dict[str, Counter[str]] = defaultdict(Counter)
    expected_intent_counts: Counter[str] = Counter()
    predicted_intent_counts: Counter[str] = Counter()
    safety_flag_tp: Counter[str] = Counter()
    safety_flag_fp: Counter[str] = Counter()
    safety_flag_fn: Counter[str] = Counter()
    safety_flag_support: Counter[str] = Counter()

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
        expected_entity_values = _entity_map_values(row.get("expected_entity_map", {}))

        actual = analyze_message_nlp(user_text=text, mode=mode, history=[])
        validation_issues = validate_nlp_output(actual, taxonomy)
        if validation_issues:
            output_issues.append({"row": idx, "issues": validation_issues, "text": text})

        actual_intent = str(actual.get("intent", "other"))
        actual_flags = _safe_set(actual.get("safety_flags", []))
        actual_tags = _safe_set(actual.get("secondary_tags", []))
        actual_entity_pairs = _entity_pairs(actual.get("entity_map", {}))
        actual_entity_values = _entity_map_values(actual.get("entity_map", {}))

        expected_intent_counts[expected_intent] += 1
        predicted_intent_counts[actual_intent] += 1
        intent_confusions[expected_intent][actual_intent] += 1

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

        for flag in supported_safety_flags:
            in_expected = flag in expected_flags
            in_actual = flag in actual_flags
            if in_expected:
                safety_flag_support[flag] += 1
            if in_expected and in_actual:
                safety_flag_tp[flag] += 1
            elif in_actual and not in_expected:
                safety_flag_fp[flag] += 1
            elif in_expected and not in_actual:
                safety_flag_fn[flag] += 1

        tags_tp += len(actual_tags & expected_tags)
        tags_fp += len(actual_tags - expected_tags)
        tags_fn += len(expected_tags - actual_tags)

        entity_tp += len(actual_entity_pairs & expected_entity_pairs)
        entity_fp += len(actual_entity_pairs - expected_entity_pairs)
        entity_fn += len(expected_entity_pairs - actual_entity_pairs)

        p_tp, p_fp, p_fn = _partial_entity_counts(expected_entity_values, actual_entity_values)
        partial_entity_tp += p_tp
        partial_entity_fp += p_fp
        partial_entity_fn += p_fn

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
    partial_entity_map_precision = (
        partial_entity_tp / (partial_entity_tp + partial_entity_fp)
        if (partial_entity_tp + partial_entity_fp)
        else 1.0
    )
    partial_entity_map_recall = (
        partial_entity_tp / (partial_entity_tp + partial_entity_fn)
        if (partial_entity_tp + partial_entity_fn)
        else 1.0
    )

    seen_intents = set(expected_intent_counts) | set(predicted_intent_counts)
    ordered_intents = sorted(seen_intents)

    intent_confusion_matrix: Dict[str, Dict[str, int]] = {}
    for expected in ordered_intents:
        intent_confusion_matrix[expected] = {
            actual: intent_confusions[expected].get(actual, 0)
            for actual in ordered_intents
        }

    intent_metrics: Dict[str, Dict[str, float | int]] = {}
    for intent in ordered_intents:
        tp_intent = intent_confusions[intent].get(intent, 0)
        support = expected_intent_counts.get(intent, 0)
        predicted = predicted_intent_counts.get(intent, 0)

        precision = (tp_intent / predicted) if predicted else 0.0
        recall = (tp_intent / support) if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        intent_metrics[intent] = {
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    top_confusions: List[Dict[str, Any]] = []
    for expected in ordered_intents:
        for actual in ordered_intents:
            if expected == actual:
                continue
            count = intent_confusions[expected].get(actual, 0)
            if count <= 0:
                continue
            top_confusions.append(
                {
                    "expected": expected,
                    "actual": actual,
                    "count": count,
                }
            )
    top_confusions.sort(key=lambda item: item["count"], reverse=True)

    safety_flag_metrics: Dict[str, Dict[str, float | int]] = {}
    for flag in sorted(supported_safety_flags):
        tp_flag = safety_flag_tp.get(flag, 0)
        fp_flag = safety_flag_fp.get(flag, 0)
        fn_flag = safety_flag_fn.get(flag, 0)
        support = safety_flag_support.get(flag, 0)

        precision = (tp_flag / (tp_flag + fp_flag)) if (tp_flag + fp_flag) else 1.0
        recall = (tp_flag / (tp_flag + fn_flag)) if (tp_flag + fn_flag) else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        safety_flag_metrics[flag] = {
            "support": support,
            "tp": tp_flag,
            "fp": fp_flag,
            "fn": fn_flag,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "count": len(rows),
        "intent_accuracy": round(intent_accuracy, 4),
        "safety_precision": round(safety_precision, 4),
        "safety_recall": round(safety_recall, 4),
        "secondary_tags_precision": round(secondary_tags_precision, 4),
        "secondary_tags_recall": round(secondary_tags_recall, 4),
        "entity_map_precision": round(entity_map_precision, 4),
        "entity_map_recall": round(entity_map_recall, 4),
        "partial_entity_map_precision": round(partial_entity_map_precision, 4),
        "partial_entity_map_recall": round(partial_entity_map_recall, 4),
        "dataset_issues": dataset_issues,
        "output_issues": output_issues,
        "failures": failures,
        "intent_confusion_matrix": intent_confusion_matrix,
        "intent_metrics": intent_metrics,
        "top_intent_confusions": top_confusions[:20],
        "safety_flag_metrics": safety_flag_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NLP pipeline on labeled JSONL data")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "eval_intent_safety_template.jsonl"),
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--threshold-profile",
        choices=sorted(THRESHOLD_PROFILES.keys()),
        default="production",
        help="Threshold profile to apply before any explicit --min-* overrides",
    )
    parser.add_argument("--min-intent-accuracy", type=float)
    parser.add_argument("--min-safety-recall", type=float)
    parser.add_argument("--min-secondary-tags-recall", type=float)
    parser.add_argument("--min-entity-map-recall", type=float)
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

    profile_thresholds = THRESHOLD_PROFILES.get(args.threshold_profile, THRESHOLD_PROFILES["production"])
    min_intent_accuracy = args.min_intent_accuracy if args.min_intent_accuracy is not None else profile_thresholds["min_intent_accuracy"]
    min_safety_recall = args.min_safety_recall if args.min_safety_recall is not None else profile_thresholds["min_safety_recall"]
    min_secondary_tags_recall = (
        args.min_secondary_tags_recall
        if args.min_secondary_tags_recall is not None
        else profile_thresholds["min_secondary_tags_recall"]
    )
    min_entity_map_recall = (
        args.min_entity_map_recall
        if args.min_entity_map_recall is not None
        else profile_thresholds["min_entity_map_recall"]
    )

    result = evaluate_dataset(dataset_path)

    print(f"Samples: {result['count']}")
    print(f"Intent accuracy: {result['intent_accuracy']:.2%}")
    print(f"Safety precision: {result['safety_precision']:.2%}")
    print(f"Safety recall: {result['safety_recall']:.2%}")
    print(f"Secondary tags precision: {result['secondary_tags_precision']:.2%}")
    print(f"Secondary tags recall: {result['secondary_tags_recall']:.2%}")
    print(f"Entity-map precision: {result['entity_map_precision']:.2%}")
    print(f"Entity-map recall: {result['entity_map_recall']:.2%}")
    print(f"Partial entity-map precision: {result['partial_entity_map_precision']:.2%}")
    print(f"Partial entity-map recall: {result['partial_entity_map_recall']:.2%}")

    if result.get("dataset_issues"):
        print(f"Dataset validation issues: {len(result['dataset_issues'])}")
    if result.get("output_issues"):
        print(f"Model output schema issues: {len(result['output_issues'])}")

    failures = result.get("failures") or []
    if failures:
        print("\nTop mismatches:")
        for item in failures[:10]:
            print(json.dumps(item, ensure_ascii=False))

    top_confusions = result.get("top_intent_confusions") or []
    if top_confusions:
        print("\nTop intent confusions:")
        for item in top_confusions[:10]:
            print(f"- {item['expected']} -> {item['actual']}: {item['count']}")

    safety_flag_metrics = result.get("safety_flag_metrics") or {}
    non_zero_safety_flags = [
        (flag, metrics)
        for flag, metrics in safety_flag_metrics.items()
        if int(metrics.get("support", 0)) > 0 or int(metrics.get("fp", 0)) > 0
    ]
    if non_zero_safety_flags:
        print("\nSafety flag metrics:")
        for flag, metrics in non_zero_safety_flags:
            print(
                f"- {flag}: support={metrics['support']} "
                f"p={metrics['precision']:.2f} r={metrics['recall']:.2f} f1={metrics['f1']:.2f}"
            )

    checks = {
        "intent_accuracy": (result["intent_accuracy"], min_intent_accuracy),
        "safety_recall": (result["safety_recall"], min_safety_recall),
        "secondary_tags_recall": (
            result["secondary_tags_recall"],
            min_secondary_tags_recall,
        ),
        "entity_map_recall": (result["entity_map_recall"], min_entity_map_recall),
    }

    failed = [name for name, (actual, threshold) in checks.items() if actual < threshold]

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "dataset": str(dataset_path),
        "result": result,
        "thresholds": {
            "profile": args.threshold_profile,
            "min_intent_accuracy": min_intent_accuracy,
            "min_safety_recall": min_safety_recall,
            "min_secondary_tags_recall": min_secondary_tags_recall,
            "min_entity_map_recall": min_entity_map_recall,
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
