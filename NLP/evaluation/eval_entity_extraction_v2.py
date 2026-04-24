#!/usr/bin/env python3
"""Evaluate entity extraction improvements (v2) against baseline.

Compares:
1. Dictionary-only (exact match)
2. Dictionary + fuzzy matching
3. Dictionary + fuzzy + spaCy NER (if available)

Measures recall, precision, and F1 on evaluation dataset.
"""

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

from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2


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


def _extract_expected_entities(entity_map: Any) -> Set[str]:
    """Extract normalized entity set from expected entity map."""
    entities: Set[str] = set()
    if not isinstance(entity_map, dict):
        return entities
    for entity_type, values in entity_map.items():
        if not isinstance(entity_type, str) or not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                entities.add(value.strip().lower())
    return entities


def _extract_predicted_entities(entities_dict: Dict[str, Any]) -> Set[str]:
    """Extract normalized entity set from predicted entities."""
    entities: Set[str] = set()
    for entity_type, entity_list in entities_dict.items():
        if not isinstance(entity_list, list):
            continue
        for entity in entity_list:
            if isinstance(entity, dict):
                value = entity.get("value")
                if isinstance(value, str) and value.strip():
                    entities.add(value.strip().lower())
    return entities


def evaluate_method(
    rows: List[Dict[str, Any]],
    extractor: EntityExtractorV2,
    use_fuzzy: bool = False,
    use_ner: bool = False,
) -> Dict[str, Any]:
    """Evaluate entity extraction method."""
    tp = 0
    fp = 0
    fn = 0
    failures: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        text = row.get("text")
        expected_entity_map = row.get("expected_entity_map")
        if not isinstance(text, str) or not isinstance(expected_entity_map, dict):
            continue

        expected = _extract_expected_entities(expected_entity_map)
        predicted_dict = extractor.extract_entities(text, use_fuzzy=use_fuzzy, use_ner=use_ner)
        predicted = _extract_predicted_entities(predicted_dict)

        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

        if predicted != expected:
            failures.append({
                "row": idx,
                "text": text,
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "missed": sorted(expected - predicted),
                "false_positives": sorted(predicted - expected),
            })

    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "sample_failures": failures[:5],  # Top 5 failures for debugging
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate entity extraction improvements (v2)")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "eval_intent_safety_v5.jsonl"),
        help="Path to entity evaluation dataset",
    )
    parser.add_argument("--output-json", default="NLP/evaluation/results/eval_entity_extraction_v2.json")
    parser.add_argument("--min-recall", type=float, default=0.80)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    rows = _read_jsonl(dataset_path)
    if not rows:
        print(f"No valid rows found in: {dataset_path}")
        return 1

    print(f"Evaluating on {len(rows)} rows from {dataset_path}")
    extractor = EntityExtractorV2(use_spacy=False)  # Set to True if spaCy is available

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "methods": {},
    }

    # Method 1: Exact match only
    print("\nMethod 1: Exact match (baseline)...")
    baseline_result = evaluate_method(rows, extractor, use_fuzzy=False, use_ner=False)
    results["methods"]["exact_match"] = baseline_result
    print(f"  Recall: {baseline_result['recall']:.2%} | Precision: {baseline_result['precision']:.2%} | F1: {baseline_result['f1']:.4f}")

    # Method 2: Exact + Fuzzy
    print("\nMethod 2: Exact + Fuzzy matching...")
    fuzzy_result = evaluate_method(rows, extractor, use_fuzzy=True, use_ner=False)
    results["methods"]["exact_fuzzy"] = fuzzy_result
    print(f"  Recall: {fuzzy_result['recall']:.2%} | Precision: {fuzzy_result['precision']:.2%} | F1: {fuzzy_result['f1']:.4f}")

    # Method 3: Exact + Fuzzy + NER (if available)
    print("\nMethod 3: Exact + Fuzzy + NER (if available)...")
    ner_result = evaluate_method(rows, extractor, use_fuzzy=True, use_ner=True)
    results["methods"]["exact_fuzzy_ner"] = ner_result
    print(f"  Recall: {ner_result['recall']:.2%} | Precision: {ner_result['precision']:.2%} | F1: {ner_result['f1']:.4f}")

    # Summary
    print("\n=== SUMMARY ===")
    baseline_recall = baseline_result["recall"]
    fuzzy_recall = fuzzy_result["recall"]
    ner_recall = ner_result["recall"]
    print(f"Baseline recall: {baseline_recall:.2%}")
    print(f"Fuzzy recall: {fuzzy_recall:.2%} (delta: {fuzzy_recall - baseline_recall:+.2%})")
    print(f"Fuzzy+NER recall: {ner_recall:.2%} (delta: {ner_recall - baseline_recall:+.2%})")

    # Quality gate
    best_recall = max(fuzzy_recall, ner_recall)
    gate_passed = best_recall >= args.min_recall
    results["quality_gate"] = "pass" if gate_passed else "fail"
    results["summary"] = {
        "best_method": "exact_fuzzy_ner" if ner_recall > fuzzy_recall else "exact_fuzzy",
        "best_recall": max(baseline_recall, fuzzy_recall, ner_recall),
        "improvement_vs_baseline": max(fuzzy_recall - baseline_recall, ner_recall - baseline_recall),
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArtifact saved to: {output_path}")

    if not gate_passed:
        print(f"QUALITY GATE: FAIL (best recall {best_recall:.2%} < {args.min_recall:.2%})")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())