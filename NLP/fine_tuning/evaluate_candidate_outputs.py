#!/usr/bin/env python3
"""Evaluate fine-tuned candidate outputs against SFT test labels."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _safe_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _safe_entity_map(value: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(value, dict):
        return out
    for key, vals in value.items():
        if not isinstance(key, str) or not isinstance(vals, list):
            continue
        cleaned = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
        if cleaned:
            out[key] = cleaned
    return out


def _entity_pairs(entity_map: Dict[str, List[str]]) -> Set[str]:
    pairs: Set[str] = set()
    for key, vals in entity_map.items():
        for val in vals:
            normalized = normalize_entity_value(key, val)
            if normalized:
                pairs.add(f"{key}::{normalized}")
    return pairs


def _parse_reference(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    refs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        meta_any = row.get("meta")
        meta = meta_any if isinstance(meta_any, dict) else {}
        mode = str(meta.get("mode", "physician_portal"))

        messages_any = row.get("messages")
        messages = messages_any if isinstance(messages_any, list) else []
        user_text = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = str(msg.get("content", ""))
                marker = "Message:"
                if marker in content:
                    user_text = content.split(marker, 1)[1].strip().splitlines()[0].strip()
                else:
                    user_text = content.strip()
                break

        assistant_obj: Dict[str, Any] = {}
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                try:
                    parsed = json.loads(str(msg.get("content", "")))
                    if isinstance(parsed, dict):
                        assistant_obj = parsed
                except Exception:
                    assistant_obj = {}
                break

        if user_text:
            refs[(mode, user_text)] = assistant_obj

    return refs


def _parse_predictions(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    preds: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        mode = str(row.get("mode", "physician_portal"))
        text = row.get("text")
        pred = row.get("prediction")
        if not isinstance(text, str) or not text.strip() or not isinstance(pred, dict):
            continue
        preds[(mode, text.strip())] = pred
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned candidate outputs")
    parser.add_argument("--reference-openai-jsonl", required=True)
    parser.add_argument("--candidate-predictions-jsonl", required=True)
    parser.add_argument("--min-intent-accuracy", type=float, default=0.80)
    parser.add_argument("--min-clarification-recall", type=float, default=0.80)
    parser.add_argument("--min-entity-recall", type=float, default=0.45)
    parser.add_argument("--output-json", default="NLP/evaluation/results/ci_finetune_candidate_eval.json")
    args = parser.parse_args()

    ref_path = Path(args.reference_openai_jsonl).expanduser().resolve()
    pred_path = Path(args.candidate_predictions_jsonl).expanduser().resolve()

    if not ref_path.exists():
        print(f"Reference dataset not found: {ref_path}")
        return 1
    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}")
        return 1

    refs = _parse_reference(ref_path)
    preds = _parse_predictions(pred_path)

    total = 0
    matched = 0
    intent_correct = 0

    clar_tp = 0
    clar_fp = 0
    clar_fn = 0

    entity_tp = 0
    entity_fn = 0

    missing: List[Dict[str, str]] = []

    for key, expected in refs.items():
        total += 1
        pred = preds.get(key)
        if pred is None:
            missing.append({"mode": key[0], "text": key[1]})
            continue

        matched += 1

        exp_intent = str(expected.get("intent", "other"))
        pred_intent = str(pred.get("intent", "other"))
        if exp_intent == pred_intent:
            intent_correct += 1

        exp_clar = bool(expected.get("needs_clarification", False))
        pred_clar = bool(pred.get("needs_clarification", False))

        if exp_clar and pred_clar:
            clar_tp += 1
        elif (not exp_clar) and pred_clar:
            clar_fp += 1
        elif exp_clar and (not pred_clar):
            clar_fn += 1

        exp_entities = _entity_pairs(_safe_entity_map(expected.get("entity_map")))
        pred_entities = _entity_pairs(_safe_entity_map(pred.get("entity_map")))
        entity_tp += len(exp_entities & pred_entities)
        entity_fn += len(exp_entities - pred_entities)

    intent_accuracy = (intent_correct / matched) if matched else 0.0
    clarification_recall = (clar_tp / (clar_tp + clar_fn)) if (clar_tp + clar_fn) else 1.0
    entity_recall = (entity_tp / (entity_tp + entity_fn)) if (entity_tp + entity_fn) else 1.0
    coverage = (matched / total) if total else 0.0

    checks = {
        "coverage": coverage >= 0.98,
        "intent_accuracy": intent_accuracy >= args.min_intent_accuracy,
        "clarification_recall": clarification_recall >= args.min_clarification_recall,
        "entity_recall": entity_recall >= args.min_entity_recall,
    }
    failed = [name for name, ok in checks.items() if not ok]

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": str(ref_path),
        "candidate_predictions": str(pred_path),
        "result": {
            "reference_rows": total,
            "matched_predictions": matched,
            "coverage": round(coverage, 4),
            "intent_accuracy": round(intent_accuracy, 4),
            "clarification_recall": round(clarification_recall, 4),
            "entity_recall": round(entity_recall, 4),
            "missing_predictions": missing[:100],
        },
        "thresholds": {
            "min_intent_accuracy": args.min_intent_accuracy,
            "min_clarification_recall": args.min_clarification_recall,
            "min_entity_recall": args.min_entity_recall,
            "min_coverage": 0.98,
        },
        "quality_gate": "pass" if not failed else "fail",
        "failed_checks": failed,
    }

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Reference rows: {total}")
    print(f"Matched predictions: {matched}")
    print(f"Intent accuracy: {intent_accuracy:.2%}")
    print(f"Clarification recall: {clarification_recall:.2%}")
    print(f"Entity recall: {entity_recall:.2%}")
    print(f"Artifact saved to: {out_path}")

    if failed:
        print("QUALITY GATE: FAIL")
        for item in failed:
            print(f"- {item}")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
