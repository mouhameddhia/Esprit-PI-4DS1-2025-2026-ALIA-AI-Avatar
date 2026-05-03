"""Evaluation runner — runs the pipeline against labeled JSONL datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alia_nlp.src.pipeline.online import analyze_message_nlp
from alia_nlp.data.taxonomy.validator import (
    load_taxonomy,
    validate_dataset_row_labels,
    validate_nlp_output,
)
from alia_nlp.evaluation.metrics import _safe_set, _entity_pairs, precision_recall


def evaluate_dataset(dataset_path: Path) -> Dict[str, Any]:
    taxonomy = load_taxonomy()
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"count": 0}

    correct_intent = 0
    s_tp = s_fp = s_fn = 0
    t_tp = t_fp = t_fn = 0
    e_tp = e_fp = e_fn = 0
    failures: List[Dict] = []
    dataset_issues: List[Dict] = []
    output_issues: List[Dict] = []

    for idx, row in enumerate(rows, start=1):
        issues = validate_dataset_row_labels(row, taxonomy)
        if issues:
            dataset_issues.append({"row": idx, "issues": issues})

        text   = str(row.get("text", ""))
        mode   = str(row.get("mode", "physician_portal"))
        e_int  = str(row.get("expected_intent", "other"))
        e_safe = _safe_set(row.get("expected_safety_flags", []))
        e_tags = _safe_set(row.get("expected_secondary_tags", []))
        e_ent  = _entity_pairs(row.get("expected_entity_map", {}))

        actual = analyze_message_nlp(user_text=text, mode=mode, history=[])
        out_issues = validate_nlp_output(actual, taxonomy)
        if out_issues:
            output_issues.append({"row": idx, "issues": out_issues, "text": text})

        a_int  = str(actual.get("intent", "other"))
        a_safe = _safe_set(actual.get("safety_flags", []))
        a_tags = _safe_set(actual.get("secondary_tags", []))
        a_ent  = _entity_pairs(actual.get("entity_map", {}))

        if a_int == e_int:
            correct_intent += 1
        else:
            failures.append({"row": idx, "type": "intent", "text": text,
                             "expected": e_int, "actual": a_int})

        s_tp += len(a_safe & e_safe); s_fp += len(a_safe - e_safe); s_fn += len(e_safe - a_safe)
        t_tp += len(a_tags & e_tags); t_fp += len(a_tags - e_tags); t_fn += len(e_tags - a_tags)
        e_tp += len(a_ent  & e_ent);  e_fp += len(a_ent  - e_ent);  e_fn += len(e_ent  - a_ent)

        if a_safe != e_safe:
            failures.append({"row": idx, "type": "safety_flags", "text": text,
                             "expected": sorted(e_safe), "actual": sorted(a_safe)})

    sp, sr = precision_recall(s_tp, s_fp, s_fn)
    tp_, tr = precision_recall(t_tp, t_fp, t_fn)
    ep, er  = precision_recall(e_tp, e_fp, e_fn)

    return {
        "count": len(rows),
        "intent_accuracy": round(correct_intent / len(rows), 4),
        "safety_precision": sp,  "safety_recall": sr,
        "secondary_tags_precision": tp_, "secondary_tags_recall": tr,
        "entity_map_precision": ep, "entity_map_recall": er,
        "dataset_issues": dataset_issues,
        "output_issues": output_issues,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NLP pipeline on labeled JSONL")
    parser.add_argument("dataset", nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "eval_intent_safety_v6.jsonl"))
    parser.add_argument("--min-intent-accuracy",        type=float, default=0.90)
    parser.add_argument("--min-safety-recall",          type=float, default=0.95)
    parser.add_argument("--min-secondary-tags-recall",  type=float, default=0.60)
    parser.add_argument("--min-entity-map-recall",      type=float, default=0.50)
    parser.add_argument("--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "last_eval.json"))
    args = parser.parse_args()

    path = Path(args.dataset).expanduser().resolve()
    if not path.exists():
        print(f"Dataset not found: {path}"); return 1

    result = evaluate_dataset(path)

    print(f"Samples:                  {result['count']}")
    print(f"Intent accuracy:          {result['intent_accuracy']:.2%}")
    print(f"Safety precision/recall:  {result['safety_precision']:.2%} / {result['safety_recall']:.2%}")
    print(f"Tags precision/recall:    {result['secondary_tags_precision']:.2%} / {result['secondary_tags_recall']:.2%}")
    print(f"Entity precision/recall:  {result['entity_map_precision']:.2%} / {result['entity_map_recall']:.2%}")

    checks = {
        "intent_accuracy":       (result["intent_accuracy"],      args.min_intent_accuracy),
        "safety_recall":         (result["safety_recall"],         args.min_safety_recall),
        "secondary_tags_recall": (result["secondary_tags_recall"], args.min_secondary_tags_recall),
        "entity_map_recall":     (result["entity_map_recall"],     args.min_entity_map_recall),
    }
    failed = [k for k, (v, t) in checks.items() if v < t]

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "dataset": str(path), "result": result,
        "thresholds": {f"min_{k}": v for k, (_, v) in checks.items()},
        "quality_gate": "fail" if failed else "pass",
        "failed_checks": failed,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArtifact: {out_path}")

    if failed:
        print("QUALITY GATE: FAIL")
        for k in failed:
            v, t = checks[k]
            print(f"  {k}: {v:.4f} < {t:.4f}")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
