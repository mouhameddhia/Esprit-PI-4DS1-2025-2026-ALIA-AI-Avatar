#!/usr/bin/env python3
"""Measure divergence between primary and shadow NLP predictions.

Expected input: JSONL with at least these keys per row:
- text: str
- primary_intent: str
- shadow_intent: str

Optional keys:
- primary_confidence: float
- shadow_confidence: float
- ground_truth_intent: str
- mode: str
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            rows.append(item)
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_shadow_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "count": 0,
            "divergence_rate": 0.0,
            "agreement_rate": 0.0,
            "by_primary_intent": {},
            "top_disagreements": [],
        }

    disagreements = 0
    disagreement_pairs: Counter[str] = Counter()
    by_primary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "disagreements": 0})

    primary_conf_sum = 0.0
    primary_conf_count = 0
    shadow_conf_sum = 0.0
    shadow_conf_count = 0

    primary_correct = 0
    shadow_correct = 0
    labeled_rows = 0

    row_issues: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        primary_intent = str(row.get("primary_intent", "")).strip()
        shadow_intent = str(row.get("shadow_intent", "")).strip()

        issues: List[str] = []
        if not primary_intent:
            issues.append("missing primary_intent")
        if not shadow_intent:
            issues.append("missing shadow_intent")
        if not text:
            issues.append("missing text")
        if issues:
            row_issues.append({"row": idx, "issues": issues})
            continue

        by_primary[primary_intent]["count"] += 1
        if primary_intent != shadow_intent:
            disagreements += 1
            by_primary[primary_intent]["disagreements"] += 1
            disagreement_pairs[f"{primary_intent} -> {shadow_intent}"] += 1

        primary_conf = _safe_float(row.get("primary_confidence"))
        if primary_conf is not None:
            primary_conf_sum += primary_conf
            primary_conf_count += 1

        shadow_conf = _safe_float(row.get("shadow_confidence"))
        if shadow_conf is not None:
            shadow_conf_sum += shadow_conf
            shadow_conf_count += 1

        ground_truth = str(row.get("ground_truth_intent", "")).strip()
        if ground_truth:
            labeled_rows += 1
            if primary_intent == ground_truth:
                primary_correct += 1
            if shadow_intent == ground_truth:
                shadow_correct += 1

    valid_rows = total - len(row_issues)
    if valid_rows <= 0:
        return {
            "count": total,
            "valid_rows": 0,
            "row_issues": row_issues,
            "divergence_rate": 0.0,
            "agreement_rate": 0.0,
            "by_primary_intent": {},
            "top_disagreements": [],
        }

    by_primary_report: Dict[str, Dict[str, float]] = {}
    for intent, stats in sorted(by_primary.items()):
        count = stats["count"]
        divergent = stats["disagreements"]
        by_primary_report[intent] = {
            "count": count,
            "disagreements": divergent,
            "divergence_rate": round((divergent / count) if count else 0.0, 4),
        }

    report: Dict[str, Any] = {
        "count": total,
        "valid_rows": valid_rows,
        "row_issues": row_issues,
        "divergence_rate": round(disagreements / valid_rows, 4),
        "agreement_rate": round(1.0 - (disagreements / valid_rows), 4),
        "avg_primary_confidence": round(primary_conf_sum / primary_conf_count, 4) if primary_conf_count else None,
        "avg_shadow_confidence": round(shadow_conf_sum / shadow_conf_count, 4) if shadow_conf_count else None,
        "by_primary_intent": by_primary_report,
        "top_disagreements": [
            {"pair": pair, "count": count}
            for pair, count in disagreement_pairs.most_common(10)
        ],
    }

    if labeled_rows:
        report["labeled_rows"] = labeled_rows
        report["primary_intent_accuracy"] = round(primary_correct / labeled_rows, 4)
        report["shadow_intent_accuracy"] = round(shadow_correct / labeled_rows, 4)
        report["shadow_accuracy_delta"] = round((shadow_correct - primary_correct) / labeled_rows, 4)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze intent divergence between primary and shadow NLP outputs")
    parser.add_argument("input_jsonl", help="Path to JSONL log with primary/shadow predictions")
    parser.add_argument(
        "--max-divergence",
        type=float,
        default=0.15,
        help="Maximum acceptable divergence rate before failing quality gate",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "shadow_latest.json"),
        help="Path to save shadow monitoring artifact",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    rows = _read_jsonl(input_path)
    report = build_shadow_report(rows)

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "input": str(input_path),
        "result": report,
        "thresholds": {
            "max_divergence": args.max_divergence,
        },
        "quality_gate": "pass" if report.get("divergence_rate", 0.0) <= args.max_divergence else "fail",
    }
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Samples: {report.get('count', 0)}")
    print(f"Valid rows: {report.get('valid_rows', 0)}")
    print(f"Divergence rate: {report.get('divergence_rate', 0.0):.2%}")
    print(f"Agreement rate: {report.get('agreement_rate', 0.0):.2%}")

    if report.get("avg_primary_confidence") is not None:
        print(f"Avg primary confidence: {report['avg_primary_confidence']:.4f}")
    if report.get("avg_shadow_confidence") is not None:
        print(f"Avg shadow confidence: {report['avg_shadow_confidence']:.4f}")

    if report.get("labeled_rows"):
        print(f"Primary intent accuracy: {report['primary_intent_accuracy']:.2%}")
        print(f"Shadow intent accuracy: {report['shadow_intent_accuracy']:.2%}")
        delta = report["shadow_accuracy_delta"]
        print(f"Shadow accuracy delta: {delta:+.2%}")

    top_disagreements = report.get("top_disagreements") or []
    if top_disagreements:
        print("\nTop disagreement pairs:")
        for item in top_disagreements[:5]:
            print(f"- {item['pair']}: {item['count']}")

    row_issues = report.get("row_issues") or []
    if row_issues:
        print(f"\nRow issues: {len(row_issues)}")

    print(f"Artifact saved to: {output_path}")

    if artifact["quality_gate"] == "fail":
        print("\nQUALITY GATE: FAIL")
        print(f"- divergence_rate: {report.get('divergence_rate', 0.0):.4f} > {args.max_divergence:.4f}")
        return 2

    print("\nQUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
