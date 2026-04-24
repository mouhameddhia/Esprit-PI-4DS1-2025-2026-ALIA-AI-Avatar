#!/usr/bin/env python3
"""Build a hard-negative labeling queue from shadow disagreement logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _build_queue(
    rows: List[Dict[str, Any]],
    min_pair_count: int,
    max_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    disagreements: List[Dict[str, Any]] = []
    pair_counter: Counter[str] = Counter()
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        text = str(row.get("text", "")).strip()
        primary_intent = str(row.get("primary_intent", "")).strip()
        shadow_intent = str(row.get("shadow_intent", "")).strip()
        mode = str(row.get("mode", "physician_portal")).strip() or "physician_portal"

        if not text or not primary_intent or not shadow_intent:
            continue
        if primary_intent == shadow_intent:
            continue

        pair = f"{primary_intent} -> {shadow_intent}"
        pair_counter[pair] += 1
        by_pair[pair].append(
            {
                "text": text,
                "mode": mode,
                "expected_intent": primary_intent,
                "shadow_intent": shadow_intent,
                "disagreement_pair": pair,
                "source": "shadow_logs",
            }
        )

    unique_texts: set[str] = set()
    for pair, count in pair_counter.most_common():
        if count < min_pair_count:
            continue

        for row in by_pair[pair]:
            text_key = row["text"].lower()
            if text_key in unique_texts:
                continue
            unique_texts.add(text_key)

            disagreements.append(
                {
                    **row,
                    "priority": count,
                    "priority_reason": "high_frequency_disagreement_pair",
                }
            )

            if len(disagreements) >= max_rows:
                break
        if len(disagreements) >= max_rows:
            break

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(rows),
        "disagreement_rows": sum(pair_counter.values()),
        "queue_rows": len(disagreements),
        "min_pair_count": min_pair_count,
        "max_rows": max_rows,
        "top_pairs": [
            {"pair": pair, "count": count}
            for pair, count in pair_counter.most_common(20)
        ],
    }

    return disagreements, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hard-negative queue from shadow disagreement logs")
    parser.add_argument("input_jsonl", help="Path to shadow logs JSONL")
    parser.add_argument(
        "--output-jsonl",
        default=str(Path(__file__).resolve().parent / "results" / "hard_negative_queue_latest.jsonl"),
        help="Output JSONL queue path",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "hard_negative_queue_latest.json"),
        help="Output summary JSON path",
    )
    parser.add_argument("--min-pair-count", type=int, default=2, help="Minimum disagreement pair frequency")
    parser.add_argument("--max-rows", type=int, default=200, help="Maximum queue rows")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    rows = _read_jsonl(input_path)
    queue_rows, summary = _build_queue(
        rows=rows,
        min_pair_count=max(1, args.min_pair_count),
        max_rows=max(1, args.max_rows),
    )

    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in queue_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input rows: {summary['input_rows']}")
    print(f"Disagreement rows: {summary['disagreement_rows']}")
    print(f"Queue rows written: {summary['queue_rows']}")
    if summary["top_pairs"]:
        print("Top disagreement pairs:")
        for item in summary["top_pairs"][:5]:
            print(f"- {item['pair']}: {item['count']}")
    print(f"Queue JSONL: {output_jsonl}")
    print(f"Summary JSON: {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
