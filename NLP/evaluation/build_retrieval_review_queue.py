#!/usr/bin/env python3
"""Build a reviewer queue from real retrieval rerank rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _is_verified(row: Dict[str, Any]) -> bool:
    value = row.get("human_verified")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _review_row(base: Dict[str, Any]) -> Dict[str, Any]:
    candidates = base.get("candidates") if isinstance(base.get("candidates"), list) else []
    return {
        "query": base.get("query", ""),
        "expected_top_id": base.get("expected_top_id", ""),
        "human_verified": bool(base.get("human_verified", False)),
        "reviewer_id": str(base.get("reviewer_id", "") or ""),
        "reviewed_at": base.get("reviewed_at"),
        "review_notes": str(base.get("review_notes", "") or ""),
        "label_source": base.get("label_source", ""),
        "label_confidence": base.get("label_confidence"),
        "label_margin": base.get("label_margin"),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrieval review queue from real-label dataset")
    parser.add_argument(
        "--dataset-jsonl",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "eval_retrieval_rerank_real_v1.jsonl"),
        help="Input real retrieval dataset",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(Path(__file__).resolve().parent / "results" / "retrieval_review_queue_latest.jsonl"),
        help="Output review queue JSONL (unverified rows)",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "retrieval_review_coverage_latest.json"),
        help="Output coverage summary JSON",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    rows = _read_jsonl(dataset_path)
    verified = [_review_row(r) for r in rows if _is_verified(r)]
    unverified = [_review_row(r) for r in rows if not _is_verified(r)]

    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in unverified:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    verified_count = len(verified)
    coverage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "total_rows": total,
        "verified_rows": verified_count,
        "unverified_rows": len(unverified),
        "verified_ratio": round((verified_count / total), 4) if total else 0.0,
    }

    output_json.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total rows: {total}")
    print(f"Verified rows: {verified_count}")
    print(f"Unverified rows: {len(unverified)}")
    print(f"Verified ratio: {coverage['verified_ratio']:.2%}")
    print(f"Queue written to: {output_jsonl}")
    print(f"Coverage written to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
