#!/usr/bin/env python3
"""Check human review coverage for real retrieval rerank labels."""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check coverage of human-verified retrieval labels")
    parser.add_argument(
        "--dataset-jsonl",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "eval_retrieval_rerank_real_v1.jsonl"),
        help="Input real retrieval dataset",
    )
    parser.add_argument("--min-verified-ratio", type=float, default=0.20)
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "retrieval_review_gate_latest.json"),
        help="Output JSON gate artifact",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    rows = _read_jsonl(dataset_path)
    total = len(rows)
    verified = sum(1 for row in rows if _is_verified(row))
    ratio = (verified / total) if total else 0.0

    gate_pass = ratio >= args.min_verified_ratio

    artifact = {
        "dataset": str(dataset_path),
        "result": {
            "total_rows": total,
            "verified_rows": verified,
            "verified_ratio": round(ratio, 4),
        },
        "thresholds": {"min_verified_ratio": args.min_verified_ratio},
        "quality_gate": "pass" if gate_pass else "fail",
        "failed_checks": [] if gate_pass else ["verified_ratio"],
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rows: {total}")
    print(f"Verified rows: {verified}")
    print(f"Verified ratio: {ratio:.2%}")
    print(f"Artifact saved to: {output_path}")

    if not gate_pass:
        print("QUALITY GATE: FAIL")
        print("- verified_ratio")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
