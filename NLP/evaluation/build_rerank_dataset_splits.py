#!/usr/bin/env python3
"""Build deterministic train/val/test splits for reranker datasets."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
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


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic reranker train/val/test splits")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", default="NLP/evaluation/results")
    parser.add_argument("--prefix", default="ci_rerank_split_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        print("Invalid ratios: require train_ratio > 0, val_ratio > 0, and train+val < 1")
        return 1

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    rows = _read_jsonl(dataset_path)
    valid_rows = [
        row
        for row in rows
        if isinstance(row.get("query"), str)
        and str(row.get("query", "")).strip()
        and isinstance(row.get("expected_top_id"), str)
        and str(row.get("expected_top_id", "")).strip()
        and isinstance(row.get("candidates"), list)
        and len(row.get("candidates", [])) > 0
    ]

    rnd = random.Random(args.seed)
    shuffled = list(valid_rows)
    rnd.shuffle(shuffled)

    n = len(shuffled)
    train_n = max(1, int(round(n * args.train_ratio)))
    val_n = max(1, int(round(n * args.val_ratio)))
    if train_n + val_n >= n:
        val_n = max(1, min(val_n, n - train_n - 1))
    test_n = max(1, n - train_n - val_n)

    train_rows = shuffled[:train_n]
    val_rows = shuffled[train_n : train_n + val_n]
    test_rows = shuffled[train_n + val_n : train_n + val_n + test_n]

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{args.prefix}_train.jsonl"
    val_path = out_dir / f"{args.prefix}_val.jsonl"
    test_path = out_dir / f"{args.prefix}_test.jsonl"
    meta_path = out_dir / f"{args.prefix}_metadata.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(test_path, test_rows)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_path),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 4),
        },
        "counts": {
            "source_rows": len(rows),
            "valid_rows": len(valid_rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())