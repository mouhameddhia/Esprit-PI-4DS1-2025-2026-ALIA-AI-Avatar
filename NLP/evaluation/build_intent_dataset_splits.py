#!/usr/bin/env python3
"""Build deterministic stratified train/val/test splits for intent datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def _jsonl_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))

    train_n = max(1, min(train_n, n - 2))
    val_n = max(1, min(val_n, n - train_n - 1))
    test_n = n - train_n - val_n
    if test_n < 1:
        test_n = 1
        if val_n > 1:
            val_n -= 1
        else:
            train_n -= 1
    return train_n, val_n, test_n


def _dist(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        label = row.get("expected_intent")
        if isinstance(label, str) and label.strip():
            c[label.strip()] += 1
    return dict(sorted(c.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic stratified intent train/val/test splits")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", default="NLP/evaluation/results")
    parser.add_argument("--prefix", default="ci_intent_split_v1")
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
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for row in rows:
        label = row.get("expected_intent")
        text = row.get("text")
        if not isinstance(label, str) or not label.strip() or not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        grouped[label.strip()].append(row)

    rnd = random.Random(args.seed)
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for label in sorted(grouped.keys()):
        bucket = list(grouped[label])
        rnd.shuffle(bucket)
        n = len(bucket)
        train_n, val_n, _ = _split_indices(n, args.train_ratio, args.val_ratio)
        train_rows.extend(bucket[:train_n])
        val_rows.extend(bucket[train_n : train_n + val_n])
        test_rows.extend(bucket[train_n + val_n :])

    rnd.shuffle(train_rows)
    rnd.shuffle(val_rows)
    rnd.shuffle(test_rows)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{args.prefix}_train.jsonl"
    val_path = out_dir / f"{args.prefix}_val.jsonl"
    test_path = out_dir / f"{args.prefix}_test.jsonl"
    meta_path = out_dir / f"{args.prefix}_metadata.json"

    for path, split_rows in [(train_path, train_rows), (val_path, val_rows), (test_path, test_rows)]:
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
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
            "skipped_rows": skipped,
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "distributions": {
            "train": _dist(train_rows),
            "val": _dist(val_rows),
            "test": _dist(test_rows),
        },
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "train_sha256": _jsonl_hash(train_path),
            "val_sha256": _jsonl_hash(val_path),
            "test_sha256": _jsonl_hash(test_path),
        },
    }

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
