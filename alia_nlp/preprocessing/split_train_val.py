"""Stratified train/val split by mode and intent."""

import json
import pathlib
import random
from collections import defaultdict
from typing import Dict, List

VAL_RATIO = 0.15


def split(dataset_path: pathlib.Path, seed: int = 42) -> None:
    rows = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(seed)

    buckets: Dict[str, List] = defaultdict(list)
    for row in rows:
        key = f"{row.get('mode','?')}::{row.get('expected_intent','?')}"
        buckets[key].append(row)

    train, val = [], []
    for bucket in buckets.values():
        random.shuffle(bucket)
        n_val = max(1, int(len(bucket) * VAL_RATIO))
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    out_dir = dataset_path.parent.parent / "processed"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (out_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val), encoding="utf-8")
    print(f"Train: {len(train)}  Val: {len(val)}")


if __name__ == "__main__":
    import sys
    p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        pathlib.Path(__file__).parents[1] / "data" / "raw" / "eval_intent_safety_v6.jsonl"
    split(p)
