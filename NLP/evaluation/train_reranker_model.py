#!/usr/bin/env python3
"""Train lightweight reranker by fitting blend weights on labeled rerank data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.reranker import rerank_candidates


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


def _rank_metrics(rows: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, float]:
    hit_at_1 = 0
    reciprocal_rank_sum = 0.0
    total = 0

    for row in rows:
        query = str(row.get("query", "")).strip()
        expected_id = str(row.get("expected_top_id", "")).strip()
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        if not query or not expected_id or not candidates:
            continue

        total += 1
        ranked = rerank_candidates(query=query, candidates=candidates, weights=weights)

        rank = None
        for idx, cand in enumerate(ranked, start=1):
            if str(cand.get("id", "")).strip() == expected_id:
                rank = idx
                break

        if rank == 1:
            hit_at_1 += 1
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

    if total == 0:
        return {"count": 0, "hit_at_1": 0.0, "mrr": 0.0}

    return {
        "count": total,
        "hit_at_1": round(hit_at_1 / total, 4),
        "mrr": round(reciprocal_rank_sum / total, 4),
    }


def _candidate_weight_sets() -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    steps = [i / 20.0 for i in range(1, 20)]
    for base in steps:
        for overlap in steps:
            jaccard = 1.0 - base - overlap
            if jaccard <= 0:
                continue
            out.append(
                {
                    "base_score": round(base, 4),
                    "overlap": round(overlap, 4),
                    "jaccard": round(jaccard, 4),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train reranker blend weights")
    parser.add_argument("--dataset-jsonl", required=True, help="Training split JSONL")
    parser.add_argument("--val-jsonl", default="", help="Optional validation split JSONL")
    parser.add_argument("--output-json", default="NLP/evaluation/results/ci_reranker_model_v1.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Training dataset not found: {dataset_path}")
        return 1

    train_rows = _read_jsonl(dataset_path)
    val_rows: List[Dict[str, Any]] = []
    val_path = None
    if args.val_jsonl:
        val_path = Path(args.val_jsonl).expanduser().resolve()
        if not val_path.exists():
            print(f"Validation dataset not found: {val_path}")
            return 1
        val_rows = _read_jsonl(val_path)

    best_weights = None
    best_metrics = None
    for weights in _candidate_weight_sets():
        metrics = _rank_metrics(train_rows, weights=weights)
        score = (metrics["mrr"], metrics["hit_at_1"])
        if best_metrics is None or score > (best_metrics["mrr"], best_metrics["hit_at_1"]):
            best_weights = weights
            best_metrics = metrics

    if best_weights is None or best_metrics is None:
        print("No valid training rows found")
        return 1

    val_metrics = _rank_metrics(val_rows, weights=best_weights) if val_rows else {"count": 0, "hit_at_1": None, "mrr": None}

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "reranker_blend_v1",
        "train_dataset": str(dataset_path),
        "val_dataset": str(val_path) if val_path else None,
        "weights": best_weights,
        "train_metrics": best_metrics,
        "val_metrics": val_metrics,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Train rows: {best_metrics['count']}")
    print(f"Train hit@1: {best_metrics['hit_at_1']:.2%}")
    print(f"Train MRR: {best_metrics['mrr']:.4f}")
    if isinstance(val_metrics.get("mrr"), float):
        print(f"Val rows: {val_metrics['count']}")
        print(f"Val hit@1: {val_metrics['hit_at_1']:.2%}")
        print(f"Val MRR: {val_metrics['mrr']:.4f}")
    print(f"Artifact saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())