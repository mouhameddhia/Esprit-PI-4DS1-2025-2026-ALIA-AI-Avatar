#!/usr/bin/env python3
"""Evaluate retrieval reranking quality on a labeled candidate dataset."""

from __future__ import annotations

import argparse
import json
import sys
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


def _load_model_artifact(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _rank_metrics(rows: List[Dict[str, Any]], use_rerank: bool, reranker_model: Dict[str, Any] | None = None) -> Dict[str, float]:
    hit_at_1 = 0
    reciprocal_rank_sum = 0.0

    for row in rows:
        query = str(row.get("query", "")).strip()
        expected_id = str(row.get("expected_top_id", "")).strip()
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []

        if not query or not expected_id or not candidates:
            continue

        ranked = rerank_candidates(query, candidates, model=reranker_model) if use_rerank else sorted(
            [dict(item) for item in candidates],
            key=lambda item: float(item.get("score", 0.0) or 0.0),
            reverse=True,
        )

        rank = None
        for idx, cand in enumerate(ranked, start=1):
            if str(cand.get("id", "")).strip() == expected_id:
                rank = idx
                break

        if rank == 1:
            hit_at_1 += 1
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

    total = len(rows)
    if total == 0:
        return {"count": 0, "hit_at_1": 0.0, "mrr": 0.0}

    return {
        "count": total,
        "hit_at_1": round(hit_at_1 / total, 4),
        "mrr": round(reciprocal_rank_sum / total, 4),
    }


def evaluate_dataset(path: Path, reranker_model: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    baseline = _rank_metrics(rows=rows, use_rerank=False)
    reranked = _rank_metrics(rows=rows, use_rerank=True, reranker_model=reranker_model)

    return {
        "count": baseline.get("count", 0),
        "baseline": baseline,
        "reranked": reranked,
        "delta": {
            "hit_at_1": round(reranked.get("hit_at_1", 0.0) - baseline.get("hit_at_1", 0.0), 4),
            "mrr": round(reranked.get("mrr", 0.0) - baseline.get("mrr", 0.0), 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval reranking quality")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "eval_retrieval_rerank_v3.jsonl"),
        help="Path to retrieval rerank benchmark dataset",
    )
    parser.add_argument("--min-rerank-hit-at-1", type=float, default=0.70)
    parser.add_argument("--min-rerank-mrr", type=float, default=0.85)
    parser.add_argument(
        "--max-hit-at-1-regression",
        type=float,
        default=0.01,
        help="Maximum allowed drop vs baseline hit@1",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "eval_retrieval_rerank_latest.json"),
        help="Path to write evaluation artifact",
    )
    parser.add_argument("--model-json", default="", help="Optional trained reranker model artifact")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    reranker_model = None
    model_path = None
    if args.model_json:
        model_path = Path(args.model_json).expanduser().resolve()
        if not model_path.exists():
            print(f"Model artifact not found: {model_path}")
            return 1
        reranker_model = _load_model_artifact(model_path)

    result = evaluate_dataset(dataset_path, reranker_model=reranker_model)

    print(f"Samples: {result['count']}")
    print(f"Baseline hit@1: {result['baseline']['hit_at_1']:.2%}")
    print(f"Baseline MRR: {result['baseline']['mrr']:.4f}")
    print(f"Reranked hit@1: {result['reranked']['hit_at_1']:.2%}")
    print(f"Reranked MRR: {result['reranked']['mrr']:.4f}")

    hit_regression = result["baseline"]["hit_at_1"] - result["reranked"]["hit_at_1"]
    checks = {
        "rerank_hit_at_1": result["reranked"]["hit_at_1"] >= args.min_rerank_hit_at_1,
        "rerank_mrr": result["reranked"]["mrr"] >= args.min_rerank_mrr,
        "hit_at_1_regression": hit_regression <= args.max_hit_at_1_regression,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    artifact = {
        "dataset": str(dataset_path),
        "model_json": str(model_path) if model_path else None,
        "result": result,
        "thresholds": {
            "min_rerank_hit_at_1": args.min_rerank_hit_at_1,
            "min_rerank_mrr": args.min_rerank_mrr,
            "max_hit_at_1_regression": args.max_hit_at_1_regression,
        },
        "quality_gate": "fail" if failed_checks else "pass",
        "failed_checks": failed_checks,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artifact saved to: {output_path}")

    if failed_checks:
        print("QUALITY GATE: FAIL")
        for name in failed_checks:
            print(f"- {name}")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
