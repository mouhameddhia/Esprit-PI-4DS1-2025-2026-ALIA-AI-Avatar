#!/usr/bin/env python3
"""Build retrieval rerank benchmark rows from real production query logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.embeddings import EmbeddingEncoder
from backend.vector_db import VectorDBClient


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
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _lexical_recall(query: str, text: str) -> float:
    q = set(_tokenize(query))
    d = set(_tokenize(text))
    if not q or not d:
        return 0.0
    return len(q & d) / max(1, len(q))


def _domain_keyword_boost(query: str, text: str) -> float:
    q = (query or "").lower()
    t = (text or "").lower()
    boosts: List[Tuple[List[str], float]] = [
        (["contra", "contraind"], 0.15),
        (["dose", "dosage", "dosing"], 0.15),
        (["adverse", "side effect", "reaction"], 0.12),
        (["pregnan", "lactation", "breast"], 0.12),
        (["crm", "follow up", "follow-up"], 0.10),
        (["role play", "simulation", "skeptical"], 0.10),
        (["mechanism", "moa"], 0.10),
        (["trial", "evidence", "endpoint"], 0.10),
    ]

    score = 0.0
    for kws, delta in boosts:
        if any(kw in q for kw in kws) and any(kw in t for kw in kws):
            score += delta
    return score


async def _fetch_candidates(
    query: str,
    db,
    vector_client: VectorDBClient,
    encoder: EmbeddingEncoder,
    top_k_each: int,
) -> List[Dict[str, Any]]:
    emb = encoder.encode(query)
    if not emb:
        return []

    product_results = await vector_client.search(
        query_vector=emb,
        top_k=top_k_each,
        include_metadata=True,
        filter_dict={"type": {"$eq": "product"}},
    )
    document_results = await vector_client.search(
        query_vector=emb,
        top_k=top_k_each,
        include_metadata=True,
        filter_dict={"type": {"$eq": "document"}},
    )

    out: List[Dict[str, Any]] = []

    for result in product_results:
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        pid = metadata.get("product_id")
        if not isinstance(pid, str) or not pid:
            continue
        try:
            product = await db.products.find_one({"_id": ObjectId(pid)})
        except Exception:
            product = None
        if not product:
            continue

        text = " ".join(
            part
            for part in [
                product.get("name", ""),
                product.get("description", ""),
                " ".join(product.get("indications", [])),
                product.get("category", ""),
            ]
            if isinstance(part, str) and part.strip()
        ).strip()

        out.append(
            {
                "id": f"product_{pid}",
                "score": float(result.get("score", 0.0) or 0.0),
                "text": text,
            }
        )

    for result in document_results:
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        did = metadata.get("document_id")
        if not isinstance(did, str) or not did:
            continue
        try:
            doc = await db.knowledge_documents.find_one({"_id": ObjectId(did)})
        except Exception:
            doc = None
        if not doc:
            continue

        chunk = str(doc.get("chunk_text", "") or "").strip()
        if len(chunk) > 600:
            chunk = chunk[:600].rstrip() + "..."
        text = " ".join(
            part
            for part in [
                str(doc.get("source_name", "") or "").strip(),
                str(doc.get("title", "") or "").strip(),
                str(doc.get("section_title", "") or "").strip(),
                chunk,
            ]
            if part
        ).strip()

        out.append(
            {
                "id": f"document_{did}",
                "score": float(result.get("score", 0.0) or 0.0),
                "text": text,
            }
        )

    out.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    # De-duplicate by id while preserving order.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in out:
        rid = row.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(row)
    return deduped


def _pick_expected(
    query: str,
    candidates: List[Dict[str, Any]],
    min_label_score: float,
    min_margin: float,
) -> Tuple[str | None, float, float]:
    scored: List[Tuple[str, float]] = []
    for row in candidates:
        text = str(row.get("text", "") or "")
        lexical = _lexical_recall(query, text)
        domain = _domain_keyword_boost(query, text)
        score = lexical + domain
        scored.append((str(row.get("id", "")), score))

    if not scored:
        return None, 0.0, 0.0

    scored.sort(key=lambda item: item[1], reverse=True)
    best_id, best = scored[0]
    second = scored[1][1] if len(scored) > 1 else 0.0
    margin = best - second

    if best < min_label_score or margin < min_margin:
        return None, best, margin
    return best_id, best, margin


async def build_real_dataset(
    source_jsonl: Path,
    output_jsonl: Path,
    target_rows: int,
    min_query_len: int,
    top_k_each: int,
    dedupe_queries: bool,
    min_label_score: float,
    min_margin: float,
) -> Dict[str, Any]:
    backend_dir = REPO_ROOT / "backend"
    load_dotenv(dotenv_path=backend_dir / ".env")

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia")
    client = AsyncIOMotorClient(mongodb_url)
    db = client.alia

    vector_client = VectorDBClient()
    encoder = EmbeddingEncoder()

    if not vector_client.is_ready():
        raise RuntimeError("Vector database not ready")
    if not encoder.is_ready():
        raise RuntimeError("Embedding encoder not ready")

    rows = _read_jsonl(source_jsonl)

    # Keep unique real queries in recency order.
    seen = set()
    unique_queries: List[str] = []
    for row in rows:
        text = str(row.get("text", "") or "").strip()
        if len(text) < min_query_len:
            continue
        key = text.lower()
        if dedupe_queries and key in seen:
            continue
        seen.add(key)
        unique_queries.append(text)

    out_rows: List[Dict[str, Any]] = []
    considered = 0
    skipped_low_conf = 0

    for query in unique_queries:
        if len(out_rows) >= target_rows:
            break

        considered += 1
        candidates = await _fetch_candidates(
            query=query,
            db=db,
            vector_client=vector_client,
            encoder=encoder,
            top_k_each=top_k_each,
        )
        if len(candidates) < 3:
            continue

        # Keep top 6 baseline candidates for ranking eval.
        candidates = candidates[:6]
        expected_id, confidence, margin = _pick_expected(
            query=query,
            candidates=candidates,
            min_label_score=min_label_score,
            min_margin=min_margin,
        )
        if not expected_id:
            skipped_low_conf += 1
            continue

        out_rows.append(
            {
                "query": query,
                "expected_top_id": expected_id,
                "candidates": candidates,
                "label_source": "auto_real_log_heuristic",
                "label_confidence": round(confidence, 4),
                "label_margin": round(margin, 4),
                "human_verified": False,
                "reviewer_id": "",
                "reviewed_at": None,
                "review_notes": "",
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    client.close()

    return {
        "source_rows": len(rows),
        "unique_queries": len(unique_queries),
        "considered_queries": considered,
        "produced_rows": len(out_rows),
        "skipped_low_conf": skipped_low_conf,
        "output": str(output_jsonl),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrieval rerank dataset rows from real query logs")
    parser.add_argument(
        "--source-jsonl",
        default=str(REPO_ROOT / "NLP" / "evaluation" / "results" / "shadow_logs_latest.jsonl"),
        help="Input JSONL containing real query logs",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(REPO_ROOT / "NLP" / "datasets" / "eval_retrieval_rerank_real_v1.jsonl"),
        help="Output JSONL for real rerank benchmark rows",
    )
    parser.add_argument("--target-rows", type=int, default=40)
    parser.add_argument("--min-query-len", type=int, default=8)
    parser.add_argument("--top-k-each", type=int, default=6)
    parser.add_argument("--min-label-score", type=float, default=0.12)
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Allow repeated real queries from logs to increase row count in low-traffic datasets",
    )
    args = parser.parse_args()

    source = Path(args.source_jsonl).expanduser().resolve()
    if not source.exists():
        print(f"Source file not found: {source}")
        return 1

    result = asyncio.run(
        build_real_dataset(
            source_jsonl=source,
            output_jsonl=Path(args.output_jsonl).expanduser().resolve(),
            target_rows=max(1, args.target_rows),
            min_query_len=max(1, args.min_query_len),
            top_k_each=max(3, args.top_k_each),
            dedupe_queries=not args.no_dedupe,
            min_label_score=max(0.0, args.min_label_score),
            min_margin=max(0.0, args.min_margin),
        )
    )

    print(f"Source rows: {result['source_rows']}")
    print(f"Unique queries: {result['unique_queries']}")
    print(f"Considered queries: {result['considered_queries']}")
    print(f"Produced rows: {result['produced_rows']}")
    print(f"Skipped low confidence: {result['skipped_low_conf']}")
    print(f"Output written to: {result['output']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
