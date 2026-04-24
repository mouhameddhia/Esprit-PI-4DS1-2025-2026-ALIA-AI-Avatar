#!/usr/bin/env python3
"""Export shadow-monitoring JSONL from stored conversation NLP events.

This script treats the persisted event intent as the primary prediction and
re-runs the current NLP analyzer as the shadow prediction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.utils.nlp import analyze_message_nlp


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return None


def _safe_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


async def export_shadow_logs(days: int, limit: int, output_path: Path) -> Dict[str, Any]:
    backend_dir = REPO_ROOT / "backend"
    load_dotenv(dotenv_path=backend_dir / ".env")

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia")
    client = AsyncIOMotorClient(mongodb_url)
    db = client.alia

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = {
        "nlp_events": {"$exists": True, "$ne": []},
        "updated_at": {"$gte": cutoff.replace(tzinfo=None)},
    }

    projection = {"_id": 1, "mode": 1, "nlp_events": 1, "updated_at": 1}
    cursor = db.conversations.find(query, projection=projection).sort("updated_at", -1)

    rows: List[Dict[str, Any]] = []
    processed_events = 0
    skipped_events = 0
    disagreements = 0

    async for doc in cursor:
        session_id = str(doc.get("_id"))
        session_mode = _safe_str(doc.get("mode")) or "physician_portal"
        nlp_events = doc.get("nlp_events") or []

        for event in nlp_events:
            if len(rows) >= limit:
                break

            processed_events += 1
            text = _safe_str(event.get("message"))
            primary_intent = _safe_str(event.get("intent"))
            mode = _safe_str(event.get("mode")) or session_mode

            if not text or not primary_intent:
                skipped_events += 1
                continue

            shadow = analyze_message_nlp(user_text=text, history=[], mode=mode)
            shadow_intent = _safe_str(shadow.get("intent")) or "other"

            if shadow_intent != primary_intent:
                disagreements += 1

            rows.append(
                {
                    "session_id": session_id,
                    "event_at": _iso(event.get("at")),
                    "mode": mode,
                    "text": text,
                    "primary_intent": primary_intent,
                    "primary_confidence": event.get("confidence"),
                    "shadow_intent": shadow_intent,
                    "shadow_confidence": shadow.get("confidence"),
                }
            )

        if len(rows) >= limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    client.close()

    divergence_rate = (disagreements / len(rows)) if rows else 0.0
    return {
        "rows": len(rows),
        "processed_events": processed_events,
        "skipped_events": skipped_events,
        "disagreements": disagreements,
        "divergence_rate": round(divergence_rate, 4),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export shadow-monitoring JSONL from Mongo conversation NLP events")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum number of rows to export")
    parser.add_argument(
        "--output-jsonl",
        default=str(REPO_ROOT / "NLP" / "evaluation" / "results" / "shadow_logs_latest.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    output_path = Path(args.output_jsonl).expanduser().resolve()
    stats = asyncio.run(export_shadow_logs(days=args.days, limit=args.limit, output_path=output_path))

    print(f"Rows exported: {stats['rows']}")
    print(f"Events processed: {stats['processed_events']}")
    print(f"Events skipped: {stats['skipped_events']}")
    print(f"Intent disagreements: {stats['disagreements']}")
    print(f"Estimated divergence: {stats['divergence_rate']:.2%}")
    print(f"Output written to: {stats['output']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
