#!/usr/bin/env python3
"""Export real user messages from conversation history as JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


async def export_user_messages(days: int, limit: int, output_path: Path) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / "backend" / ".env")

    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia"))
    db = client.alia

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = {"updated_at": {"$gte": cutoff.replace(tzinfo=None)}}
    projection = {"_id": 1, "updated_at": 1, "messages": 1, "mode": 1}

    rows: List[Dict[str, Any]] = []
    cursor = db.conversations.find(query, projection=projection).sort("updated_at", -1)

    async for doc in cursor:
        sid = str(doc.get("_id"))
        mode = doc.get("mode") if isinstance(doc.get("mode"), str) else "physician_portal"

        for msg in doc.get("messages") or []:
            if len(rows) >= limit:
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            at = msg.get("at")
            at_iso = at.astimezone(timezone.utc).isoformat() if isinstance(at, datetime) else None

            rows.append(
                {
                    "session_id": sid,
                    "event_at": at_iso,
                    "mode": mode,
                    "text": content.strip(),
                    "source": "conversation_messages",
                }
            )

        if len(rows) >= limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    client.close()

    return {"rows": len(rows), "output": str(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export user message logs from conversation history")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--output-jsonl",
        default=str(Path(__file__).resolve().parents[2] / "NLP" / "evaluation" / "results" / "user_message_logs_latest.jsonl"),
    )
    args = parser.parse_args()

    output_path = Path(args.output_jsonl).expanduser().resolve()
    stats = asyncio.run(export_user_messages(days=max(1, args.days), limit=max(1, args.limit), output_path=output_path))

    print(f"Rows exported: {stats['rows']}")
    print(f"Output written to: {stats['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
