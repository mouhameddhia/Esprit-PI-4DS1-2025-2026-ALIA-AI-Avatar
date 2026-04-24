#!/usr/bin/env python3
"""Build clarification KPI report from stored conversation NLP events."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _safe_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


async def build_report(days: int, output_path: Path) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root / "backend"
    load_dotenv(dotenv_path=backend_dir / ".env")

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia")
    client = AsyncIOMotorClient(mongodb_url)
    db = client.alia

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = {
        "nlp_events": {"$exists": True, "$ne": []},
        "updated_at": {"$gte": cutoff.replace(tzinfo=None)},
    }
    projection = {"_id": 1, "mode": 1, "nlp_events": 1}

    totals = {
        "events": 0,
        "clarification_requested": 0,
        "clarification_follow_up": 0,
        "clarification_resolved": 0,
        "clarification_unresolved": 0,
    }

    by_mode: Dict[str, Dict[str, int]] = {}
    confidence_sum = 0.0
    confidence_count = 0

    cursor = db.conversations.find(query, projection=projection)
    async for doc in cursor:
        mode = _safe_str(doc.get("mode")) or "physician_portal"
        if mode not in by_mode:
            by_mode[mode] = {
                "events": 0,
                "clarification_requested": 0,
                "clarification_follow_up": 0,
                "clarification_resolved": 0,
            }

        for event in doc.get("nlp_events") or []:
            totals["events"] += 1
            by_mode[mode]["events"] += 1

            confidence = _safe_float(event.get("confidence"))
            if confidence > 0.0:
                confidence_sum += confidence
                confidence_count += 1

            requested = _safe_bool(event.get("clarification_requested"))
            follow_up = _safe_bool(event.get("clarification_follow_up"))
            resolved = _safe_bool(event.get("clarification_resolved"))

            if requested:
                totals["clarification_requested"] += 1
                by_mode[mode]["clarification_requested"] += 1
            if follow_up:
                totals["clarification_follow_up"] += 1
                by_mode[mode]["clarification_follow_up"] += 1
            if resolved:
                totals["clarification_resolved"] += 1
                by_mode[mode]["clarification_resolved"] += 1

    totals["clarification_unresolved"] = max(
        0,
        totals["clarification_follow_up"] - totals["clarification_resolved"],
    )

    clarification_rate = (
        totals["clarification_requested"] / totals["events"]
        if totals["events"]
        else 0.0
    )
    follow_up_resolution_rate = (
        totals["clarification_resolved"] / totals["clarification_follow_up"]
        if totals["clarification_follow_up"]
        else 0.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "totals": totals,
        "rates": {
            "clarification_rate": round(clarification_rate, 4),
            "follow_up_resolution_rate": round(follow_up_resolution_rate, 4),
        },
        "avg_confidence": round((confidence_sum / confidence_count), 4) if confidence_count else None,
        "by_mode": by_mode,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    client.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Report clarification KPIs from conversation NLP events")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parents[2] / "NLP" / "evaluation" / "results" / "clarification_kpi_latest.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    report = asyncio.run(build_report(days=max(1, args.days), output_path=output_path))

    print(f"Events analyzed: {report['totals']['events']}")
    print(f"Clarification requested: {report['totals']['clarification_requested']}")
    print(f"Clarification rate: {report['rates']['clarification_rate']:.2%}")
    print(f"Follow-up resolution rate: {report['rates']['follow_up_resolution_rate']:.2%}")
    print(f"Output written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())