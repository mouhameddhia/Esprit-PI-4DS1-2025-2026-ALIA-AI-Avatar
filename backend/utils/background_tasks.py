import json
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorDatabase

from .. import config
from .summary import generate_summary_with_caching
from NLP.pipeline.nlp import analyze_message_nlp
from NLP.evaluation.shadow_monitoring import build_shadow_report


async def auto_finalize_idle_sessions(db: AsyncIOMotorDatabase) -> int:
    """
    Automatically finalize (close + summarize) conversations that have been idle
    for longer than INACTIVITY_THRESHOLD_MINUTES.
    
    Returns:
        Number of sessions finalized
    """
    cutoff_time = datetime.utcnow() - timedelta(minutes=config.INACTIVITY_THRESHOLD_MINUTES)
    
    # Find all open conversations that haven't been updated within the threshold
    idle_sessions = await db.conversations.find(
        {
            "status": "open",
            "updated_at": {"$lt": cutoff_time},
        }
    ).to_list(None)
    
    finalized_count = 0
    
    for session in idle_sessions:
        try:
            session_id = session["_id"]
            messages = session.get("messages", [])
            
            # Generate summary and metadata
            summary, metadata, rolling_summaries = await generate_summary_with_caching(
                str(session_id),
                messages,
                force_regenerate=False,
            )
            
            now = datetime.utcnow()
            
            # Update the session with audit trail
            await db.conversations.update_one(
                {"_id": session_id},
                {
                    "$set": {
                        "summary": summary,
                        "summary_created_at": now,
                        "summary_method": "auto",
                        "summary_triggered_by": "system",
                        "rolling_summaries": rolling_summaries,
                        "topics": metadata.get("topics", []),
                        "objections": metadata.get("objections", []),
                        "action_items": metadata.get("action_items", []),
                        "status": "closed",
                        "updated_at": now,
                    }
                },
            )
            finalized_count += 1
        except Exception as e:
            # Log the error but continue processing other sessions
            print(f"Error auto-finalizing session {session_id}: {e}")
    
    return finalized_count


async def auto_generate_shadow_monitoring_snapshot(
    db: AsyncIOMotorDatabase,
    days: Optional[int] = None,
    limit: Optional[int] = None,
    max_divergence: Optional[float] = None,
) -> dict:
    """Build daily shadow JSONL + report from conversation NLP events.

    Returns summary stats for scheduler logging.
    """
    days = days if days is not None else config.SHADOW_LOG_LOOKBACK_DAYS
    limit = limit if limit is not None else config.SHADOW_LOG_MAX_ROWS
    max_divergence = max_divergence if max_divergence is not None else config.SHADOW_MAX_DIVERGENCE

    results_dir = Path("NLP") / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.utcnow().strftime("%Y%m%d")
    dated_jsonl = results_dir / f"shadow_logs_{today}.jsonl"
    latest_jsonl = results_dir / "shadow_logs_latest.jsonl"
    dated_json = results_dir / f"shadow_{today}.json"
    latest_json = results_dir / "shadow_latest.json"

    cutoff_time = datetime.utcnow() - timedelta(days=days)
    query = {
        "nlp_events": {"$exists": True, "$ne": []},
        "updated_at": {"$gte": cutoff_time},
    }
    projection = {"_id": 1, "mode": 1, "nlp_events": 1, "updated_at": 1}

    rows = []
    disagreements = 0
    processed_events = 0
    skipped_events = 0

    cursor = db.conversations.find(query, projection=projection).sort("updated_at", -1)
    async for doc in cursor:
        session_id = str(doc.get("_id"))
        session_mode = doc.get("mode") if isinstance(doc.get("mode"), str) else "physician_portal"
        events = doc.get("nlp_events") or []

        for event in events:
            if len(rows) >= limit:
                break
            processed_events += 1

            text = event.get("message") if isinstance(event.get("message"), str) else ""
            text = text.strip()
            primary_intent = event.get("intent") if isinstance(event.get("intent"), str) else ""
            primary_intent = primary_intent.strip()
            mode = event.get("mode") if isinstance(event.get("mode"), str) else session_mode

            if not text or not primary_intent:
                skipped_events += 1
                continue

            shadow = analyze_message_nlp(user_text=text, history=[], mode=mode)
            shadow_intent = shadow.get("intent") if isinstance(shadow.get("intent"), str) else "other"
            if shadow_intent != primary_intent:
                disagreements += 1

            event_at = event.get("at")
            event_at_iso = event_at.isoformat() if isinstance(event_at, datetime) else None

            rows.append(
                {
                    "session_id": session_id,
                    "event_at": event_at_iso,
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

    for path in (dated_jsonl, latest_jsonl):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_shadow_report(rows)
    artifact = {
        "generated_at": datetime.utcnow().isoformat(),
        "thresholds": {"max_divergence": max_divergence},
        "result": report,
        "quality_gate": "pass" if report.get("divergence_rate", 0.0) <= max_divergence else "fail",
        "meta": {
            "processed_events": processed_events,
            "skipped_events": skipped_events,
            "disagreements": disagreements,
            "lookback_days": days,
            "max_rows": limit,
        },
    }

    for path in (dated_json, latest_json):
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "rows": len(rows),
        "divergence_rate": report.get("divergence_rate", 0.0),
        "quality_gate": artifact["quality_gate"],
        "latest_jsonl": str(latest_jsonl),
        "latest_json": str(latest_json),
    }
