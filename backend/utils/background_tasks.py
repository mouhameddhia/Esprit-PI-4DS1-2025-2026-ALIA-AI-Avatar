import os
from datetime import datetime, timedelta
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from .summary import generate_summary_with_caching


INACTIVITY_THRESHOLD_MINUTES = int(os.getenv("INACTIVITY_THRESHOLD_MINUTES", "15"))


async def auto_finalize_idle_sessions(db: AsyncIOMotorDatabase) -> int:
    """
    Automatically finalize (close + summarize) conversations that have been idle
    for longer than INACTIVITY_THRESHOLD_MINUTES.
    
    Returns:
        Number of sessions finalized
    """
    cutoff_time = datetime.utcnow() - timedelta(minutes=INACTIVITY_THRESHOLD_MINUTES)
    
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
