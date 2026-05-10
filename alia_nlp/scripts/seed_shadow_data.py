"""
Seed MongoDB with realistic NLP conversation events for shadow monitoring.

Creates two batches:
  A) 18 messages processed by the CURRENT pipeline (intent stored = what pipeline returns now)
     → shadow monitoring will show 0% divergence on these (pipeline is self-consistent)
  B)  7 messages where we store a DIFFERENT intent (simulating a previous pipeline version)
     → shadow monitoring shows ~28% divergence → quality gate FAILS

This tells the demo story: "We updated the pipeline. Shadow monitoring detected 24% of
stored intents no longer match. We investigate, confirm the new pipeline is more accurate,
reset the baseline."

Usage:
    python -m alia_nlp.scripts.seed_shadow_data
    python -m alia_nlp.scripts.seed_shadow_data --mongodb-url mongodb://localhost:27017/alia
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Load .env before any alia_nlp imports
_ENV = Path(__file__).resolve().parents[2] / "backend" / ".env"
if _ENV.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV)
    except ImportError:
        for line in _ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Test messages ─────────────────────────────────────────────────────────────
# Batch A: pipeline should classify these consistently → no divergence
BATCH_A = [
    # medrep_training
    {"text": "Hello doctor, good morning! I'd like to take 2 minutes of your time if that's okay.", "mode": "medrep_training"},
    {"text": "Can we do a flash visit simulation? I want to practice the opening.", "mode": "medrep_training"},
    {"text": "The physician says it's too expensive. How do I handle that objection?", "mode": "medrep_training"},
    {"text": "What's my current competency level? Am I ready for Junior?", "mode": "medrep_training"},
    {"text": "I want to practice the discovery phase. Can you be a skeptical cardiologist?", "mode": "medrep_training"},
    {"text": "How do I structure a standard visit for Cardivex with a general practitioner?", "mode": "medrep_training"},
    {"text": "What are the clinical benefits of Cardivex I should highlight in argumentation?", "mode": "medrep_training"},
    {"text": "The doctor says she always prescribes generic amlodipine. How do I respond?", "mode": "medrep_training"},
    {"text": "I need to log my visit notes and plan the follow-up for next Tuesday.", "mode": "medrep_training"},
    {"text": "What's the recommended dose of Cardivex for hypertension?", "mode": "medrep_training"},
    # physician_portal
    {"text": "What is the mechanism of action of Cardivex?", "mode": "physician_portal"},
    {"text": "Can Cardivex be used in patients with mild renal impairment without dose adjustment?", "mode": "physician_portal"},
    {"text": "What are the main side effects of amlodipine-based therapies?", "mode": "physician_portal"},
    {"text": "Is Cardivex contraindicated in pregnant patients?", "mode": "physician_portal"},
    {"text": "Good morning.", "mode": "physician_portal"},
    {"text": "What are the approved indications for Cardivex?", "mode": "physician_portal"},
    {"text": "Are there any known drug interactions between Cardivex and statins?", "mode": "physician_portal"},
    {"text": "What's the starting dose for elderly patients?", "mode": "physician_portal"},
]

# Batch B: we'll store an INTENTIONALLY WRONG intent (simulating old pipeline version)
# real_intent = what current pipeline actually returns (used for shadow re-run)
# stored_intent = what we PUT in the DB (old/wrong classification)
BATCH_B = [
    {
        "text": "Can we do a role-play where you're a resistant cardiologist?",
        "mode": "medrep_training",
        "stored_intent": "product_information_request",  # wrong — old pipeline misclassified
    },
    {
        "text": "I want to work on my closing technique.",
        "mode": "medrep_training",
        "stored_intent": "other",  # old pipeline gave up, new one classifies correctly
    },
    {
        "text": "The doctor raised a QT prolongation concern. Help me respond.",
        "mode": "medrep_training",
        "stored_intent": "dosage_question",  # wrong classification
    },
    {
        "text": "What evidence do I have for Cardivex superiority over generic?",
        "mode": "medrep_training",
        "stored_intent": "other",
    },
    {
        "text": "Does Cardivex interact with beta-blockers?",
        "mode": "physician_portal",
        "stored_intent": "product_information_request",  # should be safety_question
    },
    {
        "text": "Is it safe to prescribe Cardivex to a breastfeeding patient?",
        "mode": "physician_portal",
        "stored_intent": "dosage_question",  # wrong — should be safety_question
    },
    {
        "text": "My patient is on warfarin. Any interaction concerns with Cardivex?",
        "mode": "physician_portal",
        "stored_intent": "product_information_request",  # should be safety_question
    },
]


# ── DB insertion ──────────────────────────────────────────────────────────────

def _make_nlp_event(text: str, intent: str, mode: str, minutes_ago: int) -> dict:
    at = datetime.utcnow() - timedelta(minutes=minutes_ago)
    return {
        "at":               at,
        "mode":             mode,
        "message":          text,
        "intent":           intent,
        "confidence":       0.80,
        "secondary_tags":   [],
        "entities":         [],
        "entity_map":       {},
        "topics":           [],
        "objections":       [],
        "action_items":     [],
        "safety_flags":     [],
        "rewritten_query":  text[:80],
        "taxonomy_version": "v1",
        "explainability":   {"intent_source": "seeded"},
        "affect":           {
            "rep_confidence":    "medium",
            "frustration_signal": False,
            "stress_signal":      False,
            "engagement_level":  "engaged",
            "query_urgency":     "routine",
            "affect_source":     "rules",
        },
    }


def _make_conversation(user_email: str, mode: str, nlp_event: dict, minutes_ago: int) -> dict:
    at = datetime.utcnow() - timedelta(minutes=minutes_ago)
    return {
        "user_email":  user_email,
        "mode":        mode,
        "status":      "closed",
        "messages": [
            {"role": "user",      "content": nlp_event["message"], "at": at},
            {"role": "assistant", "content": "Thank you for your query.",  "at": at + timedelta(seconds=3)},
        ],
        "nlp_events":           [nlp_event],
        "summary":              None,
        "rolling_summaries":    [],
        "topics":               [],
        "objections":           [],
        "action_items":         [],
        "competency_level":     None,
        "evaluation_score":     None,
        "evaluation_dimensions": {},
        "evaluation_strengths": [],
        "evaluation_gaps":      [],
        "evaluation_notes":     [],
        "evaluation_completed_at": None,
        "created_at":           at,
        "updated_at":           at,
    }


async def seed(mongodb_url: str) -> None:
    from motor.motor_asyncio import AsyncIOMotorClient
    from alia_nlp.src.pipeline.online import analyze_message_nlp

    client = AsyncIOMotorClient(mongodb_url)
    db     = client.get_default_database()
    logger.info("Connected to MongoDB: %s", mongodb_url)

    inserted = 0
    seed_email = "shadow.seed@alia-demo.internal"

    # ── Batch A: authentic pipeline runs ──────────────────────────────────────
    logger.info("Seeding Batch A (%d messages — authentic pipeline intents)…", len(BATCH_A))
    for i, item in enumerate(BATCH_A):
        result  = analyze_message_nlp(item["text"], history=[], mode=item["mode"])
        intent  = result.get("intent", "other")
        event   = _make_nlp_event(item["text"], intent, item["mode"], minutes_ago=60 + i * 3)
        conv    = _make_conversation(seed_email, item["mode"], event, minutes_ago=60 + i * 3)
        await db.conversations.insert_one(conv)
        logger.info("  [A%02d] %-55s → %s", i + 1, item["text"][:55], intent)
        inserted += 1

    # ── Batch B: intentionally mis-stored intents ─────────────────────────────
    logger.info("Seeding Batch B (%d messages — legacy misclassified intents)…", len(BATCH_B))
    for i, item in enumerate(BATCH_B):
        # Store the WRONG intent to simulate an older pipeline version
        event = _make_nlp_event(item["text"], item["stored_intent"], item["mode"], minutes_ago=120 + i * 5)
        conv  = _make_conversation(seed_email, item["mode"], event, minutes_ago=120 + i * 5)
        await db.conversations.insert_one(conv)
        # Compute what the current pipeline actually says for context
        current = analyze_message_nlp(item["text"], history=[], mode=item["mode"])
        logger.info(
            "  [B%02d] %-45s | stored=%-35s | current=%s",
            i + 1, item["text"][:45], item["stored_intent"], current.get("intent"),
        )
        inserted += 1

    client.close()
    logger.info("Done. Inserted %d conversation documents.", inserted)
    logger.info(
        "Expected shadow divergence: ~%d/%d = %.0f%%  (quality gate threshold = 8%%)",
        len(BATCH_B), inserted, len(BATCH_B) / inserted * 100,
    )


async def run_shadow_report(mongodb_url: str) -> None:
    """Run the shadow monitoring function and print the report."""
    from motor.motor_asyncio import AsyncIOMotorClient
    from backend.utils.background_tasks import auto_generate_shadow_monitoring_snapshot

    client = AsyncIOMotorClient(mongodb_url)
    db     = client.get_default_database()

    logger.info("\nRunning shadow monitoring snapshot…")
    result = await auto_generate_shadow_monitoring_snapshot(db, days=7, limit=500)
    client.close()

    print("\n" + "=" * 60)
    print("SHADOW MONITORING REPORT")
    print("=" * 60)
    print(f"  Total events analysed : {result.get('rows', 0)}")
    print(f"  Divergence rate       : {result.get('divergence_rate', 0):.1%}")
    print(f"  Quality gate          : {result.get('quality_gate', 'unknown').upper()}")
    print(f"  Report saved to       : {result.get('latest_json', '')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed MongoDB with shadow monitoring demo data")
    parser.add_argument(
        "--mongodb-url",
        default=os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia"),
        help="MongoDB connection URL (default: $MONGODB_URL or mongodb://localhost:27017/alia)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip seeding, only run the shadow monitoring report",
    )
    args = parser.parse_args()

    async def main():
        if not args.report_only:
            await seed(args.mongodb_url)
        await run_shadow_report(args.mongodb_url)

    asyncio.run(main())
