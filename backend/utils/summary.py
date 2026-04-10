import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import redis
from fastapi import HTTPException, status


def _get_redis_client() -> redis.Redis:
    """Get or create Redis client for caching."""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except redis.ConnectionError:
        # If Redis is not available, return None (caching will be skipped)
        return None


def _generate_cache_key(session_id: str, conversation_hash: str) -> str:
    """Generate a cache key for a summary."""
    return f"summary:{session_id}:{conversation_hash}"


def _hash_transcript(transcript: str) -> str:
    """Generate a hash of the transcript for cache validation."""
    return hashlib.md5(transcript.encode()).hexdigest()


def _chunk_messages(messages: List[Dict[str, Any]], chunk_size: int = 10) -> List[str]:
    """
    Split messages into chunks and create transcripts.
    For large conversations (>100 messages), this reduces API calls and context.
    """
    chunk_transcripts = []
    
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i : i + chunk_size]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)
        chunk_transcripts.append(transcript)
    
    return chunk_transcripts


def _groq_client():
    """Get Groq client."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq SDK not installed. Run: pip install groq",
        ) from exc
    
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY not configured",
        )
    return Groq(api_key=key)


def _summary_completion(transcript: str) -> str:
    """Generate summary from transcript using Groq."""
    client = _groq_client()
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize pharmaceutical training / HCP chat sessions for internal records. "
                    "Output a short headline (one line) then 3–6 bullet points. "
                    "Focus on topics discussed, products mentioned, objections, and any follow-ups. "
                    "Be factual; do not invent details not present in the transcript."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this conversation:\n\n{transcript}",
            },
        ],
        temperature=0.4,
        max_tokens=512,
    )
    choice = completion.choices[0].message
    if not choice or not choice.content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Empty summary from language model",
        )
    return choice.content.strip()


def _extract_metadata(transcript: str) -> Dict[str, List[str]]:
    """
    Extract structured metadata (topics, objections, action items) from transcript.
    Uses LLM to identify key information.
    """
    client = _groq_client()
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract metadata from a pharmaceutical conversation. "
                    "Return a JSON object with exactly three arrays: "
                    '"topics" (products/conditions discussed), '
                    '"objections" (concerns or pushback raised), '
                    '"action_items" (follow-ups, agreements, or next steps). '
                    "Each array should contain 1-5 concise strings. "
                    "If none exist, use empty arrays. Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"Extract metadata from this conversation:\n\n{transcript}",
            },
        ],
        temperature=0.3,
        max_tokens=256,
    )
    
    choice = completion.choices[0].message
    if not choice or not choice.content:
        return {"topics": [], "objections": [], "action_items": []}
    
    try:
        metadata = json.loads(choice.content.strip())
        return {
            "topics": metadata.get("topics", []),
            "objections": metadata.get("objections", []),
            "action_items": metadata.get("action_items", []),
        }
    except json.JSONDecodeError:
        return {"topics": [], "objections": [], "action_items": []}


async def generate_summary_with_caching(
    session_id: str,
    messages: List[Dict[str, Any]],
    force_regenerate: bool = False,
) -> tuple[str, Dict[str, List[str]]]:
    """
    Generate summary with Redis caching and pagination for large conversations.
    
    - Caches summaries for 24 hours
    - Chunks conversations >100 messages before summarization
    - Extracts structured metadata (topics, objections, action_items)
    
    Returns:
        (summary_text, metadata_dict)
    """
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    if not messages:
        return "No messages in this conversation.", {"topics": [], "objections": [], "action_items": []}
    
    # Build full transcript
    full_transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    transcript_hash = _hash_transcript(full_transcript)
    cache_key = _generate_cache_key(session_id, transcript_hash)
    
    # Try to get from cache
    redis_client = _get_redis_client()
    if redis_client and not force_regenerate:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                cached_data = json.loads(cached)
                return cached_data["summary"], cached_data["metadata"]
        except Exception:
            pass  # Cache miss or error, continue with generation
    
    # Generate summary
    if len(messages) > 100:
        # For large conversations, use chunked approach
        chunk_transcripts = _chunk_messages(messages, chunk_size=10)
        chunk_summaries = [_summary_completion(chunk) for chunk in chunk_transcripts]
        
        # Now summarize the summaries
        meta_transcript = "\n---\n".join(chunk_summaries)
        summary = _summary_completion(meta_transcript)
    else:
        # For smaller conversations, summarize directly
        summary = _summary_completion(full_transcript)
    
    # Extract metadata from the full transcript
    metadata = _extract_metadata(full_transcript)
    
    # Cache the result
    if redis_client:
        try:
            cache_data = {
                "summary": summary,
                "metadata": metadata,
                "generated_at": datetime.utcnow().isoformat(),
            }
            redis_client.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps(cache_data),
            )
        except Exception:
            pass  # Cache write error, continue without caching
    
    return summary, metadata
