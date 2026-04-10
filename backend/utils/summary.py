import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import redis
from fastapi import HTTPException, status


# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 10  # seconds


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


def _call_with_retry(api_call_func, max_retries=MAX_RETRIES):
    """
    Wrapper to call API with exponential backoff retry logic.
    
    Args:
        api_call_func: Callable that makes the API call
        max_retries: Number of retries before giving up
        
    Returns:
        API response or raises exception
    """
    delay = INITIAL_RETRY_DELAY
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return api_call_func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)  # Exponential backoff
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
    
    raise last_error or HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Failed to reach language model after retries",
    )


def _summary_completion(transcript: str) -> str:
    """Generate summary from transcript using Groq with retry logic."""
    def api_call():
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
    
    return _call_with_retry(api_call)


def _extract_metadata(transcript: str) -> Dict[str, List[str]]:
    """
    Extract structured metadata (topics, objections, action items) from transcript.
    Uses LLM to identify key information with retry logic.
    """
    def api_call():
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
            return None
        return choice.content.strip()
    
    try:
        content = _call_with_retry(api_call)
        metadata = json.loads(content)
        return {
            "topics": metadata.get("topics", []),
            "objections": metadata.get("objections", []),
            "action_items": metadata.get("action_items", []),
        }
    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Error extracting metadata: {e}")
        return {"topics": [], "objections": [], "action_items": []}


def _generate_fallback_preview(messages: List[Dict[str, Any]]) -> str:
    """
    Generate a simple preview when API summarization fails.
    Uses first message and last 5 messages to create a basic summary.
    """
    if not messages:
        return "No messages in this conversation."
    
    preview_parts = []
    
    # Get first message
    first_msg = messages[0]
    preview_parts.append(f"Started: {first_msg['content'][:100]}...")
    
    # Get last 5 messages
    recent_msgs = messages[-5:]
    if len(recent_msgs) > 1:
        preview_parts.append("\nRecent exchange:")
        for msg in recent_msgs[-3:]:  # Last 3 of the last 5
            role = "You" if msg['role'] == 'user' else "ALIA"
            preview_parts.append(f"{role}: {msg['content'][:60]}...")
    
    return "\n".join(preview_parts)


async def generate_summary_with_caching(
    session_id: str,
    messages: List[Dict[str, Any]],
    force_regenerate: bool = False,
) -> tuple[str, Dict[str, List[str]], List[Dict]]:
    """
    Generate summary with Redis caching, error handling, and incremental summaries.
    
    Features:
    - Caches summaries for 24 hours with retry logic
    - Chunks conversations >100 messages before summarization
    - Extracts structured metadata (topics, objections, action_items)
    - Generates incremental summaries every 10 messages (rolling summaries)
    - Falls back to preview if API fails
    
    Returns:
        (summary_text, metadata_dict, rolling_summaries_list)
    """
    
    if not messages:
        return "No messages in this conversation.", {"topics": [], "objections": [], "action_items": []}, []
    
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
                return (
                    cached_data["summary"],
                    cached_data["metadata"],
                    cached_data.get("rolling_summaries", []),
                )
        except Exception:
            pass  # Cache miss or error, continue with generation
    
    # Generate main summary with error handling
    try:
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
    except Exception as e:
        print(f"Summary generation failed: {e}, using fallback preview")
        summary = _generate_fallback_preview(messages)
    
    # Extract metadata from the full transcript with error handling
    try:
        metadata = _extract_metadata(full_transcript)
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        metadata = {"topics": [], "objections": [], "action_items": []}
    
    # Generate incremental/rolling summaries every 10 messages
    rolling_summaries = []
    try:
        for i in range(10, len(messages), 10):
            chunk = messages[:i]
            chunk_transcript = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)
            
            try:
                rolling_summary_text = _summary_completion(chunk_transcript)
                rolling_summaries.append({
                    "summary": rolling_summary_text,
                    "generated_at": datetime.utcnow().isoformat(),
                    "message_count": i,
                })
            except Exception as e:
                print(f"Rolling summary generation at message {i} failed: {e}")
                # Continue with the next batch
    except Exception as e:
        print(f"Rolling summary generation error: {e}")
    
    # Cache the result
    if redis_client:
        try:
            cache_data = {
                "summary": summary,
                "metadata": metadata,
                "rolling_summaries": rolling_summaries,
                "generated_at": datetime.utcnow().isoformat(),
            }
            redis_client.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps(cache_data),
            )
        except Exception:
            pass  # Cache write error, continue without caching
    
    return summary, metadata, rolling_summaries
