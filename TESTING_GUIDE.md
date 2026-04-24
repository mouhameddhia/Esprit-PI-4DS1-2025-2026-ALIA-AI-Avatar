# Testing Guide: 9 Implemented Features

Quick reference for testing each of the 9 implemented summarization features.

## Quick Start

### 1. Prerequisites
```bash
# Ensure Redis is running
redis-server

# Ensure backend is running
cd backend
python -m uvicorn main:app --reload

# Ensure MongoDB is running
mongod
```

### 2. Run Full Test Suite
```bash
cd backend
python tests/test_summarization_features.py
```

---

## Feature-by-Feature Testing

### ✅ Feature 1: Auto-Summarization (Inactivity)

**What it does:** Automatically finalizes and summarizes conversations idle for 15+ minutes.

**How to test:**
```bash
# Option A: Quick test with reduced threshold
# In backend/.env, add or modify:
INACTIVITY_THRESHOLD_MINUTES=1

# Restart backend and create a session
# Wait 1-2 minutes
# Check MongoDB - session should be "closed" with summary

# Option B: Manual trigger
python -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from backend.utils.background_tasks import auto_finalize_idle_sessions

async def test():
    client = AsyncIOMotorClient('mongodb://localhost:27017/alia')
    db = client.alia
    count = await auto_finalize_idle_sessions(db)
    print(f'Finalized {count} sessions')

asyncio.run(test())
"
```

**Verify in MongoDB:**
```javascript
db.conversations.findOne({status: "closed", summary_method: "auto"})
```

---

### ✅ Feature 2: Incremental Rolling Summaries

**What it does:** Generates summaries every 10 messages as conversation grows.

**How to test:**
```bash
# Send 20+ messages in a session
python tests/test_summarization_features.py  # Test 2 only

# Check rolling_summaries array
```

**Verify in MongoDB:**
```javascript
db.conversations.findOne({_id: ObjectId("...")}).rolling_summaries
// Should show multiple entries with message_count: 10, 20, 30, etc.
```

---

### ✅ Feature 3: Regeneration with force_regenerate

**What it does:** `force_regenerate=true` parameter regenerates summary bypassing cache.

**How to test:**
```bash
# Create session with 5 messages
# Finalize (generates + caches)
curl -X POST http://localhost:8000/chat/sessions/{session_id}/finalize \
  -H "Authorization: Bearer {token}"

# Finalize again (should use cache - same timestamp)
curl -X POST http://localhost:8000/chat/sessions/{session_id}/finalize \
  -H "Authorization: Bearer {token}"

# Force regenerate (new summary generated)
curl -X POST http://localhost:8000/chat/sessions/{session_id}/finalize?force_regenerate=true \
  -H "Authorization: Bearer {token}"
```

**Expected:** 
- Same summary text in calls 1 & 2
- Different/updated summary in call 3
- `summary_created_at` timestamp updates in call 3

---

### ✅ Feature 4: Structured Metadata

**What it does:** Extracts topics, objections, and action_items.

**How to test:**
```bash
# Create conversation discussing:
# - Products mentioned
# - Objections/concerns raised
# - Follow-up actions agreed

# Send messages:
1. "Tell me about CardioGuard effectiveness in hypertension"
2. "What about side effects compared to ACE inhibitors?"
3. "Do you have any concerns about renal function?"
4. "We should schedule a follow-up with your team"

# Finalize and check metadata
```

**Verify in MongoDB:**
```javascript
db.conversations.findOne({_id: ObjectId("...")})
// Look for:
// topics: ["CardioGuard", "Hypertension", "ACE inhibitors"]
// objections: ["Side effects", "Renal function concerns"]
// action_items: ["Schedule follow-up"]
```

---

### ✅ Feature 5: Redis Caching (24-hour TTL)

**How to test:**
```bash
# After finalizing a conversation

# Check Redis for cache key
redis-cli KEYS "summary:*"

# Get cache entry
redis-cli GET "summary:{session_id}:{hash}"

# Check TTL
redis-cli TTL "summary:{session_id}:{hash}"
# Should return ~86400 (24 hours in seconds)

# Simulate cache hit
# Finalize same session without force_regenerate
# Should return instantly from cache
```

**Verify:**
```bash
# Monitor Redis memory growth
redis-cli INFO memory

# All cached summaries should have 24-hour expiration
# Old entries auto-expire and free memory
```

---

### ✅ Feature 6: Retry Logic with Exponential Backoff

**How to test:**
```bash
# Simulate API failure
1. Edit .env and set invalid Groq key:
   GROQ_API_KEY=invalid_key_xxx

2. Restart backend

3. Create and finalize a session

4. Check logs for retry messages:
   "API call failed (attempt 1/3), retrying in 1s"
   "API call failed (attempt 2/3), retrying in 2s"  
   "API call failed (attempt 3/3)"
   "Using fallback preview"

5. Session should still complete with fallback summary

6. Restore valid GROQ_API_KEY and restart
```

**When to use retry logic:**
- Network timeouts
- Rate limiting (429 errors)
- Temporary service outages
- Groq API availability issues

---

### ✅ Feature 7: Fallback Preview on API Failure

**How to test:**
```bash
# Same as Feature 6 - simulate Groq API failure

# Check fallback summary format:
# "Started: [first message preview]...
#  Recent exchange:
#  You: [your last message]...
#  ALIA: [response]..."
```

**Verify:** Summary should be human-readable even without LLM.

---

### ✅ Feature 8: Audit Trail (summary_method, summary_triggered_by)

**How to test:**

**Manual finalization:**
```bash
# Call /finalize endpoint manually
curl -X POST http://localhost:8000/chat/sessions/{session_id}/finalize \
  -H "Authorization: Bearer {user_token}"

# Check database
db.conversations.findOne({_id: ObjectId("...")})
// Should show:
// summary_method: "manual"
// summary_triggered_by: "user@email.com"
```

**Auto finalization (background task):**
```bash
# Wait 15+ minutes of inactivity (or reduce INACTIVITY_THRESHOLD_MINUTES)

# Check database
db.conversations.findOne({_id: ObjectId("..."), summary_method: "auto"})
// Should show:
// summary_method: "auto"
// summary_triggered_by: "system"
```

**Audit trail query:**
```javascript
// Find all manually finalized sessions by user
db.conversations.find({
  summary_method: "manual",
  summary_triggered_by: "test@example.com"
})

// Find all auto-finalized sessions
db.conversations.find({
  summary_method: "auto",
  summary_triggered_by: "system"
})

// Find when each was summarized
db.conversations.find({}, {
  _id: 1,
  summary_created_at: 1,
  summary_method: 1,
  summary_triggered_by: 1
})
```

---

### ✅ Feature 9: Performance (Chunking for Large Conversations)

**How to test with large conversation:**

```python
# backend/tests/test_large_conversation.py
import requests
import time

API_BASE = "http://localhost:8000"
TOKEN = "your_auth_token"

session_id = None

# Send 150 messages
for i in range(150):
    msg = f"Message {i}: Sample pharmaceutical discussion content about products, efficacy, side effects, and clinical data."
    
    resp = requests.post(
        f"{API_BASE}/chat/message",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "session_id": session_id,
            "content": msg[:100],
            "mode": "physician_portal"
        }
    )
    
    if resp.status_code == 200:
        session_id = resp.json()["session_id"]
        print(f"Message {i+1} sent")

# Time the finalization
start = time.time()
resp = requests.post(
    f"{API_BASE}/chat/sessions/{session_id}/finalize",
    headers={"Authorization": f"Bearer {TOKEN}"}
)
elapsed = time.time() - start

print(f"Finalization took {elapsed:.2f} seconds")
print(f"Summary: {resp.json()['summary'][:200]}...")
```

**Expected behavior:**
- No timeout errors (Python default timeout is 30 seconds)
- Summary quality remains good
- Completes in <30 seconds
- Rolling summaries generated at 10, 20, 30... message intervals

**Verify chunking logic:**
```bash
# Check logs while processing 150+ message session
# Should see:
# - "Chunking conversation into chunks of 10 messages"
# - Multiple API calls for each chunk
# - Meta-summarization of chunk summaries
```

---

## Database Verification Checklist

Connect to MongoDB and verify these fields exist:

```javascript
// Open MongoDB compass or mongo shell
use alia
db.conversations.findOne()
```

Should contain:
- ✅ `summary` (string)
- ✅ `summary_created_at` (datetime)
- ✅ `summary_method` (string: "auto" or "manual")
- ✅ `summary_triggered_by` (string: email or "system")
- ✅ `rolling_summaries` (array of objects)
- ✅ `topics` (array)
- ✅ `objections` (array)
- ✅ `action_items` (array)

---

## Logging & Debugging

### View backend logs:
```bash
# Terminal 1: Run backend with logging
cd backend
python -m uvicorn main:app --reload --log-level debug
```

### Watch for key log messages:

**Auto-finalization:**
```
Finalizing idle session: {session_id}
Summary method: auto, triggered by: system
```

**Retry logic:**
```
API call failed (attempt 1/3), retrying in 1s
API call failed (attempt 2/3), retrying in 2s
```

**Chunking:**
```
Chunking conversation: 150 messages into chunks of 10
Generating chunk summaries...
Meta-summarizing chunk summaries...
```

**Cache hits:**
```
Cache hit for {session_id}: returning cached summary
```

---

## Troubleshooting

### Test fails: "Redis connection refused"
```bash
# Start Redis server
redis-server
# Or in background: redis-server &
```

### Test fails: "MongoDB connection failed"
```bash
# Start MongoDB
mongod
# Or ensure mongod service is running
```

### Test fails: "Groq API error"
```bash
# Verify GROQ_API_KEY is set
echo $GROQ_API_KEY

# Check API key in .env
cat backend/.env | grep GROQ
```

### Timeout on large conversation test
```bash
# Increase command timeout or use requests timeout param
# In test script:
requests.post(..., timeout=120)  # 2 minute timeout
```

---

## Summary Checklist

- [ ] Test 1: Auto-finalization triggers after inactivity
- [ ] Test 2: Rolling summaries generated every 10 messages
- [ ] Test 3: force_regenerate parameter works
- [ ] Test 4: Structured metadata extracted (topics, objections, actions)
- [ ] Test 5: Redis cache stores summaries with 24hr TTL
- [ ] Test 6: Retry logic logs attempts with exponential backoff
- [ ] Test 7: Fallback preview generated on API failure
- [ ] Test 8: Audit trail tracks manual vs auto finalization
- [ ] Test 9: Large conversation (150+ msgs) chunks successfully
- [ ] Database: All new fields present and populated correctly

---

## Additional Resources

- Backend source: `backend/utils/summary.py`
- Background tasks: `backend/utils/background_tasks.py`
- API routes: `backend/routes/chat.py`
- Models: `backend/models/conversation.py`
- Test suite: `backend/tests/test_summarization_features.py`
