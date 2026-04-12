# Quick Reference: Testing the 9 Features

## All-in-One Testing Commands

### Start Everything
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: MongoDB
mongod

# Terminal 3: Backend
cd backend
python -m uvicorn main:app --reload

# Terminal 4: Run tests
cd backend
python tests/test_summarization_features.py
```

---

## Feature Overview & Quick Tests

| # | Feature | Quick Test | Expected Result |
|---|---------|-----------|-----------------|
| 1 | Auto-finalize (idle 15 min) | Wait 15 min or set `INACTIVITY_THRESHOLD_MINUTES=1` then wait | Session status → "closed" with `summary_method="auto"` |
| 2 | Rolling summaries (every 10 msgs) | Send 20+ messages then check | `rolling_summaries` array has 2+ entries |
| 3 | force_regenerate param | `/finalize` → `/finalize?force_regenerate=true` | 2nd call faster (cache), 3rd call slower (regen) |
| 4 | Structured metadata | Send pharma discussion then finalize | `topics`, `objections`, `action_items` populated |
| 5 | Redis caching (24h) | `redis-cli KEYS "summary:*"` | Cache keys exist with `TTL ≈ 86400` |
| 6 | Retry logic (3x exponential backoff) | Set invalid `GROQ_API_KEY` then finalize | Logs show "attempt 1/3", "attempt 2/3", etc. |
| 7 | Fallback preview | Simulate API failure (Feature 6) | Fallback summary returned instead of crash |
| 8 | Audit trail | Manual finalize then check DB | `summary_method="manual"`, `summary_triggered_by=email` |
| 9 | Chunking (>100 msgs) | Send 150+ messages then finalize | Completes without timeout, summary quality good |

---

## MongoDB Verification Queries

Connect with MongoDB Compass or `mongosh`:

```bash
use alia

# Check all new fields exist
db.conversations.findOne({status: "closed"})

# Audit trail: Manual finalization
db.conversations.find({summary_method: "manual"}).pretty()

# Audit trail: Auto finalization
db.conversations.find({summary_method: "auto"}).pretty()

# Check rolling summaries (20+ messages)
db.conversations.findOne({rolling_summaries: {$exists: true, $ne: []}})

# Check metadata extraction
db.conversations.findOne({
  $or: [
    {topics: {$ne: []}},
    {objections: {$ne: []}},
    {action_items: {$ne: []}}
  ]
})

# Count sessions by method
db.conversations.aggregate([
  {$group: {_id: "$summary_method", count: {$sum: 1}}}
])
```

---

## Redis Verification Queries

```bash
# Check cache keys exist
redis-cli KEYS "summary:*"
#  → Should return cache keys like: summary:SESSION_ID:HASH

# Check specific cache TTL
redis-cli TTL "summary:*_key_*"
# → Should return ~86400 (24 hours)

# Monitor real-time cache activity
redis-cli MONITOR

# Clear all cache (for testing)
redis-cli FLUSHDB
```

---

## API Quick Commands

### Login / Signup
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"pass"}'
```

### Send Message
```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"","content":"Tell me about CardioGuard","mode":"physician_portal"}'
```

### Finalize (Manual)
```bash
curl -X POST http://localhost:8000/chat/sessions/SESSION_ID/finalize \
  -H "Authorization: Bearer TOKEN"
```

### Finalize (Force Regenerate)
```bash
curl -X POST "http://localhost:8000/chat/sessions/SESSION_ID/finalize?force_regenerate=true" \
  -H "Authorization: Bearer TOKEN"
```

### Get Session Details
```bash
curl -X GET http://localhost:8000/chat/sessions/SESSION_ID \
  -H "Authorization: Bearer TOKEN"
```

### List Sessions
```bash
curl -X GET http://localhost:8000/chat/sessions \
  -H "Authorization: Bearer TOKEN"
```

---

## Log Messages to Watch For

### Feature 1: Auto-finalization
```
Finalizing idle session: 65ab123...
Summary generated: auto_finalize_idle_sessions
Session closed and summarized
```

### Feature 2-3: Caching
```
Cache hit for summary:65ab123:abc123
Returning cached summary
Cache set with 24h TTL
```

### Feature 6: Retry Logic
```
API call failed (attempt 1/3), retrying in 1s
API call failed (attempt 2/3), retrying in 2s
API call failed (attempt 3/3): Fallback preview generated
```

### Feature 9: Chunking
```
Conversation has 150 messages, using chunked approach
Chunking into 10-message batches
Summarizing chunk 1/15...
Meta-summarizing 15 chunk summaries...
```

---

## Test Results Checklist

### Basic Flow (All should pass)
- [ ] Can login/signup
- [ ] Can send messages
- [ ] Session ID is generated
- [ ] Can finalize session

### Feature 1: Auto-Finalization
- [ ] Set `INACTIVITY_THRESHOLD_MINUTES=1` in .env
- [ ] Create session
- [ ] Wait 1-2 minutes
- [ ] Check: `status = "closed"` and `summary_method = "auto"`

### Feature 2: Rolling Summaries
- [ ] Send 20+ messages
- [ ] Finalize
- [ ] Check: `rolling_summaries` array has ≥2 entries
- [ ] Each entry has `message_count`, `summary`, `generated_at`

### Feature 3: Force Regenerate
- [ ] Finalize → T1 response time
- [ ] Finalize again → T2 (fast, from cache)
- [ ] Finalize with `?force_regenerate=true` → T3 (slower, regenerated)
- [ ] Verify: T2 < T1, T3 ≈ T1

### Feature 4: Metadata
- [ ] Check for populated `topics` array
- [ ] Check for populated `objections` array
- [ ] Check for populated `action_items` array

### Feature 5: Redis Cache
- [ ] Run `redis-cli KEYS "summary:*"`
- [ ] Should find 1+ cache keys
- [ ] Run `redis-cli TTL key_name`
- [ ] Should show ~86400 seconds (±60)

### Feature 6: Retry Logic
- [ ] Set `GROQ_API_KEY=invalid`
- [ ] Restart backend
- [ ] Finalize session
- [ ] Check logs for "attempt 1/3", "attempt 2/3" messages
- [ ] Finalization should complete (with fallback)

### Feature 7: Fallback Preview
- [ ] Same test as Feature 6
- [ ] Summary should contain "Started:" and "Recent exchange:"
- [ ] Should be readable without LLM

### Feature 8: Audit Trail
- [ ] Manual finalize: `summary_method="manual"`, `summary_triggered_by=user@email`
- [ ] Auto finalize: `summary_method="auto"`, `summary_triggered_by="system"`

### Feature 9: Chunking
- [ ] Send 150+ messages (may take a few minutes)
- [ ] Finalize and time it
- [ ] Should complete in <30 seconds
- [ ] Check logs for chunking messages

---

## File Locations

| Component | Location | Lines | Purpose |
|-----------|----------|-------|---------|
| Core Logic | `/backend/utils/summary.py` | ~330 | Summarization with retry, caching, chunking |
| Background Task | `/backend/utils/background_tasks.py` | ~60 | Auto-finalize idle sessions |
| API Routes | `/backend/routes/chat.py` | ~250 | Endpoints with audit tracking |
| Models | `/backend/models/conversation.py` | ~50 | Schema with new fields |
| Test Suite | `/backend/tests/test_summarization_features.py` | ~600 | Full test automation |
| Curl Ref | `/backend/tests/curl_commands.sh` | ~250 | Manual curl commands |
| Guide | `/TESTING_GUIDE.md` | ~450 | Detailed testing guide |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Redis connection refused | Run `redis-server`, check Redis running on port 6379 |
| MongoDB connection failed | Run `mongod`, check MongoDB running on port 27017 |
| Groq API key error | Check `.env` for valid `GROQ_API_KEY` |
| Tests timeout | Increase timeout, or test fewer messages (Feature 9) |
| Retry logic not visible | Check logs with `--log-level debug` |
| Cache not working | Verify Redis is running and accessible |
| Fallback not triggered | Intentionally break Groq API (Feature 6 test) |

---

## Success Criteria

**All 9 features are working if:**

✅ Auto-summarization: Sessions close after inactivity  
✅ Rolling summaries: Multiple summaries at 10, 20, 30... message intervals  
✅ Force regenerate: Cache bypassed with `?force_regenerate=true`  
✅ Metadata: Topics, objections, action_items populated  
✅ Redis Cache: Summary cached with 24-hour TTL  
✅ Retry Logic: Logs show exponential backoff attempts  
✅ Fallback: Session completes with preview on API failure  
✅ Audit Trail: `summary_method` and `summary_triggered_by` tracked  
✅ Chunking: 150+ message session completes without timeout  

---

## Performance Benchmarks (Expected)

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Send message | 2-5 sec | Including LLM response |
| Finalize (cached) | <100ms | From Redis |
| Finalize (new) | 5-8 sec | LLM API call + caching |
| Finalize (150 msgs) | 10-20 sec | Chunked processing |
| Retry attempt | 1s → 2s → 4s | Exponential backoff |

---

## Next Steps (Optional Enhancements)

- [ ] Auto-finalize on token threshold (500+ tokens)
- [ ] Sentiment analysis extraction
- [ ] Batch summarization optimization
- [ ] Summary similarity detection (consolidate duplicates)
- [ ] Search by topics/objections/actions
- [ ] Export summaries as PDF/CSV
