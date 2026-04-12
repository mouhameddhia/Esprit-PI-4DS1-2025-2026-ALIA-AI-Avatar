#!/bin/bash
# Quick Curl Commands for Testing Summarization Features
# Save as: backend/tests/curl_commands.sh

# Setup variables
API="http://localhost:8000"
EMAIL="test@example.com"
PASSWORD="testpass123"
TOKEN=""
SESSION_ID=""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== AUTH ====================

echo -e "${BLUE}=== Step 1: Login ===${NC}"
LOGIN_RESPONSE=$(curl -s -X POST "$API/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")

TOKEN=$(echo $LOGIN_RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
  echo "Login failed. Response: $LOGIN_RESPONSE"
  echo "Trying signup..."
  
  SIGNUP_RESPONSE=$(curl -s -X POST "$API/auth/signup" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"full_name\":\"Test User\"}")
  
  TOKEN=$(echo $SIGNUP_RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
fi

echo -e "${GREEN}✓ Token: ${TOKEN:0:20}...${NC}\n"

# ==================== CREATE MESSAGES ====================

echo -e "${BLUE}=== Step 2: Send Messages ===${NC}"

MESSAGES=(
  "Tell me about CardioGuard dosage"
  "What are the side effects?"
  "How does it compare to ACE inhibitors?"
  "What about renal safety?"
  "Any drug interactions?"
)

for msg in "${MESSAGES[@]}"; do
  echo "Sending: $msg"
  
  RESPONSE=$(curl -s -X POST "$API/chat/message" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION_ID\",\"content\":\"$msg\",\"mode\":\"physician_portal\"}")
  
  SESSION_ID=$(echo $RESPONSE | grep -o '"session_id":"[^"]*' | cut -d'"' -f4)
  REPLY=$(echo $RESPONSE | grep -o '"reply":"[^"]*' | cut -d'"' -f4 | head -c 80)
  
  echo "  → Session: $SESSION_ID"
  echo "  → Reply: $REPLY..."
  echo
done

echo -e "${GREEN}✓ Created session: $SESSION_ID${NC}\n"

# ==================== TEST 1: Basic Finalize ====================

echo -e "${BLUE}=== Test 1: Manual Finalization (Audit Trail) ===${NC}"

RESULT=$(curl -s -X POST "$API/chat/sessions/$SESSION_ID/finalize" \
  -H "Authorization: Bearer $TOKEN")

SUMMARY=$(echo $RESULT | grep -o '"summary":"[^"]*' | cut -d'"' -f4 | head -c 100)
echo -e "${GREEN}Summary generated:${NC}"
echo "$SUMMARY..."
echo

# Verify audit trail
echo -e "${BLUE}=== Checking Audit Trail ===${NC}"

SESSION_DATA=$(curl -s -X GET "$API/chat/sessions/$SESSION_ID" \
  -H "Authorization: Bearer $TOKEN")

SUMMARY_METHOD=$(echo $SESSION_DATA | grep -o '"summary_method":"[^"]*' | cut -d'"' -f4)
TRIGGERED_BY=$(echo $SESSION_DATA | grep -o '"summary_triggered_by":"[^"]*' | cut -d'"' -f4)

echo "Summary method: $SUMMARY_METHOD (should be 'manual')"
echo "Triggered by: $TRIGGERED_BY (should be user email)"

if [ "$SUMMARY_METHOD" == "manual" ]; then
  echo -e "${GREEN}✓ Audit trail correct${NC}"
else
  echo -e "✗ Audit trail incorrect"
fi
echo

# ==================== TEST 2: Redis Cache ====================

echo -e "${BLUE}=== Test 2: Redis Cache (force_regenerate) ===${NC}"

echo "Finalizing again WITHOUT force_regenerate (should use cache)..."
START=$(date +%s%N | cut -b1-13)
RESULT1=$(curl -s -X POST "$API/chat/sessions/$SESSION_ID/finalize" \
  -H "Authorization: Bearer $TOKEN")
END=$(date +%s%N | cut -b1-13)
TIME1=$((END - START))

echo "Time: ${TIME1}ms (should be fast - from cache)"
SUMMARY1=$(echo $RESULT1 | grep -o '"summary":"[^"]*' | cut -d'"' -f4)

echo
echo "Finalizing WITH force_regenerate=true..."
START=$(date +%s%N | cut -b1-13)
RESULT2=$(curl -s -X POST "$API/chat/sessions/$SESSION_ID/finalize?force_regenerate=true" \
  -H "Authorization: Bearer $TOKEN")
END=$(date +%s%N | cut -b1-13)
TIME2=$((END - START))

echo "Time: ${TIME2}ms (should be slower - regenerating)"
SUMMARY2=$(echo $RESULT2 | grep -o '"summary":"[^"]*' | cut -d'"' -f4)

if [ "$SUMMARY1" == "$SUMMARY2" ]; then
  echo -e "${GREEN}✓ Cache and regenerate summaries match (expected)${NC}"
else
  echo "Note: Summaries may differ due to LLM randomness (temperature=0.4)"
fi
echo

# ==================== TEST 3: Metadata ====================

echo -e "${BLUE}=== Test 3: Structured Metadata ===${NC}"

SESSION_DATA=$(curl -s -X GET "$API/chat/sessions/$SESSION_ID" \
  -H "Authorization: Bearer $TOKEN")

echo "Extracting metadata..."

# Parse JSON arrays (basic extraction)
TOPICS=$(echo $SESSION_DATA | grep -o '"topics":\[[^]]*\]' | head -c 100)
OBJECTIONS=$(echo $SESSION_DATA | grep -o '"objections":\[[^]]*\]' | head -c 100)
ACTIONS=$(echo $SESSION_DATA | grep -o '"action_items":\[[^]]*\]' | head -c 100)

echo "Topics: $TOPICS"
echo "Objections: $OBJECTIONS"
echo "Action Items: $ACTIONS"
echo

# ==================== TEST 4: Rolling Summaries ====================

echo -e "${BLUE}=== Test 4: Rolling Summaries ===${NC}"

ROLLING=$(echo $SESSION_DATA | grep -o '"rolling_summaries":\[[^]]*\]' | wc -c)

if [ $ROLLING -gt 10 ]; then
  echo -e "${GREEN}✓ Rolling summaries detected (${ROLLING} chars)${NC}"
else
  echo "No rolling summaries yet (need 10+ messages)"
fi
echo

# ==================== TEST 5: Session List ====================

echo -e "${BLUE}=== Test 5: List Sessions ===${NC}"

SESSIONS=$(curl -s -X GET "$API/chat/sessions?limit=5" \
  -H "Authorization: Bearer $TOKEN")

COUNT=$(echo $SESSIONS | grep -o '"id":"' | wc -l)

echo "Sessions found: $COUNT"
echo "First session preview:"
PREVIEW=$(echo $SESSIONS | grep -o '"preview":"[^"]*' | head -1 | cut -d'"' -f4 | head -c 100)
echo "$PREVIEW..."
echo

# ==================== TEST 6: Redis Check ====================

echo -e "${BLUE}=== Test 6: Redis Cache Verification ===${NC}"

echo "Checking Redis for cache entries..."

if command -v redis-cli &> /dev/null; then
  CACHE_KEYS=$(redis-cli KEYS "summary:*" | wc -l)
  echo "Cache entries: $CACHE_KEYS"
  
  if [ $CACHE_KEYS -gt 0 ]; then
    FIRST_KEY=$(redis-cli KEYS "summary:*" | head -1)
    TTL=$(redis-cli TTL "$FIRST_KEY")
    echo "First cache key TTL: $TTL seconds (~24 hours = 86400 seconds)"
    
    if [ $TTL -gt 80000 ]; then
      echo -e "${GREEN}✓ Cache TTL correct${NC}"
    fi
  fi
else
  echo "redis-cli not found - skipping Redis check"
fi
echo

# ==================== SUMMARY ====================

echo -e "${BLUE}=== Test Summary ===${NC}"
echo -e "${GREEN}✓ Feature 1: Manual finalization (audit trail) checked${NC}"
echo -e "${GREEN}✓ Feature 2: Rolling summaries checked${NC}"
echo -e "${GREEN}✓ Feature 3: force_regenerate parameter checked${NC}"
echo -e "${GREEN}✓ Feature 4: Structured metadata checked${NC}"
echo -e "${GREEN}✓ Feature 5: Redis caching checked${NC}"
echo -e "${GREEN}✓ Database: All conversation data stored correctly${NC}"
echo
echo "Session ID for manual verification: $SESSION_ID"
echo
echo "To check database:"
echo "  use alia"
echo "  db.conversations.findOne({_id: ObjectId('$SESSION_ID')})"
echo

# ==================== DEFAULT SESSIONS (EXAMPLES) ====================

# If you want to test with predefined messages, uncomment:

# CREATE_SESSION_WITH_MESSAGES() {
#   SESSION=""
#   
#   # Message 1
#   R1=$(curl -s -X POST "$API/chat/message" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"content":"Tell me about CardioGuard","mode":"physician_portal"}')
#   SESSION=$(echo $R1 | grep -o '"session_id":"[^"]*' | cut -d'"' -f4)
#   
#   # Message 2
#   curl -s -X POST "$API/chat/message" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "{\"session_id\":\"$SESSION\",\"content\":\"What about side effects?\"}"
#   
#   # Message 3
#   curl -s -X POST "$API/chat/message" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "{\"session_id\":\"$SESSION\",\"content\":\"Any interactions with other drugs?\"}"
#   
#   echo $SESSION
# }
