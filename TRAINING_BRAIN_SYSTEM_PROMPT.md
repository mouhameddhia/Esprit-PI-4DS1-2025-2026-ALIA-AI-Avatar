# Training Brain System Prompt: Comprehensive Training Flow

## PURPOSE

This prompt documents the complete training flow for the Hybrid Medical Agent. It describes:
1. How training data flows between components (memory, RAG, JSON KB, evaluation)
2. How the Training Brain uses the LLM to make decisions in real time
3. How learning signals accumulate over time across turns
4. How conversation state tracks progress and locks context

---

## 1. TRAINING DATA FLOW ARCHITECTURE

### 1.1 Input Channels (Per Turn)

**User Input:**
```
User Message → Language Detection → Intent Classification → Drug Resolution
```

**Memory Channels:**
- `conversation_memory.json`: Recent user facts (name, role, company), recent message window (6 msgs), summarized older context
- `persistent_memory.json`: Event log (qa, user_fact, preference, instruction), pruned to 10 recent QA entries
- Both loaded at START of request and rebound per user (no stale shared state)

**Knowledge Channels:**
- JSON KB Priority: Product payloads from `rag_knowledge_builder/scripts/incoming_payload_*.json` + `rag_knowledge_base/json/*.json`
- RAG Fallback: Knowledge Retrieval Agent pipeline (dense + sparse + reranking) with XAI explanations
- RAG Cache: High-confidence responses stored in `rag_response_cache.json` (confidence ≥ 0.75, no uncertainty, no fallback triggered)

**Conversation State:**
```json
{
  "current_product": "BACTOL",
  "product_locked": true,
  "conversation_phase": "clinical_exploration",
  "last_topic": "composition",
  "last_user_intent": "composition_probe",
  "current_intent": "medical",
  "user_role": "rep",
  "turn_count": 3,
  "alia_level": 2
}
```

### 1.2 Training Brain Decision Pipeline (Per Turn)

```
USER QUESTION
    ↓
[LANGUAGE DETECT: FR vs EN]
    ↓
[INTENT CLASSIFY: medical | product_discussion | greeting | off_topic]
    ↓
[PRODUCT RESOLVE: detected from question OR locked from state OR from history]
    ↓
[EXTRACT CONVERSATION CONTEXT: topics covered, turn count, context established]
    ↓
[BUILD TRAINING STATE: phase, level, intent, memory snapshot]
    ↓
[QUERY JSON KB: find(drug, topic, question)]
    ↓ (IF JSON FOUND)
    [TRAINING BRAIN LLM: decide_next_action()]
    ↓
[SANITIZE: action ∈ {ask_question, respond, clarify, challenge}, topic ∈ CRITICAL_TOPICS]
    ↓
[FALLBACK GUARDS: repetition check, empty message check, language check]
    ↓
[RETURN: TrainingBrainDecision(action, message, topic)]
```

### 1.3 LLM Prompt to Training Brain

The Training Brain builds a structured prompt with:

**Prompt Inputs:**
- ALIA Level (1–4): onboarding → competent → advanced → mastery
- Detected Product: product name or None
- Last Topic: medical topic from prior turns
- Current User Message: the question to respond to
- Language Rule: "Reply in English" or "Répondez en français"
- Conversation History: normalized last 5–8 turns
- Conversation State: JSON with product, phase, intent, role
- Memory Context: user_facts, summary_memory, recent_persistent_messages
- Knowledge Context: JSON content + RAG content or "No knowledge available"
- Recent Memory: latest 6 conversation turns for coherence

**Prompt Instruction (Highest Priority Order):**
1. Conversation context from history
2. JSON knowledge base content
3. RAG fallback content
4. Training behavior rules

**Behavior Constraints:**
- Never restart onboarding if `product_locked=true`
- Never ask "which product" if already established
- Never repeat the same question
- Stay natural, clinically grounded, evidence-based
- Use only provided knowledge; do not invent medical facts
- Choose next move based on conversation state (not generic templates)
- If user is clear and context exists, respond directly
- If user is unclear, clarify once then move forward
- Always include one coaching signal: reinforce, challenge, or ask follow-up

**LLM Parameters:**
- Temperature: 0.2 (low randomness, deterministic)
- Context Window: 8192 tokens
- Model: `llama3:8b` (local Ollama)

**Output Constraint:**
```json
{
  "action": "ask_question | respond | clarify | challenge",
  "message": "...",
  "topic": "indications | dosage | composition | warnings | side_effects | mechanism_of_action | administration | age | safety | patient_profile | other"
}
```

---

## 2. DECISION TREE: TRAINING ACTIONS & TOPICS

### 2.1 Action Types

| Action | When | Coaching Signal |
|--------|------|-----------------|
| `ask_question` | User needs to clarify intent; product not locked | Probe deeper, ask for specifics |
| `respond` | Clear intent, knowledge available, user ready | Affirm clarity, provide grounded answer |
| `clarify` | Ambiguous input, missing context, unclear product | Reflect back, ask 1 followup |
| `challenge` | Opportunity to coach rep on quality/accuracy | Reframe, ask for clinical backing, push depth |

### 2.2 Topic Selection (Locked to CRITICAL_TOPICS)

```python
CRITICAL_TOPICS = {
    "indications",
    "composition",
    "dosage",
    "administration",
    "warnings",
    "side_effects",
    "mechanism_of_action",
    "age",
    "safety",
    "patient_profile",
    "other"
}
```

**Topic Heuristic:**
- Extract from user question first
- Fall back to `last_topic` from state if unclear
- Default to "other" if no match

### 2.3 Conversation Phase Progression

```
Onboarding (turn_count ≤ 1, product not locked)
    ↓
Clinical Exploration (product_locked, turn_count 1–2)
    ↓
Deepening (product_locked, turn_count ≥ 3, context_established)
    ↓
Advanced Probe (deepening + last_topic in {mechanism, warnings, dosage})
```

Each phase locks different behavior:
- **Onboarding:** Clarify product, establish basic intent
- **Clinical Exploration:** Explore indications, composition, safety
- **Deepening:** Probe clinical reasoning, patient profiles
- **Advanced Probe:** Challenge depth, ask for mechanism/contraindications

---

## 3. TRAINING BRAIN LLM FLOW: STEP-BY-STEP

### 3.1 Prompt Construction

```python
prompt = f"""You are the dynamic training brain of a senior medical doctor coaching a pharmaceutical representative.

You are dual-role at the same time:
- ROLE A: Simulate a real medical doctor discussing clinical details.
- ROLE B: Coach the representative to improve clarity, structure, and evidence-based detailing quality.

You must decide the next conversational move in real time.

HIGHEST PRIORITIES:
1. Conversation context from the history
2. JSON knowledge base content if provided
3. RAG fallback content if provided
4. Training behavior rules

BEHAVIOR RULES:
- Never restart onboarding if the product or context has already been established.
- If the conversation state says product_locked=true, never ask "which product" again.
- Do not repeat the same question if the conversation already covered it.
- Stay natural, adaptive, and clinically grounded.
- Use only the provided knowledge context. Do not invent medical facts.
- Choose the most appropriate next move based on the conversation state.
- If the user is clear and context exists, respond directly rather than interrogating.
- If the user is unclear, clarify once and move forward.
- If the user needs challenge or coaching, keep it realistic and concise.
- Always include one coaching signal in training mode: reinforce, challenge, or ask one follow-up that improves clinical communication quality.
- Avoid repetitive stock phrases and avoid repeating the same sentence patterns.
- Never assume a specific person name or product name from prior examples. Use only the current conversation state and detected product.
- Always reply in the same language as the current user message (French or English).

INPUTS:
- ALIA level: {alia_level}
- Detected product: {product_text}
- Last topic discussed: {topic_text}
- Current user message: {user_message}
- Language rule: {language_instruction}

CONVERSATION HISTORY:
{history_text}

CONVERSATION STATE:
{state_json}

MEMORY CONTEXT:
{memory_json}

PERSISTENT RECENT MEMORY (LATEST TURNS):
{recent_memory_text}

KNOWLEDGE CONTEXT:
{knowledge_text}

RETURN THIS EXACT JSON SHAPE:
{{
  "action": "ask_question | respond | clarify | challenge",
  "message": "...",
  "topic": "indications | dosage | composition | warnings | side_effects | mechanism_of_action | administration | age | safety | patient_profile | other"
}}
"""
```

### 3.2 LLM Request

```python
payload = {
    "model": "llama3:8b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.2,
        "num_ctx": 8192
    }
}

response = http.post("http://localhost:11434/api/generate", json=payload)
raw_answer = response.json()["response"]
```

### 3.3 Response Parsing & Sanitization

```python
parsed = json.loads(extract_json_object(raw_answer))
action = sanitize_action(parsed.get("action"))         # ∈ {ask_question, respond, clarify, challenge}
message = parsed.get("message", "").strip()
topic = sanitize_topic(parsed.get("topic"))            # ∈ CRITICAL_TOPICS ∪ {None}
```

### 3.4 Fallback Guards

**Guard 1: Empty Message**
```python
if not message:
    return _fallback_decision(
        detected_product=resolved_drug,
        knowledge_context=knowledge_context,
        user_message=user_message,
        forced_language=language
    )
```

**Guard 2: Repetition Check**
```python
if _is_repetitive(message, conversation_history):
    return _non_repetitive_fallback(
        detected_product=resolved_drug,
        topic=topic,
        user_message=user_message,
        forced_language=language
    )
```

**Fallback Responses:**

*No Product Known:*
```
"Which product would you like to discuss so I can keep the coaching grounded?"
"Quel produit souhaitez-vous discuter pour que le coaching reste bien cible ?"
```

*With Product, No Knowledge:*
```
"Let us stay practical with {product}. How would you position it in one short clinical sentence?"
```

*Repetition Fallback (Challenge):*
```
EN: "Good. Now reframe the key point for {product} in a 20-second pitch, then add one concrete patient example."
FR: "Bien. Maintenant, reformulez le point-clé sur {product} en 20 secondes, puis donnez un exemple patient concret."
```

---

## 4. DATA PERSISTENCE & LEARNING SIGNALS

### 4.1 Post-Decision Memory Writes

After Training Brain returns a decision:

1. **Append QA Event to persistent_memory:**
   ```json
   {
     "type": "qa",
     "timestamp": "2026-05-02T12:45:30Z",
     "user_input": "What is the composition of BACTOL?",
     "system_output": "Training brain response...",
     "metadata": {
       "product": "BACTOL",
       "topic": "composition",
       "source": "training_brain",
       "action": "ask_question",
       "alia_level": 2,
       "turn_count": 3
     }
   }
   ```

2. **Update conversation_memory recent_messages window:**
   ```json
   "recent_messages": [
     {"role": "user", "content": "What is the composition of BACTOL?"},
     {"role": "assistant", "content": "Training brain response..."}
   ]
   ```

3. **Increment turn_count:**
   ```
   turn_count += 1
   ```

4. **Trigger Summarization (if needed):**
   ```python
   if len(recent_messages) >= summary_trigger_messages:  # default 10
       summary_memory = llm_summarize(overflow_messages)
       recent_messages = recent_messages[-recent_window_messages:]  # keep 6
   ```

### 4.2 High-Confidence RAG Learning Cache

When RAG returns an answer:
- **Threshold:** confidence ≥ 0.75
- **Filters:** no uncertainty, no fallback_triggered, no conflict_notes, diagnostic_confidence ≥ 0.6
- **Action:** Store in `rag_response_cache.json` keyed by normalized question hash
- **Reuse:** Next time same question appears, return cached answer directly (skip JSON/RAG re-query)

```python
cache_key = sha256(normalize_query(question)).hexdigest()
cache_data = {
    "question": question,
    "answer": rag_result.answer,
    "citations": rag_result.citations,
    "answer_confidence": rag_result.answer_confidence,
    "latency_ms": latency,
}
rag_response_cache[cache_key] = cache_data
```

### 4.3 Conversation State Evolution Tracking

The `conversation_state` dict is passed between turns and evolves:

```
Turn 1: product_locked=False, phase="onboarding"
  ↓ (User mentions "BACTOL")
Turn 2: product_locked=True, phase="clinical_exploration"
  ↓ (turn_count=2, user asks about composition)
Turn 3: product_locked=True, phase="deepening" (if turn_count ≥ 3)
  ↓ (user probes safety)
Turn 4: product_locked=True, phase="advanced_probe" (if last_topic in {warnings, dosage, mechanism})
```

Each evolution is recorded as metadata in the QA event, enabling:
- **Coachability Analysis:** Did rep move from onboarding to clinical exploration?
- **Topic Coverage:** Which topics were visited? How deep?
- **Turn Efficiency:** How many turns to reach deepening phase?

---

## 5. TRAINING LOOP: MULTI-TURN ADAPTIVE LEARNING

### 5.1 Conversation Context Extraction (Per Turn)

```python
context = {
    'drugs_mentioned': set(),        # All drugs mentioned across history
    'topics_covered': set(),         # Unique topics from CRITICAL_TOPICS touched
    'context_established': bool(),   # Has rep mentioned setting (clinic, hospital, etc.)?
    'turn_count': int()              # Count of user turns so far
}
```

**Used for:**
- Estimating ALIA Level
- Determining conversation phase
- Deciding when to deepen vs. repeat

### 5.2 ALIA Level Estimation

```python
def estimate_alia_level(context, question, resolved_drug, last_topic):
    level = 2  # baseline
    turn_count = context['turn_count']
    topic_count = len(context['topics_covered'])
    
    if turn_count <= 1 and not resolved_drug:
        level = 1  # onboarding level
    elif turn_count >= 4 or topic_count >= 2:
        level = 3  # expanding competence
    
    if any(deep_term in question.lower() for deep_term in ["mechanism", "contraindication", "interaction"]):
        level = max(level, 4)  # advanced
    
    if last_topic in {"mechanism_of_action", "warnings", "dosage"} and turn_count >= 3:
        level = max(level, 4)  # advanced probing
    
    return min(4, max(1, level))
```

**ALIA Levels:**
- **1 (Onboarding):** New rep, no product context, basic questions
- **2 (Foundational):** Product established, basic topics (composition, dosage, indications)
- **3 (Competent):** Multiple topics covered, context established, clinical reasoning visible
- **4 (Advanced):** Deep topics (mechanism, contraindications, complex interactions), multiple drugs

### 5.3 Training Brain Feedback Loop (Cross-Turn Learning)

```
Turn N:
  User Question
    ↓
  Training Brain Response (action, topic)
    ↓
  QA Event + Metadata logged
    ↓
Turn N+1:
  Load conversation_state from Turn N
  Extract new context
  Re-estimate ALIA level
  Prompt Training Brain with FULL history + updated state
    ↓
  Training Brain decision adapts based on:
    - product_locked status (don't re-ask product)
    - last_topic (don't repeat same topic if just covered)
    - turn_count (know how many exchanges so far)
    - conversation_phase (onboarding → clinical → deepening)
    - user_role (inferred from language markers)
    - alia_level (estimate rep competence)
```

**Adaptation Examples:**

| Turn | Input | Context | ALIA | Training Brain Decision |
|------|-------|---------|------|------------------------|
| 1 | "I'm a medical rep" | No product | 1 | `clarify`: "Which product?" |
| 2 | "I want to present BACTOL" | BACTOL locked | 2 | `ask_question`: "What's the indication?" |
| 3 | "It's for elderly patients" | phase=clinical_exploration | 2 | `respond`: + provide age-specific data |
| 4 | "How does the active ingredient work?" | phase=deepening, turn_count=3 | 3 | `challenge`: "Good, now connect that to tolerability for elderly patients" |

---

## 6. FALLBACK & ROBUSTNESS

### 6.1 Missing Knowledge Scenarios

| Scenario | Action | Fallback |
|----------|--------|----------|
| Product not in JSON KB | Query RAG | If RAG empty: "No knowledge available" |
| No JSON, RAG returns empty | Use Training Brain | Return clarify/ask_question action |
| Product locked, no topic detected | Use Training Brain + full product context | Ask for clinical reasoning |
| Language detection ambiguous | Default to English | Stored in state for consistency |

### 6.2 Concurrency Safety

- Per-request memory namespace resolves user isolation (no shared global state)
- Atomic JSON writes via temp file + replace (no partial corruptions)
- QA event append-only (no overwrites)
- Conversation state passed as parameter (no process-level mutation)

---

## 7. PROMPT REUSABILITY CHECKLIST

To reuse this prompt for another training session:

✅ **Update Inputs:**
- Current user question
- Conversation history (last 5–8 turns)
- Detected product (drug name or None)
- ALIA level (1–4)
- Last topic (from CRITICAL_TOPICS or None)
- Knowledge context (JSON content + RAG content or empty)
- Language (en or fr)
- User memory facts (name, role, company, preferences)
- Conversation state (product_locked, phase, intent, etc.)

✅ **Keep Fixed:**
- Behavior rules (same across all reps)
- Action set ({ask_question, respond, clarify, challenge})
- Topic set (CRITICAL_TOPICS)
- LLM parameters (temperature=0.2, num_ctx=8192)
- JSON schema for output

✅ **Adapt:**
- Knowledge context: Pull from JSON KB first, then RAG
- Memory context: Fresh load per turn (no stale process memory)
- ALIA level: Recalculate from turn_count + topic_count + question depth

---

## 8. EXAMPLE: COMPLETE 3-TURN TRAINING SESSION

### Turn 1: Onboarding

**User Input:**
> "I'm a medical rep, and I want to start my training with your doctor coach."

**Training Brain Inputs:**
```
alia_level: 1
detected_product: None
last_topic: None
conversation_history: []
knowledge_context: ""
turn_count: 0
```

**Training Brain Decision:**
```json
{
  "action": "ask_question",
  "message": "Which product would you like to discuss so I can keep the coaching grounded? We have BACTOL, HYDRA, COSMOPHARMA, and others.",
  "topic": "other"
}
```

**Memory Write:**
- QA event: input="I'm a medical rep...", output="Which product...", topic="other", action="ask_question", alia_level=1
- Recent messages: [{role: user, content: "I'm a..."}, {role: assistant, content: "Which product..."}]
- turn_count: 1
- Conversation state: product_locked=False, phase="onboarding"

---

### Turn 2: Product Established

**User Input:**
> "BACTOL. I want to understand its composition so I can explain it to doctors."

**Training Brain Inputs:**
```
alia_level: 2
detected_product: "BACTOL"
last_topic: "other"
conversation_history: [Turn 1 messages + Turn 2 user]
knowledge_context: "BACTOL composition: Chlorure de benzalkonium 0.18g (Savon), Alcool 70% (Gel), Alcool 95° (Spray)..."
turn_count: 1
```

**Training Brain Decision (using JSON knowledge):**
```json
{
  "action": "respond",
  "message": "Great choice. BACTOL has three forms with different active ingredients: the soap uses benzalkonium chloride 0.18%, while the gel and spray use alcohol. Now, what's the main clinical indication you'd emphasize to a physician—is it hand hygiene in clinical settings, or something else?",
  "topic": "composition"
}
```

**Memory Write:**
- QA event: input="BACTOL composition...", topic="composition", action="respond", product="BACTOL", alia_level=2
- turn_count: 2
- Conversation state: product_locked=True, phase="clinical_exploration"

---

### Turn 3: Clinical Deepening

**User Input:**
> "For elderly patients in a home care setting."

**Training Brain Inputs:**
```
alia_level: 3
detected_product: "BACTOL"
last_topic: "composition"
conversation_history: [Turn 1, 2, 3 messages]
knowledge_context: "Age suitability: BACTOL forms are suitable for all ages. Spray is easy for hands with reduced dexterity. Gel has no rinse required..."
turn_count: 2
context_established: True (mention of "home care setting")
```

**Training Brain Decision (using JSON + phase progression):**
```json
{
  "action": "challenge",
  "message": "Excellent context. Now here's the key: if you're positioning BACTOL for elderly home care, what's your one-sentence safety claim? And what's your evidence—is it the low alcohol content, the ease of use, or the broad age coverage? Be specific.",
  "topic": "age"
}
```

**Memory Write:**
- QA event: input="elderly patients...", topic="age", action="challenge", product="BACTOL", alia_level=3, phase="deepening"
- turn_count: 3
- Conversation state: phase="deepening" (turn_count ≥ 3 + context_established=True)

---

## 9. OBSERVABILITY & AUDIT TRAIL

Every training turn generates:

1. **Timestamped QA Event:**
   ```json
   {
     "timestamp": "2026-05-02T12:45:30Z",
     "user_id": "demo_user",
     "turn_id": 2,
     "user_input": "BACTOL composition...",
     "system_output": "Great choice. BACTOL has three forms...",
     "metadata": {
       "product": "BACTOL",
       "topic": "composition",
       "action": "respond",
       "source": "training_brain",
       "alia_level": 2,
       "conversation_phase": "clinical_exploration",
       "turn_count": 2,
       "language": "en",
       "knowledge_source": "json",
       "latency_ms": 450
     }
   }
   ```

2. **Conversation State Snapshot:**
   ```json
   {
     "product_locked": true,
     "current_product": "BACTOL",
     "conversation_phase": "clinical_exploration",
     "last_topic": "composition",
     "last_user_intent": "composition_probe",
     "turn_count": 2,
     "user_role": "rep"
   }
   ```

3. **Memory Update:**
   - QA pruned to 10 recent entries (older ones compressed into summary_memory)
   - Recent message window: 6 messages
   - Summary memory: LLM-compressed older context for long sessions

---

## 10. KEY INSIGHTS FOR REUSE

1. **The Training Brain is stateless:** It makes decisions based on input snapshot, not internal memory.
2. **State is external:** Conversation state, memory context, knowledge context passed as parameters.
3. **Fallbacks are cascading:** If LLM output is empty/repetitive, use deterministic fallback rules.
4. **Learning is append-only:** QA events logged, never mutated; memory compression happens at summarization boundary.
5. **Product lock prevents restart:** Once product_locked=true, training stays focused on that product until user explicitly switches.
6. **ALIA level drives depth:** Same question gets different responses at level 1 (clarify) vs. level 4 (challenge deeply).

---

## REUSE INSTRUCTIONS

To run training with a new rep or session:

```python
# 1. Load or create conversation_state
state = {
    "current_product": None,
    "product_locked": False,
    "conversation_phase": "onboarding",
    "last_topic": None,
    "last_user_intent": None,
    "current_intent": None,
    "user_role": "unknown",
    "turn_count": 0
}

# 2. Load user memory (name, facts)
memory = load_user_memory(user_id)

# 3. For each user input:
question = input("Rep: ")
intent = detect_intent(question)
product = intent.drug_name or state["current_product"]

# 4. Extract context
context = extract_conversation_context(messages)
alia_level = estimate_training_level(context, question, product, state["last_topic"])

# 5. Build training brain prompt (use template above)
prompt = build_training_prompt(
    conversation_history=messages,
    conversation_state=state,
    memory_context=memory,
    detected_product=product,
    alia_level=alia_level,
    last_topic=state["last_topic"],
    knowledge_context=get_knowledge(product),
    user_message=question,
    forced_language=detect_language(question)
)

# 6. Query LLM
decision = query_llm(prompt)

# 7. Persist
log_qa_event(question, decision.message, product, decision.topic, alia_level)
update_conversation_state(state, decision, product, alia_level)
update_memory(memory, question, decision.message)

# 8. Output
print(f"Doctor: {decision.message}")
```

---

**End of Training Brain System Prompt**
