# Per-User Memory Architecture

This document describes the user-scoped memory design for the Hybrid Medical Agent.

## 1. Scope and Goals

The memory layer is designed to be:

- identity-safe
- session-consistent
- scalable across multiple medical representatives
- traceable and auditable

A user is identified by authenticated credentials (for example `auth_user.id`) and mapped to an isolated on-disk namespace.

## 2. User Namespace Layout

Per-user memory is stored under:

```text
memory/
  users/
    <user_id>/
      conversation_memory.json
      persistent_memory.json
```

The implementation resolves these paths in:

- `hybrid_medical_agent/agent/user_memory_namespace.py`

Rules:

- `user_id` is sanitized to a filesystem-safe identifier.
- Path traversal is prevented by strict normalization.
- The folder is auto-created on first access.

## 3. Real-Time Memory Loading

At the beginning of every request, the controller rebinds memory to the current user namespace.

Implemented in:

- `HybridMedicalController._memory_store_for_user(...)`
- `HybridMedicalController.handle_query(..., user_id=...)`

Behavior:

1. resolve user namespace from `user_id`
2. create a fresh `PersistentMemoryStore` bound to that user files
3. read/write from disk during this request

This avoids stale shared memory between users.

## 4. Internal Memory Files

### 4.1 conversation_memory.json

Purpose: short-term and compressed conversation state for one user.

Main fields:

- `user_facts`: latest known profile facts (`name`, `role`, `company`, `preferences`)
- `recent_messages`: rolling short-term context window
- `summary_memory`: compressed long-range conversation summary (structured)
- `turn_count`: number of user turns persisted for this user
- `last_sync_event_id`: traceability link to latest persistent event

Why these fields exist:

- `user_facts`: stores stable profile attributes needed for identity-aware responses. Keeping this separate avoids forcing the model to re-infer profile data from raw chat each turn.
- `recent_messages`: preserves exact wording of the latest turns for short-range coherence and accurate follow-up behavior.
- `summary_memory`: keeps older context in compressed form so prompts stay within context budget while preserving intent continuity.
- `turn_count`: gives an explicit progression signal for coaching logic, onboarding transitions, and future policy rules.
- `last_sync_event_id`: supports auditability and troubleshooting when diagnosing memory write or merge issues.

### 4.2 persistent_memory.json

Purpose: append-style event history and cache source for one user.

Contains events such as:

- `qa`
- `user_fact`
- `preference`
- `instruction`
- `statement`

Important retention rule:

- QA entries are automatically pruned to the latest 10 (`max_qa_entries=10`) to reduce stale context pressure.

Why append-style events are used:

- Event logs are easier to audit than mutable snapshots because each decision is tied to a timestamped record.
- This supports root-cause analysis when a wrong answer appears (for example, checking which user fact was last written).
- It also enables selective cleanup (for example, pruning only QA while retaining user facts).

## 5. Memory Lifecycle Per Turn

For each request:

1. Load user-scoped memory from disk.
2. Parse/store new user facts (if provided).
3. Answer with JSON-first, RAG fallback logic.
4. Persist the QA event.
5. Increment `turn_count`.
6. Update `recent_messages`.
7. If threshold exceeded, summarize overflow into `summary_memory` and prune recent window.

This provides both short-term precision and long-term continuity.

Why this order is intentional:

1. Load first: guarantees decisions are based on latest on-disk state, not stale process memory.
2. Parse/store user facts early: ensures the same request can already benefit from newly provided identity data.
3. Persist after response generation: keeps a complete QA trace (input + output + metadata) for observability.
4. Summarize last: avoids summarizing partial writes and keeps summary based on committed turn history.

## 6. Why Recent + Summary + Persistent Are Separate

- `recent_messages`: high-fidelity short-term context for immediate coherence.
- `summary_memory`: compressed history to stay within model context budget.
- `persistent_memory`: auditable event log and deterministic fallback for facts/cache.

This separation reduces prompt bloat, preserves critical facts, and supports traceability.

## 7. Turn Management and Summarization

Turn management:

- One call to `add_qa(...)` increments `turn_count` by 1.

Summarization trigger:

- When `len(recent_messages) >= summary_trigger_messages`, overflow is summarized and pruned.
- This keeps bounded recent context while retaining important long-range signals.

Why summary is required in production:

- LLM context windows are finite; unbounded chat history causes prompt bloat, latency increase, and higher cost.
- Recent turns contain high-value lexical details (exact wording, disambiguations), while older turns usually only need semantic intent preservation.
- Summary acts as a compression layer that preserves continuity while controlling growth.

Why these default values are chosen:

- `max_qa_entries=10`:
  - Keeps enough recent QA pairs for deterministic cache benefit.
  - Reduces stale-answer risk from old conversational contexts.
  - Bounds disk growth and lookup time.
- `summary_trigger_messages=10`:
  - Delays compression until enough interaction signal exists.
  - Avoids over-summarizing early, which can remove useful detail.
- `recent_window_messages=6`:
  - Retains the most relevant short-term conversation window after summarization.
  - Balances coherence against context size.

Tuning guidance:

- Increase `recent_window_messages` for longer, high-precision coaching dialogues.
- Decrease `summary_trigger_messages` if latency/token usage becomes high.
- Increase `max_qa_entries` only if QA cache hit rate is too low and leakage risk is controlled.

## 8. Identity Safety

Identity is tied to authenticated `user_id`, not free-text self-introduction.

Why this is safer than parsing "I am X":

- text-based identity can be noisy/ambiguous
- users can change names in text accidentally
- authenticated user namespace guarantees memory isolation

Text facts are still useful as profile attributes, but data ownership is controlled by `user_id`.

## 9. Robustness and Concurrency Safety

Implemented protections:

- auto-initialize missing files/directories
- atomic JSON writes using temp file + replace
- bounded QA retention to avoid unbounded growth
- per-request namespace resolution to reduce stale-process contamination

Failure modes and mitigations:

- Failure mode: cross-user data leakage.
  - Mitigation: strict user namespace resolution by authenticated `user_id`; no shared global memory file.
- Failure mode: stale identity returned after user switch.
  - Mitigation: per-request memory rebinding plus newest-fact retrieval policy.
- Failure mode: JSON corruption on interrupted write.
  - Mitigation: atomic write via temporary file then replace.
- Failure mode: context explosion over long sessions.
  - Mitigation: bounded recent window + summarization trigger + QA pruning.
- Failure mode: summary drift or lossy compression.
  - Mitigation: keep exact recent messages; summary stores only older overflow.
- Failure mode: race between concurrent requests for same user.
  - Current state: atomic writes reduce partial writes.
  - Recommended next step: add per-user file lock (mutex) to serialize concurrent writes.

Operational signals to monitor:

- per-user `turn_count` growth rate
- summary update frequency
- QA prune frequency
- memory file size over time
- mismatch incidents between expected and returned user profile facts

These metrics help validate whether current retention and summary thresholds are still appropriate at scale.

## 10. Integration Points

Controller:

- `handle_query(..., user_id=...)`
- `reset_session(..., user_id=...)`
- `start_training_turn(..., user_id=...)`
- `evaluate_training_answer(..., user_id=...)`

Streamlit:

- passes current user id from sidebar field into each controller call

CLI:

- `--user-id` routes each command to the correct memory namespace

## 11. Design Decision Summary (Why This Architecture)

- Per-user files instead of shared memory:
  - Chosen to maximize isolation, simplify audits, and reduce accidental leakage.
- Real-time load/write each request instead of long-lived cache:
  - Chosen to guarantee consistency across multiple worker processes and sessions.
- Split conversation vs persistent memory:
  - Chosen to separate prompt-efficient state (`conversation_memory.json`) from auditable event history (`persistent_memory.json`).
- Summary + recent hybrid model:
  - Chosen to preserve immediate lexical fidelity while keeping long-term continuity under context limits.

This combination is intentionally conservative for pharmaceutical training contexts where identity correctness, traceability, and bounded behavior are higher priority than aggressive caching.
