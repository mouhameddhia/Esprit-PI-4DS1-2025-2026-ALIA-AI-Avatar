# ALIA NLP — Full Technical Audit (Repo-Truth, Evidence-Only)

**Scope**: NLP system as implemented in the repository at `alia-web-main` (runtime NLP + optional paths + offline training + evaluation/CI + monitoring).  
**Method**: **No guessing.** Every statement is based on real files. If something is missing/unclear/unused, it is explicitly labeled.

---

## 0) What the ALIA “NLP system” actually is

In this repo, “NLP” is an **NLP decision layer** that converts each user message into a **structured analysis object** used for:
- routing (intent + tags),
- governance (safety/compliance flags),
- retrieval steering (rewritten query),
- analytics and evaluation (logged `nlp_events`, CI gates, shadow/drift monitoring).

It is **not one standalone model**. It is a hybrid pipeline with **three possible engines**:
- **Rule-based baseline** (always available) implemented inside `NLP/pipeline/nlp.py` fallback logic.
- **Hosted LLM extraction** (optional) via Groq in `NLP/pipeline/nlp.py` if `GROQ_API_KEY` is set.
- **Local fine-tuned adapter** (optional) Qwen + PEFT adapter loaded in `backend/utils/hybrid_adapter.py` when `ALIA_USE_HYBRID_ADAPTER=1`.

Architectural intent is documented in `NLP/docs/ADVANCED_NLP_ARCHITECTURE.md`.

---

## 1) File inventory and classification (NLP-related)

### 1.1 Core Runtime NLP (executed at message-time)

| File | Classification | Evidence (why it’s core) |
|---|---|---|
| `NLP/pipeline/nlp.py` | Core Runtime NLP | Defines `analyze_message_nlp` and fallback routing/merging; imported by backend wrapper `backend/utils/nlp.py`. |
| `NLP/pipeline/safety.py` | Core Runtime NLP | `detect_safety_flags` is called by `NLP/pipeline/nlp.py` (merged into output). |
| `NLP/pipeline/language.py` | Core Runtime NLP | `detect_language` is called by `NLP/pipeline/nlp.py`. |
| `NLP/taxonomy/nlp_taxonomy.json` | Core Runtime NLP asset | Defines allowed `intents`, `safety_flags`, `entity_types`, tags; loaded by runtime NLP. |
| `NLP/taxonomy/domain_synonyms.json` | Core Runtime NLP asset | Used by `_normalize_user_text` in `NLP/pipeline/nlp.py` to canonicalize terms. |
| `NLP/prompts/nlp_extraction_system_prompt.txt` | Core Runtime NLP asset | Used as hosted extraction system prompt template in `NLP/pipeline/nlp.py`. |

### 1.2 Optional Runtime NLP (gated by env/deps)

| File | Classification | Gate / condition |
|---|---|---|
| `NLP/pipeline/entity_extractor_v2.py` | Optional Runtime NLP | Used only if `ALIA_USE_ENTITY_EXTRACTOR_V2=1` and import succeeds; invoked by `NLP/pipeline/nlp.py`. |
| `backend/utils/hybrid_adapter.py` | Optional Runtime NLP | Used only if `ALIA_USE_HYBRID_ADAPTER=1`; loads base model + PEFT adapter and generates JSON. |
| `NLP/fine_tuning/hybrid_adapter_config.json` | Optional Runtime NLP config | Points to promoted adapter dir; records metrics/training metadata. |

### 1.3 Offline Training (not used directly at message-time)

| File | Classification | What it produces |
|---|---|---|
| `NLP/fine_tuning/build_finetune_dataset.py` | Offline Training | Builds instruction-tuning datasets from benchmark JSONL to OpenAI-messages JSONL. |
| `NLP/fine_tuning/train_lora_intent_sft.py` | Offline Training | Runs LoRA/QLoRA SFT (TRL `SFTTrainer`) and saves adapter artifacts. |
| `NLP/fine_tuning/generate_candidate_predictions.py` | Offline Training/Eval | Runs hosted/local baseline/local adapter inference and projects outputs to taxonomy. |
| `NLP/fine_tuning/models/**` | Offline artifacts | Adapter weights/config/tokenizer (e.g., `adapter_model.safetensors`, `adapter_config.json`). |

### 1.4 Evaluation / Benchmarking / Gates (CI)

These are executed by `.github/workflows/nlp-eval.yml`.

| File | Classification | Purpose |
|---|---|---|
| `NLP/evaluation/run_eval.py` | Evaluation gate | Intent accuracy + safety recall + secondary tags recall + entity-map recall thresholds. |
| `NLP/evaluation/run_clarification_eval.py` | Evaluation gate | Clarification behavior quality gate. |
| `NLP/evaluation/run_retrieval_rerank_eval.py` | Evaluation gate | Reranking Hit@1, MRR, regression constraints. |
| `NLP/evaluation/run_language_detection_eval.py` | Evaluation gate | Language detection accuracy threshold. |
| `NLP/evaluation/shadow_monitoring.py` | Monitoring + gate | Computes divergence between “primary” vs “shadow” predictions; has a max-divergence gate. |
| `NLP/evaluation/drift_monitoring.py` | Monitoring | Archives eval JSON and compares to baseline metrics; emits drift warnings. |
| `NLP/evaluation/phase4_exit_gate.py` | Release gate | Asserts expected artifacts exist and summarizes pass/fail. |
| `NLP/evaluation/promote_candidate.py` | Release automation | Candidate promotion decision (baseline vs candidate + shadow constraints). |
| `NLP/evaluation/retraining_trigger.py` | Ops automation | Emits whether retraining should trigger based on multiple gate artifacts. |

### 1.5 Monitoring (runtime scheduled)

| File | Classification | What it does |
|---|---|---|
| `backend/utils/background_tasks.py` | Monitoring | Generates “shadow logs” by re-running NLP on stored messages and writing `NLP/evaluation/results/shadow_*.json(l)`. |

### 1.6 Not used in runtime (exists, but not called in message-time NLP)

| File | Classification | Evidence |
|---|---|---|
| `NLP/pipeline/intent_model.py` | Not used in runtime | Called by `NLP/evaluation/train_intent_model.py` and `run_trained_intent_shadow_eval.py` (CI), not imported by `NLP/pipeline/nlp.py`. |
| `NLP/pipeline/entity_model.py` | Not used in runtime | Called by evaluation scripts, not imported by `NLP/pipeline/nlp.py`. |

---

## 2) Exact execution flow (user message → final answer), with real files

### 2.1 Backend receives the message

Entry: `backend/routes/chat.py` handles `POST /chat/message`.

It calls:
- `backend/routes/chat.py::_analyze_message_nlp(user_text, history, mode)`

### 2.2 NLP analysis selection (baseline vs hybrid adapter)

`_analyze_message_nlp` does one of:
- **Baseline**: `backend/utils/nlp.py` → `NLP/pipeline/nlp.py::analyze_message_nlp`
- **Hybrid adapter** (optional): `backend/utils/hybrid_adapter.py` + merge with baseline analysis

### 2.3 Clarification gate

If NLP output includes action item `needs_intent_clarification`, the backend returns a clarification reply and does **not** proceed to retrieval/generation.

### 2.4 Retrieval and reranking (NLP influences retrieval)

If clarification is not needed:
- Backend uses `nlp_analysis["rewritten_query"]` as the retrieval query (or falls back to the user message).
- Retrieval context is built by `backend/utils/rag_pipeline.py::RAGPipeline.get_context(...)`.
- Reranking is performed by `NLP/pipeline/reranker.py::rerank_candidates(...)`.

*(The final response generation itself is performed in the backend using Groq; see `backend/routes/chat.py`.)*

---

## 3) NLP architecture by layer (based on real code)

### 3.1 Preprocessing layer (implemented)

Implemented in `NLP/pipeline/nlp.py`:
- **Domain synonym normalization**: `_normalize_user_text` applies `NLP/taxonomy/domain_synonyms.json` via word-boundary regex replacement.
- **Whitespace cleanup**: collapses whitespace after replacements.
- **Language detection**: `NLP/pipeline/language.py::detect_language` (heuristic marker scoring + Arabic script detection + Arabizi heuristics).

**Not present in repo** (do not claim):
- spelling correction module/service
- dedicated tokenization/lemmatization pipeline beyond regex tokenization inside certain rules

### 3.2 Understanding layer (implemented)

#### Intent classification / routing (implemented, baseline)
In `NLP/pipeline/nlp.py`:
- `_INTENT_TOKEN_MAP`: intent → keyword/phrase list
- `_intent_scores` → `_coarse_route_from_scores` (bucket routing) → `_fine_intent_from_route` (choose intent)
- `_HARD_NEGATIVE_INTENT_OVERRIDES`: phrase overrides for known confusions
- `_rule_intent_router` returns `(intent, confidence)` using a score-to-confidence heuristic

#### Secondary tags (implemented)
In `NLP/pipeline/nlp.py`, `_infer_secondary_tags` extracts visit format tags, objection/methodology tags, then filters to taxonomy-derived `SUPPORTED_SECONDARY_TAGS`.

#### Entity extraction / slot filling (implemented, hybrid merge)
In `NLP/pipeline/nlp.py`, entities are merged from:
- Hosted LLM `entity_map` (if Groq extraction is used) normalized via `_normalize_entity_map`
- Rule entity hints `_rule_entity_map`
- Optional pharma extractor v2 `_pharma_entity_map` (dictionary + fuzzy + patterns), gated by `ALIA_USE_ENTITY_EXTRACTOR_V2`

#### Confidence scoring (implemented)
- Hosted extraction: `confidence` from LLM JSON is cast to float and clamped to \([0,1]\).
- Baseline routing: confidence is derived from keyword match strength in `_rule_intent_router`.

#### Clarification detection (implemented)
Baseline path uses `confidence < 0.5` to add action item `"needs_intent_clarification"` (see `_fallback_analysis` in `NLP/pipeline/nlp.py`).

### 3.3 Governance layer (implemented)

#### Safety / compliance flagging (implemented, rule-based)
`NLP/pipeline/safety.py::detect_safety_flags` emits taxonomy-constrained safety flags (e.g., `patient_specific_advice_request`, `high_risk_interaction`, `off_label_request`, `overclaim_risk`, `source_required`).  
These are merged into final NLP output by `NLP/pipeline/nlp.py::_normalize_safety_flags`.

#### Hallucination prevention (not implemented as an NLP module)
No dedicated hallucination detector/scorer exists in `NLP/`. Governance is implemented indirectly via:
- safety flags that influence downstream prompting (in backend),
- retrieval grounding via RAG context (backend),
- CI regression/gate harnesses.

### 3.4 Knowledge layer (implemented across NLP + backend)

#### Query rewriting (implemented)
`rewritten_query` comes from hosted extraction JSON if present; otherwise it falls back to normalized user text in `NLP/pipeline/nlp.py`.

#### Retrieval reranking (implemented in NLP)
`NLP/pipeline/reranker.py` blends vector similarity with lexical overlap/Jaccard to compute `rerank_score`.

### 3.5 Hybrid decision logic (implemented)
Two hybrid merges exist:
- **Inside `NLP/pipeline/nlp.py`**: hosted LLM output is merged with rule intent router and rule/entity extractors; safety detector flags are always merged.
- **Inside backend hybrid adapter path**: adapter JSON output is merged into baseline NLP analysis (`backend/routes/chat.py`) when `ALIA_USE_HYBRID_ADAPTER=1`.

### 3.6 Monitoring layer (implemented)
- **CI gates**: `.github/workflows/nlp-eval.yml` runs evaluation, shadow divergence, drift archive, rerank eval, clarification eval, language eval, promotion decision, retraining trigger, and Phase 4 exit gate.
- **Runtime scheduled shadow snapshots**: `backend/utils/background_tasks.py` re-runs `NLP/pipeline/nlp.py::analyze_message_nlp` on stored conversation events and writes artifacts into `NLP/evaluation/results`.

---

## 4) Models and hyperparameters (exact values from repo)

### 4.1 Hosted LLM extraction (optional)
- **Provider**: Groq (SDK import in `NLP/pipeline/nlp.py`; installed in CI workflow).
- **Model**: `GROQ_MODEL` env, default **`llama-3.3-70b-versatile`** in `NLP/pipeline/nlp.py`.
- **Inference params (extraction call)**: `temperature=0.1`, `max_tokens=400` in `NLP/pipeline/nlp.py`.
- **System prompt**: `NLP/prompts/nlp_extraction_system_prompt.txt` (strict JSON schema).

### 4.2 Local fine-tuned transformer (optional runtime)
Promoted adapter config: `NLP/fine_tuning/hybrid_adapter_config.json`:
- **Base model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Adapter dir**: `NLP/fine_tuning/models/intent_qlora_v6_1p5b_multiepoch`
- **Max new tokens (default)**: 220 (also mirrored in config)
- **Hybrid strategy**: `baseline_first_intent`
- **Recorded training metadata**: epochs=3, batch_size=1, gradient_accumulation_steps=8, max_seq_length=384

LoRA adapter parameters are stored in the adapter artifact’s `adapter_config.json` (PEFT):
- **rank \(r\)**: 16
- **alpha**: 32
- **dropout**: 0.05
- **target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`

### 4.3 CI-only trained shadow models (not runtime)
- Intent shadow model: `NLP/pipeline/intent_model.py` (multinomial Naive Bayes) trained via `NLP/evaluation/train_intent_model.py`.
- Entity shadow model: `NLP/pipeline/entity_model.py` (phrase lexicon + token index + rule fallback) trained via `NLP/evaluation/train_entity_model.py`.
- Reranker weights: trained via `NLP/evaluation/train_reranker_model.py` and used in holdout gate.

---

## 5) Evaluation system (what is measured, and current values present in repo)

### 5.1 CI gates and thresholds (from `.github/workflows/nlp-eval.yml`)

This repo defines explicit quality gates in CI.

| Gate | Script | Key metrics | Thresholds enforced in CI |
|---|---|---|---|
| Core NLP gate | `NLP/evaluation/run_eval.py` | intent accuracy, safety recall, secondary tags recall, entity-map recall | intent ≥ 0.90; safety recall ≥ 0.95; secondary tags recall ≥ 0.60; entity-map recall ≥ 0.50 |
| Shadow divergence | `NLP/evaluation/shadow_monitoring.py` | divergence_rate | max divergence ≤ 0.15 |
| Clarification behavior | `NLP/evaluation/run_clarification_eval.py` | clarification recall/precision, clear-intent accuracy | recall ≥ 0.80; precision ≥ 0.70; clear intent accuracy ≥ 0.85 |
| Retrieval rerank gate | `NLP/evaluation/run_retrieval_rerank_eval.py` | hit@1, MRR, regression | hit@1 ≥ 0.70; MRR ≥ 0.85; regression ≤ 0.01 |
| Language detection | `NLP/evaluation/run_language_detection_eval.py` | accuracy | accuracy ≥ 0.90 |

### 5.2 Promoted baseline metrics (documented in repo)

`NLP/docs/PHASE4_HANDOFF_SUMMARY.md` reports the currently promoted “v6 hybrid baseline” metrics:
- Intent accuracy: **91.18%**
- Clarification recall: **100%**
- Entity recall: **72.73%**
- Coverage: **100%**

### 5.3 Drift monitoring (what exists vs what does not)

- **Exists**: `NLP/evaluation/drift_monitoring.py` archives evaluation artifacts and compares against a hard-coded baseline (v4 numbers in that file and in CI workflow).
- **Not present in `NLP/`**: runtime latency/throughput/memory benchmarking harness for model inference (no profiler modules found under `NLP/`).

---

## 6) Example walkthrough (mechanical, aligned with real pipeline semantics)

Example input:

> “Doctor says your medicine is too expensive and asks if it interacts with warfarin.”

### 6.1 Preprocessing
- `_normalize_user_text` applies domain synonyms if present (`NLP/taxonomy/domain_synonyms.json`).
- `detect_language` assigns `en`/`fr`/`ar`/`unknown` (`NLP/pipeline/language.py`).

### 6.2 Intent classification / routing
Baseline router in `NLP/pipeline/nlp.py` will score signals from both objection and safety language. The runtime output remains single-intent (taxonomy requires one intent), but can express mixture via tags (`mixed_intent` exists) and/or confidence/clarification behavior.

### 6.3 Entity extraction
Entity values come from a merge of (a) hosted JSON `entity_map` (if Groq enabled), (b) rule entity hints, and (c) optional pharma extractor v2.  
**Truthful limitation**: the v2 pharma extractor dictionary is focused on products/molecules in `NLP/taxonomy/pharmaceutical_dictionary.json`; “warfarin” is not guaranteed to be captured unless the hosted/adapter output includes it or the dictionary contains it.

### 6.4 Safety/compliance flags
The safety detector in `NLP/pipeline/safety.py` includes explicit interaction-context logic and can emit `high_risk_interaction` when interaction language is present with medication context.

### 6.5 Query rewrite + retrieval steering
- Hosted extraction can return a concise `rewritten_query`.
- Otherwise the system uses normalized message text as the retrieval query.

### 6.6 Reranking
`NLP/pipeline/reranker.py` computes `rerank_score` as a normalized blend of vector similarity and lexical overlap/Jaccard; it is used to reorder candidates returned by the vector DB.

### 6.7 Logs and monitoring hooks
The backend stores per-message `nlp_events` (intent, safety_flags, rewritten_query, etc.). Shadow monitoring can later re-run NLP on stored events and compute divergence artifacts (`backend/utils/background_tasks.py` + `NLP/evaluation/shadow_monitoring.py`).

---

## 7) Academic justification (defensible, but constrained to repo-truth)

### 7.1 Why a hybrid decision layer (rules + model + gates)
- **Deterministic governance**: safety flags are rule-based (`NLP/pipeline/safety.py`) so compliance signals don’t depend on model randomness.
- **Schema discipline**: taxonomy + strict JSON schema make the system testable and integrable (`NLP/taxonomy/nlp_taxonomy.json`, `NLP/prompts/nlp_extraction_system_prompt.txt`).
- **Measurable quality**: CI defines explicit thresholds for key behaviors (`.github/workflows/nlp-eval.yml`).
- **Safe rollout**: shadow divergence gating enables side-by-side evaluation before adopting changes (`NLP/evaluation/shadow_monitoring.py`).

### 7.2 Why QLoRA + a small Qwen base (what is explicitly implemented)
- QLoRA/4-bit loading is implemented for local adapter inference (`backend/utils/hybrid_adapter.py` uses 4-bit load) and supported in training script (`NLP/fine_tuning/train_lora_intent_sft.py`).
- The promoted base model is `Qwen/Qwen2.5-1.5B-Instruct` (`NLP/fine_tuning/hybrid_adapter_config.json` and adapter artifact config).
- LoRA settings (r=16, alpha=32, dropout=0.05; attention projection targets) are stored in adapter `adapter_config.json`.

### 7.3 What you should not claim as repo-truth
- Exact transformer architectural counts (layers/heads/hidden size) for Qwen unless you cite upstream model documentation (not stored in this repo).
- Any dedicated “hallucination score” subsystem inside `NLP/` (not present).


