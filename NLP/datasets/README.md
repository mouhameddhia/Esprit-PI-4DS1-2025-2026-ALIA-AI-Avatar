# NLP Datasets

This folder stores labeled data used to evaluate and improve the NLP pipeline.

## Files

- eval_intent_safety_template.jsonl: Starter evaluation dataset format.
- eval_intent_safety_v1.jsonl: First labeled evaluation set (24 samples).
- eval_intent_safety_v2.jsonl: Expanded set derived from teacher files in useful-files/ (58 samples).
- eval_intent_safety_v3.jsonl: Larger benchmark with same-domain variants (71 samples).
- eval_intent_safety_v4.jsonl: Extended benchmark covering methodology, objections, safety, physician questions (153 samples) - **STABLE BASELINE**.
- eval_intent_safety_v5.jsonl: Expansion to 250 samples for future work (needs NLP heuristic refinement).
- eval_intent_safety_public_supplement.jsonl: Public-source curated examples (30 samples) - requires domain-specific fine-tuning to adopt.
- eval_intent_clarification_v1.jsonl: Ambiguity and low-confidence clarification benchmark (48 samples).
- eval_retrieval_rerank_v1.jsonl: Retrieval rerank seed benchmark with labeled candidate sets (10 samples).
- eval_retrieval_rerank_v2.jsonl: Expanded retrieval rerank benchmark for production gating (60 samples).
- eval_retrieval_rerank_real_v1.jsonl: Real-log derived rerank pairs from production user messages (auto-labeled heuristic set).
- eval_retrieval_rerank_v3.jsonl: Mixed benchmark (real-log pairs + synthetic backfill) used for stable CI gating.
- eval_language_detection_v1.jsonl: Runtime multilingual detection benchmark (English/French/Arabic) for production gate coverage.

## JSONL Schema

Each line is one JSON object:

- text: user message text
- mode: physician_portal or medrep_training
- expected_intent: target primary intent label
- expected_safety_flags: expected safety flags (optional list)
- expected_secondary_tags: expected secondary tags (optional list)
- expected_entity_map: expected entity map keyed by taxonomy entity type (optional object)

Clarification dataset fields (`eval_intent_clarification_v1.jsonl`):

- text: user message text
- mode: physician_portal or medrep_training
- expected_clarification: whether `needs_intent_clarification` should be emitted
- expected_intent: expected primary intent for non-clarification rows

Retrieval rerank dataset fields (`eval_retrieval_rerank_v1.jsonl`, `eval_retrieval_rerank_v2.jsonl`):

- query: user query text
- expected_top_id: expected rank-1 candidate id after reranking
- candidates: list of candidate objects with `id`, baseline `score`, and candidate `text`

Real-label review fields (`eval_retrieval_rerank_real_v1.jsonl`):

- human_verified: whether expected label has been human reviewed
- reviewer_id: reviewer identifier (email/alias)
- reviewed_at: ISO timestamp for review completion
- review_notes: free-text reviewer rationale or corrections

Language detection dataset fields (`eval_language_detection_v1.jsonl`):

- text: user message text
- expected_language: expected runtime language code (`en`, `fr`, `ar`, `unknown`)

Example:

{"text":"Can this be used for my patient with renal issues?", "mode":"physician_portal", "expected_intent":"safety_question", "expected_safety_flags":["patient_specific_advice_request"], "expected_secondary_tags":["objection_handling"], "expected_entity_map":{"patient_profile":["renal issues"]}}
