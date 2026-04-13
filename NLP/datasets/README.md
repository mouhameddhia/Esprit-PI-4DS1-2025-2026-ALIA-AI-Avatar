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

## JSONL Schema

Each line is one JSON object:

- text: user message text
- mode: physician_portal or medrep_training
- expected_intent: target primary intent label
- expected_safety_flags: expected safety flags (optional list)
- expected_secondary_tags: expected secondary tags (optional list)
- expected_entity_map: expected entity map keyed by taxonomy entity type (optional object)

Example:

{"text":"Can this be used for my patient with renal issues?", "mode":"physician_portal", "expected_intent":"safety_question", "expected_safety_flags":["patient_specific_advice_request"], "expected_secondary_tags":["objection_handling"], "expected_entity_map":{"patient_profile":["renal issues"]}}
