## Current Promoted Metrics (v6 Hybrid Baseline)

- Intent accuracy: **91.18%**
- Clarification recall: **100.00%**
- Entity recall: **72.73%**
- Coverage: **100.00%**
- Quality gate: **PASS** on both v2 and v3 test sets.

Evaluation artifacts:

- `NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v2.json`
- `NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v3.json`

## Post-Promotion Improvements: Entity Extraction v2

**Current work-in-progress:** Improving entity recall from 72.73% baseline using:

1. **Pharmaceutical domain dictionary** – 15+ products, 7+ molecules, dosage patterns, disease indications
2. **Fuzzy matching** – tolerates misspellings (e.g., "Omevie" → "Omévie") and alternative forms
3. **NER model fallback** (optional) – spaCy French model for out-of-dictionary entities
4. **Intelligent merging** – deduplicates and prioritizes by confidence source

**Expected improvements:**
- Dictionary + exact matching: ~72% recall (baseline)
- + Fuzzy matching: +5–8% recall → ~77–80%
- + NER fallback: +2–3% additional recall → ~79–83%

**Integration path:**
- [NLP/pipeline/entity_extractor_v2.py](../pipeline/entity_extractor_v2.py) – main extractor module
- [NLP/evaluation/eval_entity_extraction_v2.py](../evaluation/eval_entity_extraction_v2.py) – evaluation script
- [NLP/pipeline/ENTITY_EXTRACTION_V2_README.md](../pipeline/ENTITY_EXTRACTION_V2_README.md) – integration guide
- Dictionary: [NLP/taxonomy/pharmaceutical_dictionary.json](../taxonomy/pharmaceutical_dictionary.json)

**Next steps:**
1. Run evaluation: `python NLP/evaluation/eval_entity_extraction_v2.py NLP/datasets/eval_intent_safety_v5.jsonl`
2. Install optional deps: `pip install fuzzywuzzy python-Levenshtein`
3. Integrate into backend NLP pipeline (optional spaCy NER fallback)
4. Promote to production if recall target met and latency acceptable

## Promotion Detail

- Canonical hybrid artifact names were synced to v6 outputs.
- Legacy references now resolve to v6-equivalent results.
- In practice, "current promoted NLP" and "v6 hybrid baseline" are the same.

## Backend Integration

Hybrid adapter is wired as an optional runtime path in:

- `backend/routes/chat.py`
- `backend/utils/hybrid_adapter.py`

Environment toggles:

- `ALIA_USE_HYBRID_ADAPTER`
- `ALIA_HYBRID_MAX_NEW_TOKENS`
- `ALIA_USE_ENTITY_EXTRACTOR_V2`

Default env posture:

- `backend/.env` keeps `ALIA_USE_HYBRID_ADAPTER=0` by default.
- `ALIA_USE_ENTITY_EXTRACTOR_V2=0` keeps the legacy entity path unless explicitly enabled.
# ALIA NLP Phase 4 Handoff Summary

Date: 2026-04-18

## Status

- Phase 4 exit gate is passing: **28/28 checks green**.
- NLP is **production-candidate** for current scope.
- Current promoted baseline is **v6 hybrid** (supersedes v5 references).

## Current NLP Architecture

- **Base model**: Qwen2.5 1.5B Instruct (decoder-only transformer).
- **Fine-tuning method**: QLoRA (4-bit quantized base + LoRA adapter).
- **Runtime mode**: Hybrid adapter + baseline fallback.
- **Fallback behavior**:
  - Uses baseline intent when adapter emits generic intent.
  - Merges entity maps.
  - Preserves clarification and safety constraints.

## Promoted Model And Config

- Active config: `NLP/fine_tuning/hybrid_adapter_config.json`
- Promoted adapter directory:
  - `NLP/fine_tuning/models/intent_qlora_v6_1p5b_multiepoch`

## Current Promoted Metrics (v6 Hybrid Baseline)

- Intent accuracy: **91.18%**
- Clarification recall: **100.00%**
- Entity recall: **72.73%**
- Coverage: **100.00%**
- Quality gate: **PASS** on both v2 and v3 test sets.

Evaluation artifacts:

- `NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v2.json`
- `NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v3.json`

## Promotion Detail

- Canonical hybrid artifact names were synced to v6 outputs.
- Legacy references now resolve to v6-equivalent results.
- In practice, "current promoted NLP" and "v6 hybrid baseline" are the same.

## Backend Integration

Hybrid adapter is wired as an optional runtime path in:

- `backend/routes/chat.py`
- `backend/utils/hybrid_adapter.py`

Environment toggles:

- `ALIA_USE_HYBRID_ADAPTER`
- `ALIA_HYBRID_MAX_NEW_TOKENS`

Default env posture:

- `backend/.env` keeps `ALIA_USE_HYBRID_ADAPTER=0` by default.

## Operational Recommendation

1. Keep `ALIA_USE_HYBRID_ADAPTER=0` by default.
2. Run canary with `ALIA_USE_HYBRID_ADAPTER=1` for a small traffic slice.
3. Promote to full if latency and error rates remain healthy for 24-72 hours.

## Why These Choices

- **Qwen 1.5B**: strong quality-to-cost ratio for local inference.
- **QLoRA**: practical for constrained hardware and efficient local fine-tuning.
- **Hybrid strategy**: combines adapter quality gains with baseline safety/robustness.
- **Metrics selected for business behavior**:
  - Intent correctness
  - Clarification safety
  - Entity extraction utility
  - Output coverage reliability
