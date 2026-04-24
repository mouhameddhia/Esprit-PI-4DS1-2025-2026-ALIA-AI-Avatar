# NLP Workspace

This folder contains NLP-specific assets and implementation for ALIA.

## Structure

- `pipeline/`: Runtime NLP extraction and orchestration modules.
- `evaluation/`: Conversation scoring and competency evaluation modules.
- `taxonomy/`: Shared JSON taxonomies and label definitions.
- `prompts/`: Prompt templates for extraction and evaluation.
- `datasets/`: Labeled samples and evaluation sets.
- `docs/`: NLP-specific design and notes.

## Integration Notes

- Backend runtime imports remain stable via compatibility wrappers in `backend/utils/`.
- The source of truth for NLP taxonomy is now `NLP/taxonomy/nlp_taxonomy.json`.
- Shadow divergence tracking for side-by-side rollout is available via `NLP/evaluation/shadow_monitoring.py`.
- Optional backend hybrid NLP activation is controlled by `ALIA_USE_HYBRID_ADAPTER=1` and `ALIA_HYBRID_MAX_NEW_TOKENS` in the backend env file.
- The promoted hybrid adapter baseline currently points to `NLP/fine_tuning/models/intent_qlora_v6_1p5b_multiepoch` and passes the Phase 4 exit gate.
