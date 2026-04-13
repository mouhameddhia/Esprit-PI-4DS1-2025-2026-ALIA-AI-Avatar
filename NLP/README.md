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
