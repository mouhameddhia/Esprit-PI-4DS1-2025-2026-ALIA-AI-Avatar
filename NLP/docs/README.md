# NLP Docs Index

This folder is the source of truth for NLP architecture, taxonomy, ingestion, and evaluation.

## Recommended Reading Order

1. [ADVANCED_NLP_ARCHITECTURE.md](ADVANCED_NLP_ARCHITECTURE.md)
2. [NLP_LABEL_TAXONOMY.md](NLP_LABEL_TAXONOMY.md)
3. [RAG_INGESTION_PLAN.md](RAG_INGESTION_PLAN.md)
4. [NLP_EVALUATION_MATRIX.md](NLP_EVALUATION_MATRIX.md)

## Document Purpose

- [ADVANCED_NLP_ARCHITECTURE.md](ADVANCED_NLP_ARCHITECTURE.md): End-to-end design of the advanced NLP system and phased build plan.
- [NLP_LABEL_TAXONOMY.md](NLP_LABEL_TAXONOMY.md): Label schema for intents, entities, tags, and safety signals.
- [RAG_INGESTION_PLAN.md](RAG_INGESTION_PLAN.md): How to ingest and structure domain documents for retrieval.
- [NLP_EVALUATION_MATRIX.md](NLP_EVALUATION_MATRIX.md): Competency scoring and progression criteria for Debutant, Junior, Confirme, and Expert.

## Operational Notes

- Root-level docs are lightweight pointers to this folder.
- Runtime taxonomy JSON lives in [../taxonomy/nlp_taxonomy.json](../taxonomy/nlp_taxonomy.json).
- Runtime NLP pipeline lives in [../pipeline/nlp.py](../pipeline/nlp.py).
- Conversation evaluation logic lives in [../evaluation/evaluator.py](../evaluation/evaluator.py).
