Retrieval reranker benchmark dataset.

Schema per JSONL row:
- query: user query text
- expected_top_id: the candidate id expected at rank 1
- candidates: list of candidate objects with:
  - id: candidate identifier
  - score: baseline vector score (pre-rerank)
  - text: candidate text used for lexical reranking
