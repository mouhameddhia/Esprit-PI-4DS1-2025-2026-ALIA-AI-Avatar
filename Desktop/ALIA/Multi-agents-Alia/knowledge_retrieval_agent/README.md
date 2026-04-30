# Knowledge Retrieval Agent (Advanced Hybrid + Corrective RAG)

Production-style modular RAG system for pharmaceutical/scientific question answering.

This project is not a basic vector-search demo. It combines:

- Hybrid retrieval: semantic dense search + lexical sparse search
- Metadata-aware ranking (authority tier, recency, source completeness)
- Corrective RAG (CRAG): retrieval confidence evaluation + corrective fallback retrieval
- Domain query rewriting for pharma/scientific language
- Cross-encoder reranking for final relevance ordering
- Citation-grounded local generation with Ollama and structured output
- Ingestion hardening: OCR fallback, adaptive embedding batching, chunk quality gates, failure logging

## Technologies Used (Specific)

Core runtime and configuration:

- Python 3.11+
- `pydantic` + `pydantic-settings` for typed config and `.env` support
- `python-dotenv` for environment loading

Retrieval and indexing stack:

- `sentence-transformers` for dense embeddings and cross-encoder reranking
- `langchain-huggingface` embedding wrappers
- `faiss-cpu` + `langchain-community` FAISS integration for dense vector search
- `rank-bm25` for sparse lexical BM25 retrieval

Generation stack:

- `langchain-core` prompt composition
- `langchain-ollama` (`ChatOllama`) for local LLM inference
- Ollama model: `llama3:8b` (default in config)

Document ingestion and parsing:

- `pymupdf` (`fitz`) for PDF parsing
- `pytesseract` + `pillow` for OCR fallback (scanned pages/images)
- Native Tesseract binary (required for OCR at system level)
- Built-in XML + ZIP parsing for PPTX text extraction

Reliability, observability, and DX:

- `tqdm` for embedding progress visibility
- `psutil` for adaptive batch sizing from available RAM
- Structured logging utilities in `app/utils/logger.py`
- `pytest` test suite for retrieval, CRAG, metrics, ingestion robustness, and citations

Optional serving:

- `fastapi` + `uvicorn` are included for API exposure of the pipeline

## How the Advanced Hybrid + Corrective Pipeline Works

Pipeline implementation: `app/pipelines/rag_pipeline.py`

1. Query rewrite/expansion
- `QueryRewriter` adds domain-specific synonyms (for example, contraindications -> warnings/precautions) to improve recall.

2. Parallel signal retrieval (dense + sparse)
- Dense retrieval (`DenseRetriever`): searches FAISS embeddings index.
- Sparse retrieval (`SparseRetriever`): BM25 over tokenized chunks.

3. Weighted hybrid fusion
- `HybridRetriever.combine()` merges both result sets by `doc_id` and computes:

```text
Final Score = alpha * Dense + beta * BM25 + gamma * Metadata
```

- Metadata score includes source authority tier, source/page presence, pharma domain tag, and document recency.

4. CRAG validation (corrective quality gate)
- `CRAGValidator.evaluate()` computes retrieval confidence from top score, mean score, and source diversity.
- Labels quality into `CORRECT`, `AMBIGUOUS`, or `INCORRECT`.

5. Corrective behavior when confidence is weak
- If `AMBIGUOUS`: expand queries and merge best results (corrective expansion path).
- If `INCORRECT` and fallback enabled: run fallback query variants and rebuild top results (corrective fallback path).

6. Cross-encoder reranking
- `CrossEncoderReranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default) reranks candidates with pairwise query-document scoring.

7. Structured local generation with citations
- `AnswerGenerator` sends only budget-safe context to Ollama.
- Uses structured response schema (`GeneratedAnswer`) and extracts citations from referenced chunks.

8. Full response payload
- Returns answer, citations, reranked docs, diagnostics (confidence + fallback flags + reasons), uncertainty/conflict notes, and latency.

Why this is advanced:

- Multi-signal retrieval (dense + sparse + metadata) instead of single-vector ranking
- Explicit retrieval confidence modeling before generation
- Automated corrective search loops when retrieval quality is ambiguous/incorrect
- Structured answer generation with citation traceability and uncertainty fields

## Architecture Refactor: From Dual-Generation to Single-Generation RAG

This repository now follows a single-generation architecture when the hybrid controller falls back to RAG.

### BEFORE

- RAG pipeline generated an answer in `app/generation/generator.py`.
- The adapter discarded that generated answer and returned context-only text.
- The hybrid controller generated again using the same context via DoctorLLM/training LLM calls.
- RAG confidence and diagnostics (`answer_confidence`, `uncertainty`, `conflict_notes`, CRAG diagnostics) were not preserved end-to-end.

Flow:

```text
RAG -> context -> Controller -> generate
```

### AFTER

- RAG adapter returns the full `PipelineResponse` from `app/pipelines/rag_pipeline.py`.
- Hybrid controller routes and returns RAG output directly for RAG fallback paths.
- No second controller-side generation is executed for RAG fallback.
- Full RAG output is preserved:
	- `answer`
	- `citations`
	- `answer_confidence`
	- `uncertainty`
	- `conflict_notes`
	- `diagnostics`

Flow:

```text
RAG -> answer -> Controller -> return
```

### BENEFITS

- No duplicate LLM calls for RAG fallback.
- Better grounding and consistency because the returned answer is the same one produced with citation-aware generation.
- Confidence and uncertainty signals are preserved and usable upstream.
- Lower latency and compute cost from eliminating redundant generation.

## Folder-by-Folder Guide (What Each Contains)

```text
knowledge_retrieval_agent/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── correction/
│   │   └── crag.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── generation/
│   │   └── generator.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── indexer.py
│   ├── inspection/
│   │   └── raw_text.py
│   ├── pipelines/
│   │   └── rag_pipeline.py
│   ├── prompts/
│   │   └── prompt_registry.py
│   ├── query/
│   │   └── rewrite.py
│   ├── reranking/
│   │   └── reranker.py
│   ├── retrieval/
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   └── hybrid.py
│   ├── schemas/
│   │   └── models.py
│   └── utils/
│       ├── helpers.py
│       ├── logger.py
│       └── text.py
├── data/
├── tests/
├── vector_store/
├── requirements.txt
└── README.md
```

Detailed responsibilities:

- `app/main.py`
	CLI entrypoint with 3 commands:
	- `ingest`: load files -> chunk -> build FAISS + BM25
	- `query`: run full hybrid + CRAG + rerank + generation pipeline
	- `inspect-raw`: inspect extracted text before embedding/retrieval

- `app/config.py`
	Central typed settings for all runtime controls:
	embedding model, reranker model, LLM endpoint/model, chunking, top-k, fusion weights (`alpha`,`beta`,`gamma`), CRAG thresholds, fallback toggle, ingestion safety knobs.

- `app/ingestion/`
	- `loader.py`: recursive file loading (PDF/PPTX/images/text), OCR fallback, metadata enrichment (`source_tier`, `doc_date`, `domain`).
	- `chunker.py`: splits source docs into retrieval chunks.
	- `embedder.py`: embedding model factory for indexing and retrieval.
	- `indexer.py`: builds FAISS and BM25 artifacts, applies chunk filtering, adaptive batching, disk checks, and logs embedding failures.

- `app/retrieval/`
	- `dense_retriever.py`: FAISS semantic retrieval with normalized dense scores.
	- `sparse_retriever.py`: BM25 lexical retrieval with normalized sparse scores.
	- `hybrid.py`: score fusion + metadata quality scoring.

- `app/correction/crag.py`
	Retrieval confidence scoring and corrective decisions:
	whether to expand query, trigger fallback, and record reasons.

- `app/query/rewrite.py`
	Domain-aware query expansion and fallback query variant generation.

- `app/reranking/reranker.py`
	Cross-encoder reranking of retrieved candidates for final precision.

- `app/generation/generator.py`
	Prompt building, context budget trimming, structured LLM call, citation extraction, fallback answer if generation fails.

- `app/prompts/prompt_registry.py`
	Named prompt templates selected by `prompt_name` config.

- `app/pipelines/rag_pipeline.py`
	Orchestrator that wires every stage into one deterministic end-to-end flow.

- `app/evaluation/metrics.py`
	Retrieval and generation evaluation metrics:
	Precision@K, Recall@K, nDCG@K, reciprocal rank/MRR, answer relevance, context relevance, faithfulness (NLI), hallucination rate, retrieval confidence, latency, batch summaries.

- `app/evaluation/llm_judge/`
	Strict LLM-as-judge evaluator for pharma answers with JSON-only scoring output:
	- prompt template in `prompt_template.py`
	- runtime evaluator in `judge.py`
	- default model: `llama3.2:3b-text-q4_K_M`

- `app/inspection/raw_text.py`
	Utilities to cache and search extracted raw text for debugging ingestion quality.

- `app/schemas/models.py`
	Typed data contracts for chunks, scored docs, diagnostics, pipeline response, generated answer schema.

- `app/utils/`
	- `logger.py`: structured logger setup.
	- `helpers.py`: shared helper utilities (including timing helpers).
	- `text.py`: scientific tokenization helpers used by BM25 and filtering.

- `tests/`
	Automated checks for pipeline behavior, CRAG logic, metrics, metadata filters, citation correctness, ingestion robustness, PPTX loading, and text utilities.

- `data/`
	Source corpus folder (your PDFs/PPTX/images/text files).

- `vector_store/`
	Persisted retrieval artifacts and ingestion traces:
	- `faiss/`: dense index
	- `bm25.pkl`: sparse index
	- `ingest_manifest.json`: change detection to skip unchanged ingestion
	- `raw_extracted.jsonl`: cached extracted raw text
	- `embedding_failures.jsonl`: failed embedding batch records

## Prerequisites

- Python 3.11+

## LLM-as-Judge Quick Use

Run a strict judge evaluation for one QA sample:

python -m app.main judge --mode commercial --question "What is the dosage?" --answer "..." --context "..."

Optional model override:

python -m app.main judge --mode training --question "..." --answer "..." --context "..." --model "llama3.2:3b-text-q4_K_M"
- Ollama installed and running
- Pull local model:

```powershell
ollama pull llama3:8b
```

For scanned PDFs/images, install native Tesseract OCR and ensure it is in PATH.

## Setup

```powershell
cd knowledge_retrieval_agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ingest Documents

Put source files in `data/` (nested folders supported), then run:

```powershell
python -m app.main ingest
```

Generated artifacts:

- `vector_store/faiss/`
- `vector_store/bm25.pkl`
- `vector_store/ingest_manifest.json`
- `vector_store/raw_extracted.jsonl`
- `vector_store/embedding_failures.jsonl` (if failures occur)

## Run Query Pipeline

```powershell
python -m app.main query --text "What are the contraindications of amoxicillin?"
```

Inspect raw extraction before retrieval:

```powershell
python -m app.main inspect-raw --text "composition bactol" --top-k 5
```

Pipeline response includes:

- `answer`
- `citations`
- `retrieved_docs`
- `diagnostics` (confidence, quality label, fallback status, reasons, query variants)
- `answer_confidence`
- `uncertainty`
- `conflict_notes`
- `latency_ms`

## Testing

```powershell
pytest -q
```

## Notes for High-Accuracy Pharma Usage

- Use high-authority curated sources (labels, SMPC, guidelines, peer-reviewed clinical evidence).
- Tune CRAG thresholds (`min_top_score`, `min_mean_score`, `min_source_count`) on validation questions.
- Keep chunk size conservative for safety-critical queries (contraindications/interactions/dose).
- Add stricter faithfulness and contradiction checks for clinical production scenarios.
