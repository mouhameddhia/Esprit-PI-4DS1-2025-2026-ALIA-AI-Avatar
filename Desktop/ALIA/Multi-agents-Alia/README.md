# Multi-agents Alia

Domain-constrained pharmaceutical assistant platform with a JSON-first policy, corrective hybrid RAG fallback, and dual interaction modes for commercial support and training simulations.

The repository is organized around two cooperating systems:

- knowledge_retrieval_agent: advanced retrieval and grounded generation engine.
- hybrid_medical_agent: orchestration layer that enforces policy, memory, and mode behavior.

## Why This Project Exists

Pharmaceutical assistance requires more than fluent text generation. This platform is designed for safer behavior in regulated contexts by combining deterministic knowledge access and evidence-grounded generation.

Key safeguards:

- JSON cache and curated payloads are prioritized before LLM generation.
- Retrieval uses dense, sparse, and metadata signals.
- CRAG validates retrieval quality before final answer generation.
- Answers include source grounding and diagnostics when RAG is used.
- Off-domain requests are blocked with a fixed pharma-only response.

## Core Capabilities

- Hybrid retrieval: FAISS semantic search + BM25 lexical search + metadata scoring.
- Corrective RAG: retrieval quality classification with corrective expansion/fallback.
- Cross-encoder reranking for final document precision.
- Structured generation with citations, uncertainty, and conflict notes.
- JSON-first response path for product facts.
- Per-user persistent memory and rolling conversation memory.
- Dual operational modes:
	- Commercial Mode: concise factual medical/product support.
	- Training Mode: doctor-style coaching dialogue for medical reps.

## End-to-End Flow

1. User query enters the HybridMedicalController.
2. Controller performs language and intent routing and checks personal-memory queries.
3. JSON retriever attempts direct answer from curated payloads.
4. If JSON misses, RAG pipeline runs:
	 - Query rewrite
	 - Dense retrieval (FAISS)
	 - Sparse retrieval (BM25)
	 - Hybrid score fusion
	 - CRAG quality evaluation and corrective behavior
	 - Cross-encoder reranking
	 - Context budgeting and grounded LLM generation
5. Response is returned with source metadata and memory is updated.

## Repository Structure

```text
.
├── hybrid_medical_agent/
│   ├── streamlit_app.py
│   ├── agent/
│   │   ├── controller.py
│   │   ├── json_retriever.py
│   │   ├── rag_adapter.py
│   │   ├── persistent_memory.py
│   │   └── training_brain.py
│   ├── scripts/
│   │   └── run_agent.py
│   └── storage/
├── knowledge_retrieval_agent/
│   ├── app/
│   │   ├── main.py
│   │   ├── pipelines/rag_pipeline.py
│   │   ├── retrieval/
│   │   ├── correction/
│   │   ├── reranking/
│   │   ├── generation/
│   │   └── evaluation/
│   ├── data/
│   ├── tests/
│   ├── vector_store/
│   └── requirements.txt
├── rag_knowledge_builder/
├── ARCHITECTURE_FLOW.md
├── TECHNICAL_DOCUMENTATION.md
└── README.md
```

## Prerequisites

- Python 3.11+
- Ollama running locally (default URL: http://localhost:11434)
- At least one local Ollama model pulled (default: llama3:8b)
- Tesseract OCR installed and available in PATH (needed for OCR fallback)

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r knowledge_retrieval_agent/requirements.txt
pip install streamlit
```

Start Ollama in a separate terminal, then pull the model if needed:

```powershell
ollama pull llama3:8b
```

## Quick Start

### 1) Build Retrieval Indexes

```powershell
cd knowledge_retrieval_agent
python -m app.main ingest
```

What this does:

- Loads source documents from knowledge_retrieval_agent/data.
- Extracts and chunks text.
- Builds FAISS and BM25 artifacts in knowledge_retrieval_agent/vector_store.

### 2) Run a RAG Query (CLI)

```powershell
cd knowledge_retrieval_agent
python -m app.main query --text "What are the indications of BACTOL?"
```

### 3) Launch the Streamlit Hybrid Assistant

```powershell
cd ..
python -m streamlit run hybrid_medical_agent/streamlit_app.py
```

In the sidebar you can select:

- Mode: Training Mode or Commercial Mode
- Response language: auto, English, or French
- Authenticated user ID for memory namespace
- Optional product scope

## Main Operational Modes

### Commercial Mode

- Prioritizes direct factual answers.
- Enforces pharmaceutical domain restriction.
- Uses JSON-first, then RAG fallback if needed.

### Training Mode

- Simulates doctor-rep coaching conversations.
- Tracks locked product and conversation phase.
- Uses memory and turn-level context for continuity.

## Command Reference

### Knowledge Retrieval Agent CLI

Run from knowledge_retrieval_agent:

```powershell
python -m app.main ingest
python -m app.main query --text "Your question"
python -m app.main inspect-raw --text "search phrase" --top-k 5
python -m app.main judge --mode commercial --question "..." --answer "..." --context "..."
python -m app.main xai-report --question "..." --answer "..." --context "..."
```

### Hybrid Agent CLI

Run from repository root:

```powershell
python hybrid_medical_agent/scripts/run_agent.py --question "What is the dosage of BACTOL?" --user-id demo_user
python hybrid_medical_agent/scripts/run_agent.py --training --drug BACTOL --user-id demo_user
python hybrid_medical_agent/scripts/run_agent.py --training --rep-answer "..." --user-id demo_user
```

## Data, Knowledge, and Memory

- Curated JSON payloads:
	- rag_knowledge_builder/scripts/incoming_payload_*.json
	- rag_knowledge_builder/scripts/*.json
- Retrieval source documents:
	- knowledge_retrieval_agent/data/
- Vector artifacts:
	- knowledge_retrieval_agent/vector_store/
- Per-user memory namespace:
	- memory/users/<user_id>/

## Testing

Run tests from knowledge_retrieval_agent:

```powershell
cd knowledge_retrieval_agent
pytest -q
```

## Troubleshooting

### No answer quality or weak grounding

- Re-run ingestion after updating documents.
- Verify vector artifacts exist in knowledge_retrieval_agent/vector_store.
 - Validate that relevant product JSON exists in rag_knowledge_builder/scripts.

### Streamlit starts but replies fail

- Confirm Ollama is running and model is available.
- Check the configured LLM base URL in Streamlit sidebar.
- Ensure Python environment has both requirements and streamlit installed.

### OCR content missing

- Ensure Tesseract is installed and accessible via PATH.
- Re-run ingestion to refresh extracted raw cache.

## Additional Documentation

- Technical deep dive: TECHNICAL_DOCUMENTATION.md
- Runtime flow diagram: ARCHITECTURE_FLOW.md
- Memory architecture: hybrid_medical_agent/MEMORY_ARCHITECTURE.md

## Design Principles

- Deterministic first, probabilistic second.
- Retrieval confidence before generation.
- Evidence-grounded answers over stylistic fluency.
- Strict domain specialization for safer behavior.
