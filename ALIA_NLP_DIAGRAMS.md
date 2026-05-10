# ALIA NLP — Diagrams (Repo-Truth)

This file contains **Mermaid diagrams** that are generated strictly from what exists in the repository.  
If your PDF viewer can’t render Mermaid, these diagrams still serve as **executable architecture source**.

---

## 1) Core runtime sequence (end-to-end, real code)

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant API as FastAPI (/chat/message)
  participant NLP as NLP decision layer
  participant RAG as Retrieval (vector + mongo)
  participant RR as Reranker (hybrid lexical)
  participant LLM as Hosted LLM (Groq)
  participant DB as MongoDB

  U->>API: POST /chat/message (content, mode)
  API->>NLP: _analyze_message_nlp(user_text, history, mode)
  alt ALIA_USE_HYBRID_ADAPTER enabled and runtime loads
    NLP->>NLP: backend/utils/hybrid_adapter.py infer_hybrid_adapter()
    NLP->>NLP: merge with baseline NLP analyzer output
  else default
    NLP->>NLP: backend/utils/nlp.py -> NLP/pipeline/nlp.py analyze_message_nlp()
  end

  alt needs clarification (action_items has needs_intent_clarification)
    API-->>U: clarification prompt (no retrieval, no generation)
  else proceed
    API->>RAG: RAGPipeline.get_context(rewritten_query)
    RAG->>RR: rerank_candidates(query, candidates)
    RR-->>RAG: candidates sorted by rerank_score
    RAG-->>API: formatted context
    API->>LLM: Groq chat.completions.create(messages)
    LLM-->>API: final response text
    API->>DB: insert/update conversation + nlp_events
    API-->>U: response text
  end
```

**Evidence**:
- Request handler + orchestration: `backend/routes/chat.py`
- Baseline NLP analyzer: `backend/utils/nlp.py` → `NLP/pipeline/nlp.py`
- Optional hybrid adapter: `backend/utils/hybrid_adapter.py` and `ALIA_USE_HYBRID_ADAPTER` gate in `backend/routes/chat.py`
- RAG + rerank call: `backend/utils/rag_pipeline.py` → `NLP/pipeline/reranker.py`

---

## 2) NLP-only runtime flow (inside `NLP/pipeline/nlp.py`)

```mermaid
flowchart TD
  A["analyze_message_nlp (NLP/pipeline/nlp.py)"] --> B["normalize text via domain_synonyms.json"]
  B --> C["detect_language() (NLP/pipeline/language.py)"]
  C --> D{"Groq client available? (GROQ_API_KEY + groq pkg)"}
  D -- yes --> E["Hosted extraction (Groq chat.completions.create; model=GROQ_MODEL default llama-3.3-70b-versatile)"]
  E --> F["Parse JSON + normalize intent/tags/entity_map"]
  D -- no --> G["_fallback_analysis (rule routing + rule entities)"]
  F --> H["Rule router cross-check (low confidence override)"]
  G --> H
  H --> I["Entity merge (LLM entity_map + rule entity_map + optional pharma v2)"]
  I --> J["Safety merge (_detect_safety_flags; NLP/pipeline/safety.py)"]
  J --> K["Return strict analysis dict (intent, entity_map, safety_flags, rewritten_query, confidence, language, ...)"]
```

---

## 3) Dependency map (NLP + backend call graph)

```mermaid
graph LR
  subgraph Backend
    CHAT[backend/routes/chat.py] --> BNLP[backend/utils/nlp.py]
    CHAT --> HAD[backend/utils/hybrid_adapter.py]
    CHAT --> RAG[backend/utils/rag_pipeline.py]
    RAG --> RERANK[NLP/pipeline/reranker.py]
  end

  subgraph NLP_Runtime
    BNLP --> NLPMAIN[NLP/pipeline/nlp.py]
    NLPMAIN --> LANG[NLP/pipeline/language.py]
    NLPMAIN --> SAFE[NLP/pipeline/safety.py]
    NLPMAIN --> EV2[NLP/pipeline/entity_extractor_v2.py]
  end

  subgraph NLP_Eval_CI
    CI[.github/workflows/nlp-eval.yml] --> EVAL[NLP/evaluation/run_eval.py]
    CI --> CLAR[NLP/evaluation/run_clarification_eval.py]
    CI --> RREVAL[NLP/evaluation/run_retrieval_rerank_eval.py]
    CI --> LEVAL[NLP/evaluation/run_language_detection_eval.py]
    CI --> SHADOW[NLP/evaluation/shadow_monitoring.py]
    CI --> DRIFT[NLP/evaluation/drift_monitoring.py]
  end
```

---

## 4) “Three NLP engines” position diagram (truthful)

```mermaid
flowchart LR
  U[User message] --> D[Decision layer output: JSON analysis]

  D --> RB["Rule-based baseline (always available; NLP/pipeline/nlp.py fallback)"]
  D --> HL["Hosted LLM extraction (optional; Groq in NLP/pipeline/nlp.py)"]
  D --> LA["Local adapter extraction (optional; backend/utils/hybrid_adapter.py)"]

  RB --> MERGE[Merge + constraints]
  HL --> MERGE
  LA --> MERGE

  MERGE --> OUT["analysis dict (intent, entity_map, safety_flags, rewritten_query, confidence, ...)"]
```

