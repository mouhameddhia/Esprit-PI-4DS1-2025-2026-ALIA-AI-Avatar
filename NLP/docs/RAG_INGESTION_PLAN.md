# RAG Ingestion Plan for ALIA

This plan converts the teacher-provided files into searchable, retrievable knowledge sources for the chatbot and training simulator.

## 1. Source Files

- `useful-files/Manuel_FINAL_ALIA_AVATAR_VM_VITAL_V1.pdf`
- `useful-files/Référentiel des 4 niveaux de compétence - AVATAR ALIA.pdf`
- `useful-files/TECHNIQUES DE VENTES VF (2).pdf`
- `useful-files/Matrice_progression_ALIA_V1.xlsx`

## 2. Ingestion Goals

The retrieval system should answer:

- how to structure a visit
- how to handle objections
- how to adapt to medical profiles
- how to score competency levels
- how to plan follow-up and CRM actions
- how to run a simulation or training scenario

## 3. Recommended Knowledge Collections

Create separate collections or document types.

### 3.1 Training Manual Collection

Content:

- 6-step visit process
- visit formats
- objection-handling method
- argumentation structure
- CRM and follow-up rules

### 3.2 Competency Reference Collection

Content:

- beginner / junior / confirmed / expert definitions
- limits and capabilities per level
- KPIs and criteria for success

### 3.3 Sales Method Collection

Content:

- visit preparation
- discovery questions
- objection handling
- conclusion and engagement
- general sales method concepts

### 3.4 Progression Matrix Collection

Content:

- checkpoints
- scoring thresholds
- checklist items
- pass/fail conditions

## 4. Chunking Strategy

Use semantic chunking, not naive fixed-size splitting.

### 4.1 PDF Chunks

- chunk by section and sub-section
- keep headings with the body text
- preserve labels like `Etape 1`, `Objection`, `KPI`, `Seuil`
- target about 300 to 800 words per chunk

### 4.2 Table and Matrix Chunks

- ingest each competency row as one structured chunk
- keep the level columns intact
- preserve thresholds and metrics as metadata

### 4.3 Metadata per Chunk

Add metadata fields like:

- `source_name`
- `source_type`
- `document_title`
- `section_title`
- `subsection_title`
- `page_number`
- `competency_level`
- `visit_phase`
- `topic_tags`
- `language`
- `chunk_index`

## 5. Suggested Document Schema

Each chunk should be stored in MongoDB and vector DB with fields like:

```json
{
  "source_name": "Manuel FINAL ALIA",
  "source_type": "manual",
  "document_title": "MANUEL FINAL ALIA",
  "section_title": "4.2 Etape 2 - SONDAGE",
  "page_number": 4,
  "language": "fr",
  "chunk_text": "Objectif: comprendre le besoin avant d'argumenter...",
  "topic_tags": ["discovery", "questions", "listening"],
  "visit_phase": "discovery_sondage"
}
```

## 6. Retrieval Use Cases

### 6.1 Chat Answering

Retrieve the most relevant training rule or reference when the user asks:

- how to handle a specific objection
- how to structure a visit
- what level a persona corresponds to
- how to close a visit

### 6.2 Simulation Mode

Retrieve the exact phase or competency rule needed to drive a scenario.

### 6.3 Evaluation Mode

Retrieve the level criteria and scoring thresholds to judge a conversation.

## 7. Retrieval Filters

At query time, filter by:

- `visit_phase`
- `competency_level`
- `source_type`
- `topic_tags`
- `language`

## 8. Ranking Strategy

Ranking should prioritize:

1. exact section match
2. competency or visit-phase match
3. semantic similarity
4. recency or operational priority if needed

If two chunks are similar, prefer the one with the clearest operational rule.

## 9. Ingestion Workflow

1. Extract text from PDF/Excel.
2. Normalize and clean the text.
3. Split by section or row.
4. Attach metadata.
5. Generate embeddings.
6. Store in MongoDB.
7. Upsert into vector DB.
8. Validate a sample of retrieval queries.

## 10. Quality Checks

- ensure headings are not lost during chunking
- ensure thresholds remain attached to the right level
- verify that objections retrieve the objection-handling sections
- verify that competency questions retrieve the matrix and rubric
- verify that visit-format questions retrieve the correct format section
