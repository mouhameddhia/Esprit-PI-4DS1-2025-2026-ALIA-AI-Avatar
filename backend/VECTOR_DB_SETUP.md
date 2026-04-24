# Vector Database & Embeddings Integration Guide

This document provides a complete overview of the RAG (Retrieval-Augmented Generation) system integrated into ALIA.

## Architecture Overview

The system consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                    ALIA Backend                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │  Chat Routes     │      │  Admin Routes    │           │
│  └────────┬─────────┘      └────────┬─────────┘           │
│           │                          │                    │
│           └──────────────┬───────────┘                    │
│                          │                                │
│                   ┌──────▼──────┐                         │
│                   │ RAG Pipeline │                        │
│                   └──────┬───────┘                        │
│                          │                                │
│        ┌─────────────────┼─────────────────┐              │
│        │                 │                 │              │
│   ┌────▼────┐    ┌──────▼──────┐   ┌─────▼────┐         │
│   │Embedding │    │ Vector DB   │   │ MongoDB  │         │
│   │ Encoder  │    │ (Pinecone)  │   │          │         │
│   └──────────┘    └─────────────┘   └──────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
backend/
├── vector_db/
│   ├── __init__.py
│   ├── client.py              # Pinecone client wrapper
│   └── indexing.py            # ProductIndexer class
│
├── embeddings/
│   ├── __init__.py
│   └── encoder.py             # EmbeddingEncoder (sentence-transformers)
│
├── routes/
│   ├── admin.py               # Admin endpoints for management
│   ├── chat.py                # Chat routes (to be enhanced with RAG)
│   └── auth.py
│
├── utils/
│   ├── rag_pipeline.py        # RAGPipeline for context retrieval
│   ├── conversation_embeddings.py  # ConversationEmbedder
│   ├── summary.py             # Existing summarization
│   └── ...
│
└── tests/
    └── test_embeddings_setup.py  # Verification tests
```

## Setup Instructions

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Key new dependencies:
- `sentence-transformers` (2.2.2) - For embeddings
- `pinecone-client` (3.0.0) - For vector database
- `numpy` (1.24.0) - Numerical operations

### Step 2: Configure Environment Variables

Add these to `.env` file in the `backend/` directory:

```env
# Vector Database & Embeddings
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your_actual_api_key_here
PINECONE_INDEX_NAME=alia-knowledge
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

Get your Pinecone API key from: https://www.pinecone.io/

### Step 3: Create Pinecone Index

In Pinecone dashboard:
1. Create a new index named `alia-knowledge`
2. Set dimensions to `384` (matches the embedding model)
3. Use `cosine` as similarity metric

Or use the Pinecone CLI:
```bash
pinecone index create --name alia-knowledge --dimension 384 --metric cosine
```

### Step 4: Verify Installation

```bash
# Run from backend directory
python tests/test_embeddings_setup.py
```

Expected output:
```
✓ Encoder initialized successfully
✓ Single text encoded successfully
✓ Batch encoding successful
✓ Vector Database client initialized
```

### Step 5: Start the Backend

```bash
uvicorn main:app --reload
```

On startup, the system will:
1. Initialize the embedding encoder
2. Connect to Pinecone
3. **Automatically index all products** into the vector database
4. Log progress and any warnings

## Usage

### For Physicians/Med Reps - Using RAG in Chat

When a user sends a message, the system now:

1. **Encodes the message** into a 384-dimensional embedding
2. **Searches Pinecone** for similar products
3. **Retrieves full product info** from MongoDB
4. **Passes context to Groq LLM** for more accurate responses
5. **Generates response** with proper product information

**Example:**
```
User: "What are the indications for Product X?"

System flow:
- Encodes query: "What are the indications for Product X?"
- Finds similar products in vector DB
- Retrieves Product X data from MongoDB
- Sends to Groq with context
- Returns accurate, evidence-based response
```

### Admin Endpoints

Access admin functionality via these endpoints:

#### Reindex All Products
```bash
POST /admin/reindex-products
Authorization: Bearer <admin_token>
```

Response:
```json
{
  "success": true,
  "message": "Successfully indexed 42 products",
  "details": {
    "indexed_count": 42,
    "total_products": 42,
    "products": [...]
  }
}
```

#### Reindex All Conversations
```bash
POST /admin/reindex-conversations
Authorization: Bearer <admin_token>
```

#### Get Vector DB Status
```bash
GET /admin/vector-db-status
Authorization: Bearer <admin_token>
```

Response:
```json
{
  "vector_db_ready": true,
  "vector_db_type": "pinecone",
  "embedding_encoder_ready": true,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "index_stats": {
    "total_vector_count": 42,
    "dimension": 384,
    "index_fullness": 0.001
  }
}
```

#### Get Conversation Insights
```bash
GET /admin/conversation-insights/{conversation_id}
Authorization: Bearer <admin_token>
```

## Module Details

### VectorDBClient (`vector_db/client.py`)

Handles all interactions with Pinecone.

**Key Methods:**
- `is_ready()` - Check if connected to vector DB
- `upsert(vectors)` - Insert/update vectors
- `search(query_vector, top_k)` - Semantic search
- `delete(ids)` - Remove vectors
- `get_index_stats()` - Get index information

**Example:**
```python
from vector_db import VectorDBClient

client = VectorDBClient()

# Search for similar products
results = await client.search(
    query_vector=embedding,
    top_k=5,
    include_metadata=True
)
```

### EmbeddingEncoder (`embeddings/encoder.py`)

Generates embeddings using sentence-transformers.

**Key Methods:**
- `encode(text)` - Single text to embedding
- `encode_batch(texts)` - Multiple texts efficiently
- `similarity(emb1, emb2)` - Calculate cosine similarity
- `get_embedding_dimension()` - Get embedding size

**Example:**
```python
from embeddings import EmbeddingEncoder

encoder = EmbeddingEncoder()

# Encode single text
embedding = encoder.encode("Product information")

# Encode batch
embeddings = encoder.encode_batch([
    "Drug A for cancer",
    "Drug B for heart disease"
])

# Compare similarity
similarity = encoder.similarity(embeddings[0], embeddings[1])
```

### RAGPipeline (`utils/rag_pipeline.py`)

Orchestrates retrieval-augmented generation.

**Key Methods:**
- `get_context(query, top_k, min_score)` - Get relevant product context
- `get_related_conversations(query, top_k)` - Find similar past conversations
- `extract_topics_from_query(query)` - Extract keywords

**Example:**
```python
from utils.rag_pipeline import RAGPipeline

rag = RAGPipeline(vector_client, encoder, db)

# Get context for a query
context = await rag.get_context("Treatment options for hypertension")

# Returns formatted string:
# """
# RELEVANT PRODUCT INFORMATION:
# 
# ### Product Name (Relevance: 0.92)
# Category: Cardiovascular
# Description: ...
# Indications: ...
# """
```

### ProductIndexer (`vector_db/indexing.py`)

Manages indexing of product database.

**Key Methods:**
- `index_products(db)` - Index all products (called at startup)
- `reindex_product(db, product_id)` - Update single product
- `delete_product_index(product_id)` - Remove product

**Vector Format:**
```python
{
    'id': 'product_<mongo_id>',
    'values': [0.1, 0.2, ...],  # 384-dim embedding
    'metadata': {
        'type': 'product',
        'product_id': '<mongo_id>',
        'name': 'Product Name',
        'category': 'Drug Category'
    }
}
```

### ConversationEmbedder (`utils/conversation_embeddings.py`)

Handles conversation embedding for semantic search.

**Key Methods:**
- `embed_conversation(db, conv_id)` - Embed a conversation
- `embed_batch_conversations(db, conv_ids)` - Embed multiple
- `reindex_all_conversations(db)` - Full reindex
- `get_conversation_insights(db, conv_id)` - Extract insights

## Integration with Chat Routes

To enable RAG in chat responses, modify `routes/chat.py`:

```python
from utils.rag_pipeline import RAGPipeline

@router.post("/send_message")
async def send_message(
    request: SendMessageRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    vector_client = Depends(get_vector_client),
    encoder = Depends(get_embedding_encoder)
):
    # NEW: Get relevant product context
    rag = RAGPipeline(vector_client, encoder, db)
    context = await rag.get_context(request.content, top_k=3)
    
    # Build messages with context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[request.mode]},
        {"role": "system", "content": context},  # Add context
        # ... existing messages ...
    ]
    
    # Call Groq API as before
    reply = _chat_completion(messages)
    
    # Optionally embed the conversation
    # await conversation_embedder.embed_conversation(db, session_id)
```

## Performance Considerations

### Embedding Model Selection

The default model `all-MiniLM-L6-v2` provides:
- **Fast** inference (minimal latency)
- **384D** embeddings (small vector size)
- **Good quality** for general semantic search

For better quality (slower), consider:
- `all-mpnet-base-v2` (768D) - Better accuracy, slower
- `all-roberta-large-v1` (1024D) - Best accuracy, much slower

### Batch Processing

Use `encode_batch()` for multiple texts:
```python
# Good - efficient
embeddings = encoder.encode_batch(texts, batch_size=32)

# Bad - slow
embeddings = [encoder.encode(t) for t in texts]
```

### Pinecone Pricing

Free tier provides:
- 1 project
- 1 index
- 100,000 vectors
- 1GB storage

Monitor your index:
```bash
GET /admin/vector-db-status
```

### Search Thresholds

Adjust `min_score` in RAGPipeline for relevance:
- `min_score=0.3` - More results, potentially less relevant
- `min_score=0.5` - Balanced
- `min_score=0.8` - High precision, fewer results

## Troubleshooting

### Vector DB Not Initialized

**Error:** "Vector database not ready"

**Solution:**
1. Check `.env` file for `PINECONE_API_KEY`
2. Verify Pinecone index exists
3. Check internet connection
4. Review startup logs

### Embedding Model Not Loading

**Error:** "Embedding encoder not ready"

**Solution:**
```bash
pip install sentence-transformers
python -m sentence_transformers.download_models
```

### Low Search Quality

**Causes:**
1. Model mismatch (ensure consistent embedding model)
2. Insufficient product data
3. Poor query formulation

**Solutions:**
1. Reindex products: `POST /admin/reindex-products`
2. Add more detailed product descriptions
3. Consider better embedding model (see Performance section)

### Memory Issues

If experiencing memory errors:
1. Reduce batch size: `encoder.encode_batch(texts, batch_size=8)`
2. Use smaller embedding model
3. Implement incremental indexing

## Next Steps

1. **Test the setup** with `python tests/test_embeddings_setup.py`
2. **Integrate RAG in chat routes** (see Integration section)
3. **Monitor performance** via admin endpoints
4. **Fine-tune thresholds** based on your use case
5. **Add conversation indexing** for historical search

## References

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FastAPI Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [RAG Pattern](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html)
