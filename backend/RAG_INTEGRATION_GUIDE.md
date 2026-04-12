"""
Integration Guide: How to use RAG in chat.py

This file shows the specific changes needed to integrate RAG (Retrieval-Augmented Generation)
into your existing chat endpoints.
"""

# ============================================================================
# OPTION 1: MINIMAL INTEGRATION (Recommended for MVP)
# ============================================================================

"""
Place this at the top of routes/chat.py:

from ..utils.rag_pipeline import RAGPipeline
from ..dependencies import get_vector_client, get_embedding_encoder
"""

# Modify the send_message endpoint like this:

async def send_message(
    request: SendMessageRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_user),
    # NEW: Add vector DB dependencies
    vector_client = Depends(get_vector_client),
    encoder = Depends(get_embedding_encoder),
) -> SendMessageResponse:
    """Send a message and get AI response with product context."""
    
    # Existing code to get/create session...
    # (keep your existing session management code)
    
    # NEW: Get relevant product context
    context = ""
    try:
        rag = RAGPipeline(vector_client, encoder, db)
        context = await rag.get_context(
            query=request.content,
            top_k=3,
            min_score=0.5  # Adjust based on your needs
        )
    except Exception as e:
        logger.warning(f"Failed to retrieve RAG context: {e}")
        # Continue without context if RAG fails
    
    # Build messages with context
    system_prompt = SYSTEM_PROMPTS[request.mode]
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add context as system message if available
    if context:
        messages.append({
            "role": "system",
            "content": context
        })
    
    # Add conversation history (existing code)
    messages.extend([
        {"role": msg.role, "content": msg.content}
        for msg in conversation.messages
    ])
    
    # Add current user message
    messages.append({"role": "user", "content": request.content})
    
    # Get response from Groq (existing code)
    reply = _chat_completion(messages)
    
    # Save conversation and return (existing code)
    # ... rest of your implementation
    
    return SendMessageResponse(
        session_id=str(conversation["_id"]),
        reply=reply
    )


# ============================================================================
# OPTION 2: ADVANCED INTEGRATION (With conversation embedding)
# ============================================================================

"""
This version also embeds conversations for future search.
"""

from ..utils.conversation_embeddings import ConversationEmbedder
from ..dependencies import get_conversation_embedder

async def send_message(
    request: SendMessageRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_user),
    vector_client = Depends(get_vector_client),
    encoder = Depends(get_embedding_encoder),
    conversation_embedder = Depends(get_conversation_embedder),
) -> SendMessageResponse:
    """Send a message with RAG and conversation indexing."""
    
    # ... existing code ...
    
    # Get context
    context = ""
    try:
        rag = RAGPipeline(vector_client, encoder, db)
        context = await rag.get_context(request.content, top_k=3)
    except Exception as e:
        logger.warning(f"RAG context retrieval failed: {e}")
    
    # Build and send messages (as in Option 1)
    messages = [...]
    reply = _chat_completion(messages)
    
    # NEW: Embed the conversation for future search
    try:
        await conversation_embedder.embed_conversation(db, str(conversation["_id"]))
    except Exception as e:
        logger.warning(f"Failed to embed conversation: {e}")
        # Don't fail the chat just because embedding failed
    
    # ... existing code to save and return ...
    
    return SendMessageResponse(
        session_id=str(conversation["_id"]),
        reply=reply
    )


# ============================================================================
# OPTION 3: WITH SEMANTIC SEARCH ENDPOINT
# ============================================================================

"""
Add a new endpoint to search past conversations.
"""

@router.get("/search-conversations")
async def search_conversations(
    query: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_user),
    vector_client = Depends(get_vector_client),
    encoder = Depends(get_embedding_encoder),
):
    """
    Search past conversations using semantic similarity.
    
    Useful for:
    - Finding similar training scenarios
    - Comparing how different physicians asked questions
    - Identifying common objections
    
    Example usage:
    GET /chat/search-conversations?query=cardiac%20side%20effects
    """
    
    try:
        rag = RAGPipeline(vector_client, encoder, db)
        
        # Search for related conversations
        related_convs = await rag.get_related_conversations(query, top_k=5)
        
        return {
            "query": query,
            "results": related_convs,
            "count": len(related_convs)
        }
    except Exception as e:
        logger.error(f"Conversation search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )


# ============================================================================
# CONFIG: Context Window Management
# ============================================================================

"""
If you want to manage context more aggressively to fit within token limits:
"""

async def manage_context_window(
    messages: List[dict],
    max_tokens: int = 4000,
    encoder = None
) -> List[dict]:
    """
    Intelligently trim conversation history using embeddings.
    Keeps the most relevant messages.
    """
    
    # Calculate tokens (rough estimate: 1 token ≈ 4 chars)
    current_tokens = sum(len(m["content"]) for m in messages) // 4
    
    if current_tokens <= max_tokens:
        return messages
    
    # Keep system messages (first 2)
    kept_messages = messages[:2]
    remaining_messages = messages[2:]
    
    # If encoder available, use semantic similarity to keep relevant messages
    if encoder and encoder.is_ready():
        # Embed the last user message
        last_user_msg = next(
            (m for m in reversed(remaining_messages) if m["role"] == "user"),
            None
        )
        
        if last_user_msg:
            last_embedding = encoder.encode(last_user_msg["content"])
            
            # Score remaining messages by similarity
            scored = []
            for msg in remaining_messages:
                if msg["role"] == "user":  # Prioritize user messages
                    emb = encoder.encode(msg["content"])
                    score = encoder.similarity(emb, last_embedding)
                    scored.append((score, msg))
            
            # Sort by score
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Add top scoring messages until we hit limit
            for score, msg in scored:
                msg_tokens = len(msg["content"]) // 4
                if sum(len(m["content"]) for m in kept_messages) // 4 + msg_tokens <= max_tokens:
                    kept_messages.append(msg)
    
    # Fallback: keep last N messages
    if len(kept_messages) == 2:
        allowed_tokens = max_tokens - (len(kept_messages[0]["content"]) + len(kept_messages[1]["content"])) // 4
        for msg in remaining_messages[-10:]:  # Keep last 10 messages
            msg_tokens = len(msg["content"]) // 4
            if allowed_tokens >= msg_tokens:
                kept_messages.append(msg)
                allowed_tokens -= msg_tokens
    
    return kept_messages


# ============================================================================
# TESTING
# ============================================================================

"""
Test your integration with curl or your HTTP client:

# Test single message with context
curl -X POST http://localhost:8000/chat/send-message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "session_id": null,
    "content": "What are the indications for Product X??",
    "mode": "physician_portal"
  }'

# Search conversations
curl -X GET "http://localhost:8000/chat/search-conversations?query=cardiac%20side%20effects" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Check vector DB status (admin)
curl -X GET http://localhost:8000/admin/vector-db-status \
  -H "Authorization: Bearer ADMIN_TOKEN"
"""

# ============================================================================
# DEPLOYMENT CHECKLIST
# ============================================================================

"""
Before deploying to production:

[ ] Test with actual product data
[ ] Verify Pinecone index is indexed (run /admin/reindex-products)
[ ] Check latency impact of RAG queries
[ ] Monitor token usage on Groq API
[ ] Set appropriate min_score threshold
[ ] Configure logging to monitor failures
[ ] Have fallback handling if Vector DB fails
[ ] Test with various queries and edge cases
[ ] Document any custom configurations in .env
"""
