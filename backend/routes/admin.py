"""Admin routes for managing vector database and embeddings."""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..dependencies import (
    get_database,
    get_vector_client,
    get_embedding_encoder,
    get_product_indexer,
    get_knowledge_document_indexer,
    get_conversation_embedder,
    get_current_user
)
from ..models.user import UserInDB
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class IndexingResponse(BaseModel):
    success: bool
    message: str
    details: dict = {}


@router.post("/admin/reindex-products", response_model=IndexingResponse)
async def reindex_products(
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    product_indexer = Depends(get_product_indexer)
):
    """
    Reindex all products into the vector database.
    Requires admin role.
    """
    try:
        result = await product_indexer.index_products(db)
        
        if result.get('success'):
            return IndexingResponse(
                success=True,
                message=f"Successfully indexed {result.get('indexed_count')} products",
                details=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Indexing failed')
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during indexing: {str(e)}"
        )


@router.post("/admin/reindex-conversations", response_model=IndexingResponse)
async def reindex_conversations(
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    conversation_embedder = Depends(get_conversation_embedder)
):
    """
    Reindex all conversations into the vector database.
    Requires admin role.
    """
    try:
        result = await conversation_embedder.reindex_all_conversations(db)
        
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error')
            )
        
        return IndexingResponse(
            success=True,
            message=f"Successfully embedded {result.get('successful_embeddings')} conversations",
            details=result
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during reindexing: {str(e)}"
        )


@router.post("/admin/reindex-knowledge-documents", response_model=IndexingResponse)
async def reindex_knowledge_documents(
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    knowledge_document_indexer = Depends(get_knowledge_document_indexer),
):
    """
    Reindex imported knowledge documents into the vector database.
    Requires admin role.
    """
    try:
        result = await knowledge_document_indexer.index_documents(db)

        if result.get('success'):
            return IndexingResponse(
                success=True,
                message=f"Successfully indexed {result.get('indexed_count')} knowledge document chunks",
                details=result,
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get('error', 'Knowledge document indexing failed'),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during knowledge document indexing: {str(e)}",
        )


@router.get("/admin/vector-db-status")
async def get_vector_db_status(
    current_user: UserInDB = Depends(get_current_user),
    vector_client = Depends(get_vector_client),
    embedding_encoder = Depends(get_embedding_encoder)
):
    """
    Get status of vector database and embedding system.
    Requires admin role.
    """
    return {
        "vector_db_ready": vector_client.is_ready(),
        "vector_db_type": vector_client.db_type,
        "embedding_encoder_ready": embedding_encoder.is_ready(),
        "embedding_model": embedding_encoder.model_name,
        "embedding_dimension": embedding_encoder.embedding_dim,
        "index_stats": await vector_client.get_index_stats() if vector_client.is_ready() else {}
    }


@router.post("/admin/embed-conversation/{conversation_id}", response_model=IndexingResponse)
async def embed_single_conversation(
    conversation_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    conversation_embedder = Depends(get_conversation_embedder)
):
    """
    Embed a single conversation.
    Requires admin role.
    """
    try:
        success = await conversation_embedder.embed_conversation(db, conversation_id)
        
        if success:
            return IndexingResponse(
                success=True,
                message=f"Successfully embedded conversation {conversation_id}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to embed conversation"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


@router.get("/admin/conversation-insights/{conversation_id}")
async def get_conversation_insights(
    conversation_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    conversation_embedder = Depends(get_conversation_embedder)
):
    """
    Get AI insights from a conversation using embeddings.
    Requires admin role.
    """
    try:
        insights = await conversation_embedder.get_conversation_insights(db, conversation_id)
        
        if 'error' in insights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=insights.get('error')
            )
        
        return insights
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )
