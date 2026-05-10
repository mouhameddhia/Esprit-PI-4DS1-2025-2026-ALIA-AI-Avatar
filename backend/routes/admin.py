"""Admin routes for managing vector database and embeddings."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from ..dependencies import (
    get_database,
    get_vector_client,
    get_embedding_encoder,
    get_product_indexer,
    get_knowledge_document_indexer,
    get_conversation_embedder,
    require_roles,
)
from ..models.user import UserInDB

router = APIRouter()

_admin = Depends(require_roles("admin"))


class IndexingResponse(BaseModel):
    success: bool
    message: str
    details: dict = {}


@router.post("/admin/reindex-products", response_model=IndexingResponse)
async def reindex_products(
    current_user: UserInDB = _admin,
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
    current_user: UserInDB = _admin,
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
    current_user: UserInDB = _admin,
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
    current_user: UserInDB = _admin,
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
    current_user: UserInDB = _admin,
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


@router.get("/admin/metrics")
async def get_metrics(
    limit: int = Query(default=30, ge=1, le=90),
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Return the latest shadow monitoring snapshot plus a rolling history.
    Requires admin role.
    """
    snapshots = await db.shadow_monitoring.find(
        {}, {"_id": 0}
    ).sort("generated_at", -1).limit(limit).to_list(limit)

    if not snapshots:
        return {
            "latest": None,
            "history": [],
            "summary": {"total_snapshots": 0, "avg_divergence_rate": None, "gate_pass_rate": None},
        }

    latest = snapshots[0]
    total = len(snapshots)
    avg_divergence = round(
        sum(s.get("result", {}).get("divergence_rate", 0.0) for s in snapshots) / total, 4
    )
    gate_pass_rate = round(
        sum(1 for s in snapshots if s.get("quality_gate") == "pass") / total, 4
    )

    return {
        "latest": latest,
        "history": snapshots,
        "summary": {
            "total_snapshots": total,
            "avg_divergence_rate": avg_divergence,
            "gate_pass_rate": gate_pass_rate,
        },
    }


@router.post("/admin/metrics/trigger-snapshot")
async def trigger_shadow_snapshot(
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Manually trigger a shadow monitoring snapshot outside the nightly schedule.
    Requires admin role.
    """
    from ..utils.background_tasks import auto_generate_shadow_monitoring_snapshot

    try:
        result = await auto_generate_shadow_monitoring_snapshot(db)
        return {"success": True, "snapshot": result}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Snapshot failed: {exc}",
        )


@router.get("/admin/conversation-insights/{conversation_id}")
async def get_conversation_insights(
    conversation_id: str,
    current_user: UserInDB = _admin,
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
