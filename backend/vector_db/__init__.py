"""Vector database module for semantic search and embeddings."""

from .client import VectorDBClient
from .indexing import ProductIndexer
from .knowledge_indexing import KnowledgeDocumentIndexer

__all__ = ["VectorDBClient", "ProductIndexer", "KnowledgeDocumentIndexer"]
