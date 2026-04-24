"""Vector database client for Pinecone integration."""

import logging
from typing import Any, Dict, List, Optional

from .. import config

logger = logging.getLogger(__name__)


class VectorDBClient:
    """Handles vector database operations with Pinecone."""

    def __init__(self):
        self.db_type = config.VECTOR_DB_TYPE
        self.api_key = config.PINECONE_API_KEY
        self.index_name = config.PINECONE_INDEX_NAME
        self.client = None
        self.index = None
        
        if not self.api_key:
            logger.warning("PINECONE_API_KEY not set. Vector database operations will fail.")
            return
        
        self._init_client()

    def _init_client(self):
        """Initialize Pinecone client and connect to index."""
        try:
            from pinecone import Pinecone
            
            self.client = Pinecone(api_key=self.api_key)
            self.index = self.client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except ImportError:
            logger.error("Pinecone library not installed. Run: pip install pinecone-client")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")

    def is_ready(self) -> bool:
        """Check if vector database is ready for operations."""
        return self.client is not None and self.index is not None

    async def upsert(self, vectors: List[Dict[str, Any]]) -> bool:
        """
        Upsert vectors into the index.
        
        Args:
            vectors: List of dicts with 'id', 'values' (embedding), and optional 'metadata'
                    Example: [
                        {
                            'id': 'product_1',
                            'values': [0.1, 0.2, ...],
                            'metadata': {'name': 'Product A', 'category': 'Drug'}
                        }
                    ]
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_ready():
            logger.error("Vector database not initialized")
            return False
        
        try:
            # Pinecone's upsert expects tuples of (id, values, metadata)
            upsert_data = [
                (v['id'], v['values'], v.get('metadata', {}))
                for v in vectors
            ]
            
            self.index.upsert(vectors=upsert_data, namespace="")
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False

    async def upsert_batched(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Upsert vectors in fixed-size batches.

        Returns the total number of vectors successfully upserted.
        Logs an error for each failing batch but continues with the rest.
        """
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            if await self.upsert(batch):
                total_upserted += len(batch)
            else:
                logger.error(
                    "Failed to upsert batch starting at index %d (%d vectors)",
                    i,
                    len(batch),
                )
        return total_upserted

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of top results to return
            include_metadata: Whether to include metadata in results
            filter_dict: Optional metadata filter
        
        Returns:
            List of results with format:
            [
                {
                    'id': 'product_1',
                    'score': 0.95,
                    'metadata': {...}
                }
            ]
        """
        if not self.is_ready():
            logger.error("Vector database not initialized")
            return []
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = [
                {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata if include_metadata else {}
                }
                for match in results.matches
            ]
            
            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_ready():
            logger.error("Vector database not initialized")
            return False
        
        try:
            self.index.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_ready():
            logger.error("Vector database not initialized")
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'namespaces': stats.get('namespaces', {}),
                'dimension': stats.get('dimension'),
                'index_fullness': stats.get('index_fullness'),
                'total_vector_count': stats.get('total_vector_count')
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}

    def close(self):
        """Close the vector database connection."""
        if self.client:
            logger.info("Closing Pinecone connection")
            self.client = None
            self.index = None
