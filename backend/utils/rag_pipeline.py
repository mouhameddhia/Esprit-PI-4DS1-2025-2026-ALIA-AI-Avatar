"""RAG (Retrieval-Augmented Generation) pipeline for context-aware responses."""

import logging
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from NLP.pipeline.reranker import rerank_candidates

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates retrieval-augmented generation:
    1. Encode user query
    2. Search vector database for relevant products
    3. Fetch full product info from MongoDB
    4. Format as context for LLM
    """

    def __init__(self, vector_client, encoder, db: AsyncIOMotorDatabase):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_client: VectorDBClient instance
            encoder: EmbeddingEncoder instance
            db: MongoDB database instance
        """
        self.vector_client = vector_client
        self.encoder = encoder
        self.db = db

    async def get_context(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.3
    ) -> str:
        """
        Retrieve relevant product context for a query.
        
        Args:
            query: User query or message
            top_k: Number of top results to retrieve
            min_score: Minimum similarity score threshold
        
        Returns:
            Formatted context string for the LLM
        """
        if not self.vector_client.is_ready():
            logger.warning("Vector database not ready, skipping context retrieval")
            return ""
        
        if not self.encoder.is_ready():
            logger.warning("Encoder not ready, skipping context retrieval")
            return ""
        
        try:
            # Step 1: Encode the query
            query_embedding = self.encoder.encode(query)

            context_parts: List[str] = []

            fetch_k = max(top_k * 4, top_k)

            product_results = await self.vector_client.search(
                query_vector=query_embedding,
                top_k=fetch_k,
                include_metadata=True,
                filter_dict={'type': {'$eq': 'product'}}
            )

            document_results = await self.vector_client.search(
                query_vector=query_embedding,
                top_k=fetch_k,
                include_metadata=True,
                filter_dict={'type': {'$eq': 'document'}}
            )

            candidates: List[dict] = []

            for result in product_results:
                if result.get('score', 0) < min_score:
                    continue
                try:
                    product_id = result['metadata'].get('product_id')
                    if not product_id:
                        continue

                    product = await self.db.products.find_one({'_id': ObjectId(product_id)})
                    if not product:
                        logger.warning(f"Product {product_id} not found in MongoDB")
                        continue
                    rerank_text = " ".join(
                        part
                        for part in [
                            product.get('name', ''),
                            product.get('description', ''),
                            ' '.join(product.get('indications', [])),
                            product.get('category', ''),
                        ]
                        if isinstance(part, str) and part.strip()
                    )
                    candidates.append(
                        {
                            "kind": "product",
                            "score": result.get('score', 0),
                            "metadata": result.get('metadata', {}),
                            "text": rerank_text,
                            "payload": product,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error fetching product details: {e}")

            for result in document_results:
                if result.get('score', 0) < min_score:
                    continue
                try:
                    document_id = result['metadata'].get('document_id')
                    if not document_id:
                        continue

                    document = await self.db.knowledge_documents.find_one({'_id': ObjectId(document_id)})
                    if not document:
                        logger.warning(f"Document {document_id} not found in MongoDB")
                        continue

                    candidates.append(
                        {
                            "kind": "document",
                            "score": result.get('score', 0),
                            "metadata": result.get('metadata', {}),
                            "text": document.get('chunk_text', ''),
                            "payload": document,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error fetching document details: {e}")

            if not candidates:
                logger.info("No relevant products or documents found above score threshold")
                return ""

            reranked = rerank_candidates(query=query, candidates=candidates, top_k=top_k)
            for item in reranked:
                if item.get("kind") == "product":
                    context_parts.append(
                        self._format_product_info(item.get("payload", {}), item.get("rerank_score", item.get("score", 0)))
                    )
                else:
                    context_parts.append(
                        self._format_document_info(item.get("payload", {}), item.get("rerank_score", item.get("score", 0)))
                    )

            context = self._format_context(context_parts)
            logger.info(f"Generated context from {len(context_parts)} sources")

            return context
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return ""

    def _format_product_info(self, product: dict, relevance_score: float) -> str:
        """Format product information for inclusion in context."""
        return f"""
### {product.get('name', 'Unknown Product')} (Relevance: {relevance_score:.2%})
**Category:** {product.get('category', 'N/A')}
**Description:** {product.get('description', 'N/A')}
**Indications:** {', '.join(product.get('indications', [])) or 'N/A'}
**Contraindications:** {', '.join(product.get('contraindications', [])) or 'N/A'}
**Dosage:** {product.get('dosage', 'N/A')}
"""

    def _format_context(self, product_infos: List[str]) -> str:
        """Format multiple products into a context block."""
        return f"""
**RELEVANT PRODUCT INFORMATION:**
{chr(10).join(product_infos)}

**Instructions:** Use the above product information to provide accurate, evidence-based responses in a natural, human tone. 
Do not copy the source text verbatim unless the user explicitly asks for it. Summarize the relevant points clearly and conversationally. 
Always cite the product information when available. Do not provide medical advice for individual patients.
""".strip()

    def _format_document_info(self, document: dict, relevance_score: float) -> str:
        """Format knowledge document chunks for inclusion in context."""
        chunk_text = (document.get('chunk_text') or '').strip()
        if len(chunk_text) > 1200:
            chunk_text = chunk_text[:1200].rstrip() + '...'

        page_number = document.get('page_number', 'N/A')
        source_name = document.get('source_name', 'Unknown source')
        language = document.get('language', 'N/A')

        return f"""
    ### Source note: {source_name} (Page {page_number}, Relevance: {relevance_score:.2%})
**Language:** {language}
    **Key points to synthesize naturally:** {chunk_text}
"""

    async def get_related_conversations(
        self,
        query: str,
        top_k: int = 3
    ) -> List[dict]:
        """
        Find related past conversations using semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of related conversations
        """
        if not self.vector_client.is_ready():
            logger.warning("Vector database not ready")
            return []
        
        try:
            # Encode query
            query_embedding = self.encoder.encode(query)
            
            # Search for conversations
            results = await self.vector_client.search(
                query_vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter_dict={'type': {'$eq': 'conversation'}}
            )
            
            # Fetch full conversation data
            conversations = []
            for result in results:
                try:
                    conv_id = result['metadata'].get('conversation_id')
                    if not conv_id:
                        continue
                    
                    conversation = await self.db.conversations.find_one(
                        {'_id': ObjectId(conv_id)}
                    )
                    
                    if conversation:
                        conversations.append({
                            'id': str(conversation['_id']),
                            'mode': conversation.get('mode'),
                            'created_at': conversation.get('created_at'),
                            'preview': conversation.get('messages', [{}])[0].get('content', '')[:200],
                            'similarity_score': result.get('score', 0)
                        })
                except Exception as e:
                    logger.error(f"Error fetching conversation: {e}")
                    continue
            
            return conversations
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    async def extract_topics_from_query(self, query: str) -> List[str]:
        """
        Extract key topics/concepts from a query.
        This can be extended with more sophisticated NLP.
        
        Args:
            query: User query
        
        Returns:
            List of extracted topics
        """
        try:
            # Simple implementation: split by common delimiters
            # Could be enhanced with spaCy or other NLP tools
            keywords = []
            
            # Common medical/pharma keywords to track
            pharma_keywords = [
                'drug', 'medication', 'dosage', 'side effect', 'contraindication',
                'indication', 'efficacy', 'patient', 'physician', 'trial', 'clinical',
                'adverse', 'interaction', 'treatment', 'therapy', 'disease'
            ]
            
            query_lower = query.lower()
            
            for keyword in pharma_keywords:
                if keyword in query_lower:
                    keywords.append(keyword)
            
            return list(set(keywords))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
