"""Conversation embedding and indexing module."""

import logging
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

logger = logging.getLogger(__name__)


class ConversationEmbedder:
    """
    Handles embedding and indexing of conversations for semantic search.
    Allows finding similar past conversations and extracting insights.
    """

    def __init__(self, vector_client, encoder):
        """
        Initialize conversation embedder.
        
        Args:
            vector_client: VectorDBClient instance
            encoder: EmbeddingEncoder instance
        """
        self.vector_client = vector_client
        self.encoder = encoder

    async def embed_conversation(
        self,
        db: AsyncIOMotorDatabase,
        conversation_id: str
    ) -> bool:
        """
        Embed and index a conversation.
        
        Args:
            db: MongoDB database instance
            conversation_id: ID of conversation to embed
        
        Returns:
            True if successful
        """
        if not self.vector_client.is_ready():
            logger.warning("Vector database not ready")
            return False
        
        if not self.encoder.is_ready():
            logger.warning("Encoder not ready")
            return False
        
        try:
            # Fetch conversation
            conversation = await db.conversations.find_one(
                {'_id': ObjectId(conversation_id)}
            )
            
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return False
            
            # Extract and combine all messages
            messages = conversation.get('messages', [])
            message_texts = [msg.get('content', '') for msg in messages if msg.get('content')]
            
            if not message_texts:
                logger.warning(f"No messages found in conversation {conversation_id}")
                return False
            
            # Combine all messages into one text
            full_text = ' '.join(message_texts)
            
            # Generate embedding
            embedding = self.encoder.encode(full_text)
            
            # Prepare vector for upsert
            vector_id = f"conv_{conversation_id}"
            
            # Create summary for metadata
            summary = conversation.get('summary', '')[:200] if conversation.get('summary') else ''
            
            success = await self.vector_client.upsert([{
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'type': 'conversation',
                    'conversation_id': conversation_id,
                    'user_email': conversation.get('user_email', ''),
                    'mode': conversation.get('mode', ''),
                    'created_at': str(conversation.get('created_at', '')),
                    'message_count': len(messages),
                    'summary': summary
                }
            }])
            
            if success:
                logger.info(f"Successfully embedded conversation {conversation_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error embedding conversation: {e}")
            return False

    async def embed_batch_conversations(
        self,
        db: AsyncIOMotorDatabase,
        conversation_ids: List[str]
    ) -> dict:
        """
        Embed multiple conversations.
        
        Args:
            db: MongoDB database instance
            conversation_ids: List of conversation IDs
        
        Returns:
            Dictionary with success count and failures
        """
        successes = 0
        failures = []
        
        for conv_id in conversation_ids:
            try:
                success = await self.embed_conversation(db, conv_id)
                if success:
                    successes += 1
                else:
                    failures.append(conv_id)
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                failures.append(conv_id)
        
        return {
            'total': len(conversation_ids),
            'successes': successes,
            'failures': failures
        }

    async def reindex_all_conversations(
        self,
        db: AsyncIOMotorDatabase,
        batch_size: int = 50
    ) -> dict:
        """
        Reindex all conversations in the database.
        
        Args:
            db: MongoDB database instance
            batch_size: Number of conversations to process at once
        
        Returns:
            Dictionary with indexing statistics
        """
        try:
            # Get all conversation IDs
            conversations = await db.conversations.find({}, {'_id': 1}).to_list(None)
            conversation_ids = [str(c['_id']) for c in conversations]
            
            logger.info(f"Reindexing {len(conversation_ids)} conversations")
            
            total_successes = 0
            total_failures = []
            
            # Process in batches
            for i in range(0, len(conversation_ids), batch_size):
                batch = conversation_ids[i:i + batch_size]
                result = await self.embed_batch_conversations(db, batch)
                
                total_successes += result['successes']
                total_failures.extend(result['failures'])
                
                logger.info(
                    f"Batch {i // batch_size + 1}: "
                    f"{result['successes']}/{result['total']} successful"
                )
            
            return {
                'total_conversations': len(conversation_ids),
                'successful_embeddings': total_successes,
                'failed_embeddings': total_failures,
                'success_rate': total_successes / len(conversation_ids) if conversation_ids else 0
            }
        except Exception as e:
            logger.error(f"Error reindexing conversations: {e}")
            return {'error': str(e)}

    async def delete_conversation_index(self, conversation_id: str) -> bool:
        """
        Remove conversation from vector database.
        
        Args:
            conversation_id: Conversation ID to remove
        
        Returns:
            True if successful
        """
        try:
            vector_id = f"conv_{conversation_id}"
            success = await self.vector_client.delete([vector_id])
            
            if success:
                logger.info(f"Deleted conversation index {conversation_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error deleting conversation index: {e}")
            return False

    async def get_conversation_insights(
        self,
        db: AsyncIOMotorDatabase,
        conversation_id: str
    ) -> dict:
        """
        Extract insights from a conversation using embeddings.
        
        Args:
            db: MongoDB database instance
            conversation_id: Conversation ID
        
        Returns:
            Dictionary with conversation insights
        """
        try:
            conversation = await db.conversations.find_one(
                {'_id': ObjectId(conversation_id)}
            )
            
            if not conversation:
                return {'error': 'Conversation not found'}
            
            messages = conversation.get('messages', [])
            
            # Extract first and last message for comparison
            first_msg = messages[0].get('content', '') if messages else ''
            last_msg = messages[-1].get('content', '') if messages else ''
            
            # Calculate conversation similarity (how coherent it is)
            similarity = 0.0
            if first_msg and last_msg and self.encoder.is_ready():
                emb1 = self.encoder.encode(first_msg)
                emb2 = self.encoder.encode(last_msg)
                similarity = self.encoder.similarity(emb1, emb2)
            
            return {
                'conversation_id': conversation_id,
                'message_count': len(messages),
                'user_email': conversation.get('user_email', ''),
                'mode': conversation.get('mode', ''),
                'created_at': str(conversation.get('created_at', '')),
                'summary': conversation.get('summary', ''),
                'topics': conversation.get('topics', []),
                'objections': conversation.get('objections', []),
                'action_items': conversation.get('action_items', []),
                'conversation_coherence': similarity  # 0-1 score
            }
        except Exception as e:
            logger.error(f"Error getting conversation insights: {e}")
            return {'error': str(e)}
