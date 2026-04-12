"""Product knowledge indexing module."""

import logging
from typing import List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class ProductIndexer:
    """
    Manages indexing of product data into the vector database.
    Handles chunking, embedding, and upserting product information.
    """

    def __init__(self, vector_client, encoder):
        """
        Initialize the indexer.
        
        Args:
            vector_client: VectorDBClient instance
            encoder: EmbeddingEncoder instance
        """
        self.vector_client = vector_client
        self.encoder = encoder

    async def index_products(self, db: AsyncIOMotorDatabase) -> Dict[str, Any]:
        """
        Index all products from MongoDB into vector database.
        
        Args:
            db: MongoDB database instance
        
        Returns:
            Dictionary with indexing statistics
        """
        if not self.vector_client.is_ready():
            logger.error("Vector database not ready")
            return {'success': False, 'error': 'Vector DB not initialized'}
        
        if not self.encoder.is_ready():
            logger.error("Embedding encoder not ready")
            return {'success': False, 'error': 'Encoder not initialized'}
        
        try:
            # Fetch all products
            products = await db.products.find().to_list(None)
            logger.info(f"Found {len(products)} products to index")
            
            if not products:
                logger.warning("No products found in database")
                return {'success': True, 'indexed_count': 0, 'products': []}
            
            # Prepare vectors for upsert
            vectors_to_upsert = []
            
            for product in products:
                try:
                    # Create searchable text from product fields
                    text_parts = [
                        product.get('name', ''),
                        product.get('description', ''),
                        ' '.join(product.get('indications', [])),
                        product.get('category', ''),
                    ]
                    
                    # Filter out empty strings and join
                    searchable_text = ' '.join([t for t in text_parts if t])
                    
                    if not searchable_text.strip():
                        logger.warning(f"Product {product['_id']} has no searchable text")
                        continue
                    
                    # Generate embedding
                    embedding = self.encoder.encode(searchable_text)
                    
                    # Prepare vector for upsert
                    vector_id = f"product_{str(product['_id'])}"
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': {
                            'type': 'product',
                            'product_id': str(product['_id']),
                            'name': product.get('name', ''),
                            'category': product.get('category', ''),
                            'created_at': str(product.get('created_at', '')),
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing product {product.get('_id')}: {e}")
                    continue
            
            # Upsert vectors in batches (Pinecone has limits)
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                success = await self.vector_client.upsert(batch)
                
                if success:
                    total_upserted += len(batch)
                    logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
                else:
                    logger.error(f"Failed to upsert batch {i // batch_size + 1}")
            
            logger.info(f"Indexing complete. Total vectors upserted: {total_upserted}")
            
            return {
                'success': True,
                'indexed_count': total_upserted,
                'total_products': len(products),
                'products': [v['metadata'] for v in vectors_to_upsert]
            }
        except Exception as e:
            logger.error(f"Error indexing products: {e}")
            return {'success': False, 'error': str(e)}

    async def reindex_product(
        self,
        db: AsyncIOMotorDatabase,
        product_id: str
    ) -> bool:
        """
        Reindex a single product (after update).
        
        Args:
            db: MongoDB database instance
            product_id: Product ID to reindex
        
        Returns:
            True if successful
        """
        try:
            # Delete old vector
            vector_id = f"product_{product_id}"
            await self.vector_client.delete([vector_id])
            
            # Fetch and reindex
            product = await db.products.find_one({'_id': ObjectId(product_id)})
            
            if not product:
                logger.warning(f"Product {product_id} not found")
                return False
            
            # Create searchable text
            text_parts = [
                product.get('name', ''),
                product.get('description', ''),
                ' '.join(product.get('indications', [])),
                product.get('category', ''),
            ]
            searchable_text = ' '.join([t for t in text_parts if t])
            
            # Generate embedding
            embedding = self.encoder.encode(searchable_text)
            
            # Upsert new vector
            success = await self.vector_client.upsert([{
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'type': 'product',
                    'product_id': product_id,
                    'name': product.get('name', ''),
                    'category': product.get('category', ''),
                }
            }])
            
            if success:
                logger.info(f"Successfully reindexed product {product_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error reindexing product: {e}")
            return False

    async def delete_product_index(self, product_id: str) -> bool:
        """
        Delete a product from the vector database.
        
        Args:
            product_id: Product ID to remove
        
        Returns:
            True if successful
        """
        try:
            vector_id = f"product_{product_id}"
            success = await self.vector_client.delete([vector_id])
            
            if success:
                logger.info(f"Deleted index for product {product_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error deleting product index: {e}")
            return False
