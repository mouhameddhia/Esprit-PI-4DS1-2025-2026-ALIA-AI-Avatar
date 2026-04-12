"""Knowledge document indexing module for PDF and other source text."""

import logging
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class KnowledgeDocumentIndexer:
    """Index source documents into the vector database."""

    def __init__(self, vector_client, encoder):
        self.vector_client = vector_client
        self.encoder = encoder

    async def index_documents(
        self,
        db: AsyncIOMotorDatabase,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.vector_client.is_ready():
            logger.error("Vector database not ready")
            return {"success": False, "error": "Vector DB not initialized"}

        if not self.encoder.is_ready():
            logger.error("Embedding encoder not ready")
            return {"success": False, "error": "Encoder not initialized"}

        try:
            query: Dict[str, Any] = {}
            if source_name:
                query["source_name"] = source_name

            documents = await db.knowledge_documents.find(query).to_list(None)
            logger.info(f"Found {len(documents)} knowledge documents to index")

            if not documents:
                return {"success": True, "indexed_count": 0, "documents": []}

            vectors_to_upsert = []
            for document in documents:
                try:
                    text_parts = [
                        document.get("source_name", ""),
                        document.get("title", ""),
                        document.get("chunk_text", ""),
                        document.get("keywords_text", ""),
                    ]
                    searchable_text = " ".join([part for part in text_parts if part]).strip()

                    if not searchable_text:
                        logger.warning(f"Document {document['_id']} has no searchable text")
                        continue

                    embedding = self.encoder.encode(searchable_text)
                    vector_id = f"document_{str(document['_id'])}"

                    vectors_to_upsert.append(
                        {
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {
                                "type": "document",
                                "document_id": str(document["_id"]),
                                "source_name": document.get("source_name", ""),
                                "source_path": document.get("source_path", ""),
                                "title": document.get("title", ""),
                                "language": document.get("language", ""),
                                "page_number": document.get("page_number"),
                                "chunk_index": document.get("chunk_index", 0),
                            },
                        }
                    )
                except Exception as exc:
                    logger.error(f"Error processing document {document.get('_id')}: {exc}")
                    continue

            batch_size = 100
            total_upserted = 0
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                success = await self.vector_client.upsert(batch)
                if success:
                    total_upserted += len(batch)
                    logger.info(f"Upserted knowledge batch {i // batch_size + 1} ({len(batch)} vectors)")
                else:
                    logger.error(f"Failed to upsert knowledge batch {i // batch_size + 1}")

            return {
                "success": True,
                "indexed_count": total_upserted,
                "total_documents": len(documents),
                "documents": [vector["metadata"] for vector in vectors_to_upsert],
            }
        except Exception as exc:
            logger.error(f"Error indexing knowledge documents: {exc}")
            return {"success": False, "error": str(exc)}

    async def reindex_document(self, db: AsyncIOMotorDatabase, document_id: str) -> bool:
        try:
            vector_id = f"document_{document_id}"
            await self.vector_client.delete([vector_id])

            document = await db.knowledge_documents.find_one({"_id": ObjectId(document_id)})
            if not document:
                logger.warning(f"Document {document_id} not found")
                return False

            searchable_text = " ".join(
                [
                    document.get("source_name", ""),
                    document.get("title", ""),
                    document.get("chunk_text", ""),
                    document.get("keywords_text", ""),
                ]
            ).strip()

            embedding = self.encoder.encode(searchable_text)
            success = await self.vector_client.upsert(
                [
                    {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "type": "document",
                            "document_id": document_id,
                            "source_name": document.get("source_name", ""),
                            "source_path": document.get("source_path", ""),
                            "title": document.get("title", ""),
                            "language": document.get("language", ""),
                            "page_number": document.get("page_number"),
                            "chunk_index": document.get("chunk_index", 0),
                        },
                    }
                ]
            )
            if success:
                logger.info(f"Successfully reindexed knowledge document {document_id}")
            return success
        except Exception as exc:
            logger.error(f"Error reindexing knowledge document: {exc}")
            return False
