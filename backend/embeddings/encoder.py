"""Embedding encoder using sentence-transformers."""

import os
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Generates embeddings for text using sentence-transformers.
    Provides batch processing and caching capabilities.
    """

    def __init__(self):
        """Initialize the embedding model."""
        self.model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "384"))
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

    def is_ready(self) -> bool:
        """Check if embedding model is ready."""
        return self.model is not None

    def encode(self, text: str) -> List[float]:
        """
        Encode a single text to embedding.
        
        Args:
            text: Input text to encode
        
        Returns:
            Embedding vector as list of floats
        """
        if not self.is_ready():
            logger.error("Embedding model not initialized")
            return [0.0] * self.embedding_dim
        
        try:
            if not text or not isinstance(text, str):
                logger.warning("Invalid input text")
                return [0.0] * self.embedding_dim
            
            # Encode and normalize
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return [0.0] * self.embedding_dim

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
        
        Returns:
            List of embedding vectors
        """
        if not self.is_ready():
            logger.error("Embedding model not initialized")
            return [[0.0] * self.embedding_dim for _ in texts]
        
        try:
            if not texts or not all(isinstance(t, str) for t in texts):
                logger.warning("Invalid input texts")
                return [[0.0] * self.embedding_dim for _ in texts]
            
            # Encode in batch with normalization
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
