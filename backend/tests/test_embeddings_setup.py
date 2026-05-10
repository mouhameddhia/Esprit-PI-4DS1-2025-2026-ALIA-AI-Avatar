"""Test script to verify vector database and embeddings setup."""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Add backend to path
# __file__ = backend/tests/test_embeddings_setup.py
# parent = backend/tests/
# parent.parent = backend/
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

async def test_embeddings():
    """Test embedding encoder."""
    print("\n=== Testing Embedding Encoder ===")
    
    try:
        from embeddings.encoder import EmbeddingEncoder
        
        encoder = EmbeddingEncoder()
        
        if not encoder.is_ready():
            print("❌ Encoder not ready - check if sentence-transformers is installed")
            return False
        
        print("✓ Encoder initialized successfully")
        print(f"  Model: {encoder.model_name}")
        print(f"  Embedding dimension: {encoder.get_embedding_dimension()}")
        
        # Test single encoding
        test_text = "Laboratoires Vital manufactures high-quality pharmaceutical products"
        embedding = encoder.encode(test_text)
        
        print(f"✓ Single text encoded successfully")
        print(f"  Vector length: {len(embedding)}")
        print(f"  Sample values: {embedding[:3]}...")
        
        # Test batch encoding
        texts = ["Drug A for cancer treatment", "Drug B for heart disease"]
        embeddings = encoder.encode_batch(texts)
        
        print(f"✓ Batch encoding successful")
        print(f"  Encoded {len(embeddings)} texts")
        
        # Test similarity
        sim = encoder.similarity(embeddings[0], embeddings[1])
        print(f"✓ Similarity calculation: {sim:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing encoder: {e}")
        return False


async def test_vector_client():
    """Test vector database client."""
    print("\n=== Testing Vector Database Client ===")
    
    try:
        from vector_db.client import VectorDBClient
        
        client = VectorDBClient()
        
        if not client.is_ready():
            print("⚠️  Vector Database not ready")
            print("   This is expected if PINECONE_API_KEY is not configured")
            print("   Make sure to set PINECONE_API_KEY in .env file")
            return None  # Not a failure, just not configured
        
        print("✓ Vector Database client initialized")
        print(f"  Type: {client.db_type}")
        print(f"  Index: {client.index_name}")
        
        # Get stats
        stats = await client.get_index_stats()
        print(f"✓ Index stats retrieved:")
        print(f"  {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing vector client: {e}")
        return False


async def test_imports():
    """Test that all modules import correctly."""
    print("\n=== Testing Imports ===")
    
    try:
        from vector_db import VectorDBClient
        print("✓ VectorDBClient imported")
        
        from embeddings import EmbeddingEncoder
        print("✓ EmbeddingEncoder imported")
        
        from vector_db.indexing import ProductIndexer
        print("✓ ProductIndexer imported")
        
        from utils.rag_pipeline import RAGPipeline
        print("✓ RAGPipeline imported")
        
        from utils.conversation_embeddings import ConversationEmbedder
        print("✓ ConversationEmbedder imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


async def main():
    """Run all tests."""
    print("🔍 Vector Database & Embeddings Integration Tests")
    print("=" * 50)
    
    results = {}
    
    # Test imports
    results['imports'] = await test_imports()
    
    # Test embeddings
    results['embeddings'] = await test_embeddings()
    
    # Test vector client
    results['vector_client'] = await test_vector_client()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    for test_name, result in results.items():
        status = "✓" if result is True else ("⚠️" if result is None else "❌")
        print(f"  {status} {test_name}: {'PASS' if result is True else ('SKIPPED' if result is None else 'FAIL')}")
    
    all_passed = all(r is not False for r in results.values())
    
    if all_passed:
        print("\n✅ Setup verification complete! Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Configure Pinecone credentials in .env file")
        print("   3. Run the backend: uvicorn backend.main:app --reload")
        print("   4. Access admin endpoints to trigger indexing")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
