from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import os
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
from .routes import auth, chat, admin
from .utils.background_tasks import auto_finalize_idle_sessions
from .vector_db import VectorDBClient
from .embeddings import EmbeddingEncoder
from .vector_db.indexing import ProductIndexer
from .vector_db.knowledge_indexing import KnowledgeDocumentIndexer
from .utils.conversation_embeddings import ConversationEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ALIA Backend", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # Frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia")
client = AsyncIOMotorClient(mongodb_url)
db = client.alia

# Vector Database and Embedding initialization
vector_client = VectorDBClient()
embedding_encoder = EmbeddingEncoder()
product_indexer = ProductIndexer(vector_client, embedding_encoder)
knowledge_document_indexer = KnowledgeDocumentIndexer(vector_client, embedding_encoder)
conversation_embedder = ConversationEmbedder(vector_client, embedding_encoder)

# Task scheduler
scheduler = AsyncIOScheduler()

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(admin.router, tags=["admin"])

@app.get("/")
async def root():
    return {"message": "ALIA Backend API"}

@app.on_event("startup")
async def startup_event():
    # Test DB connection
    try:
        await client.admin.command('ping')
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
    
    # Initialize Vector Database and Embeddings
    logger.info("Initializing Vector Database and Embeddings...")
    
    if vector_client.is_ready():
        logger.info("Vector Database client ready")
        
        # Index products on startup
        try:
            logger.info("Starting product indexing...")
            index_result = await product_indexer.index_products(db)
            
            if index_result.get('success'):
                logger.info(
                    f"Product indexing complete: {index_result.get('indexed_count')} "
                    f"products indexed"
                )
            else:
                logger.warning(f"Product indexing failed: {index_result.get('error')}")
        except Exception as e:
            logger.error(f"Error during product indexing: {e}")

        try:
            logger.info("Starting knowledge document indexing...")
            knowledge_result = await knowledge_document_indexer.index_documents(db)
            if knowledge_result.get('success'):
                logger.info(
                    f"Knowledge document indexing complete: {knowledge_result.get('indexed_count')} chunks indexed"
                )
            else:
                logger.warning(f"Knowledge document indexing failed: {knowledge_result.get('error')}")
        except Exception as e:
            logger.error(f"Error during knowledge document indexing: {e}")
    else:
        logger.warning(
            "Vector Database not ready. Make sure PINECONE_API_KEY is configured "
            "in .env file. Context retrieval will be disabled."
        )
    
    if embedding_encoder.is_ready():
        logger.info(
            f"Embedding encoder ready ({embedding_encoder.get_embedding_dimension()}D)"
        )
    else:
        logger.warning("Embedding encoder not ready")
    
    # Start scheduler for auto-finalize task
    if not scheduler.running:
        scheduler.add_job(
            auto_finalize_idle_sessions,
            "interval",
            minutes=5,  # Run every 5 minutes
            args=(db,),
            id="auto_finalize_idle_sessions",
            name="Auto-finalize idle conversations",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Background task scheduler started")

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
    
    vector_client.close()
    client.close()
    logger.info("Application shutdown complete")
