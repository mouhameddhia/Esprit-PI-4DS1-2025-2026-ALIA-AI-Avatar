from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging

# config.py calls load_dotenv() on import, so it must come before any module
# that reads os.getenv() at module scope.
from . import config
from .routes import auth, chat, sessions, debug, admin, products, users_admin, alerts, audio, affect
from .utils.background_tasks import (
    auto_finalize_idle_sessions,
    auto_generate_shadow_monitoring_snapshot,
)
from .vector_db import VectorDBClient
from .embeddings import EmbeddingEncoder
from .vector_db.indexing import ProductIndexer
from .vector_db.knowledge_indexing import KnowledgeDocumentIndexer
from .utils.conversation_embeddings import ConversationEmbedder
from .utils.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ALIA Backend", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
client = AsyncIOMotorClient(config.MONGODB_URL)
db = client.alia

# Vector Database and Embedding initialization
vector_client = VectorDBClient()
embedding_encoder = EmbeddingEncoder()
product_indexer = ProductIndexer(vector_client, embedding_encoder)
knowledge_document_indexer = KnowledgeDocumentIndexer(vector_client, embedding_encoder)
conversation_embedder = ConversationEmbedder(vector_client, embedding_encoder)
rag_pipeline = RAGPipeline(vector_client, embedding_encoder, db)

# Task scheduler
scheduler = AsyncIOScheduler()


async def _run_shadow_monitoring_job() -> None:
    try:
        result = await auto_generate_shadow_monitoring_snapshot(db)
        logger.info(
            "Shadow monitoring snapshot complete: rows=%s divergence=%.2f%% gate=%s",
            result.get("rows", 0),
            float(result.get("divergence_rate", 0.0)) * 100,
            result.get("quality_gate", "unknown"),
        )
    except Exception as exc:
        logger.error(f"Shadow monitoring snapshot failed: {exc}")

# Include routers
app.include_router(auth.router,         prefix="/auth",  tags=["auth"])
app.include_router(chat.router,         prefix="/chat",  tags=["chat"])
app.include_router(sessions.router,     prefix="/chat",  tags=["sessions"])
app.include_router(debug.router,        prefix="/chat",  tags=["debug"])
app.include_router(admin.router,                         tags=["admin"])
app.include_router(products.router,                      tags=["products"])
app.include_router(users_admin.router,                   tags=["users-admin"])
app.include_router(alerts.router,                        tags=["alerts"])
app.include_router(audio.router,                         tags=["audio"])
app.include_router(affect.router,                        tags=["affect"])

@app.get("/")
async def root():
    return {"message": "ALIA Backend API"}

@app.get("/health")
async def health():
    try:
        await client.admin.command("ping")
        db_status = "ok"
    except Exception:
        db_status = "unavailable"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "db": db_status,
        "vector_db": "ok" if vector_client.is_ready() else "unavailable",
        "embedding": "ok" if embedding_encoder.is_ready() else "unavailable",
    }

@app.on_event("startup")
async def startup_event():
    # Test DB connection
    try:
        await client.admin.command('ping')
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")

    # Ensure MongoDB indexes
    try:
        await db.users.create_index("email", unique=True, background=True)
        await db.conversations.create_index("user_email", background=True)
        await db.conversations.create_index("status", background=True)
        await db.conversations.create_index(
            [("user_email", 1), ("updated_at", -1)], background=True
        )
        await db.conversations.create_index("nlp_events.intent", background=True)
        await db.products.create_index("name", background=True)
        await db.products.create_index("category", background=True)
        await db.alerts.create_index("status", background=True)
        await db.alerts.create_index("severity", background=True)
        await db.alerts.create_index("timestamp", background=True)
        logger.info("MongoDB indexes ensured")
    except Exception as e:
        logger.warning(f"MongoDB index creation warning: {e}")
    
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
    
    # Pre-load affect classifier so first user request has no cold-start latency
    try:
        from alia_nlp.src.layers.L7_affect.classifier import predict as _affect_predict
        _affect_predict("warmup", "medrep_training")
        logger.info("Affect classifier pre-loaded")
    except Exception as _exc:
        logger.warning("Affect classifier warm-up skipped: %s", _exc)

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
        scheduler.add_job(
            _run_shadow_monitoring_job,
            "cron",
            hour=config.SHADOW_JOB_HOUR_UTC,
            minute=config.SHADOW_JOB_MINUTE_UTC,
            id="auto_shadow_monitoring_snapshot",
            name="Auto-generate daily shadow monitoring snapshot",
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
