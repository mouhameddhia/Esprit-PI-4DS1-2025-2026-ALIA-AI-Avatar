from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
from .routes import auth, chat
from .utils.background_tasks import auto_finalize_idle_sessions

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

# Task scheduler
scheduler = AsyncIOScheduler()

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "ALIA Backend API"}

@app.on_event("startup")
async def startup_event():
    # Test DB connection
    try:
        await client.admin.command('ping')
        print("Connected to MongoDB")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
    
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
        print("Background task scheduler started")

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
    client.close()
