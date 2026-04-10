import os
from datetime import datetime
from typing import List, Literal, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

from ..dependencies import get_database, get_current_user
from ..models.conversation import ConversationResponse, SessionListItem
from ..models.user import UserInDB

router = APIRouter()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPTS = {
    "physician_portal": (
        "You are ALIA, an AI pharmaceutical representative for Laboratoires Vital (ALIA). "
        "You help healthcare professionals with product information, clinical data summaries, "
        "dosing guidelines, and educational content. Be clear, accurate, and compliant. "
        "Do not provide medical advice for individual patients. Do not diagnose. "
        "If you are uncertain, say so and suggest consulting official labeling or a medical specialist."
    ),
    "medrep_training": (
        "You are simulating a physician in a training scenario for medical representatives. "
        "Respond realistically to the rep's messages. You may ask challenging questions, "
        "object to claims, or request evidence. Keep responses concise and professional."
    ),
}


def _groq_client():
    try:
        from groq import Groq
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq SDK not installed. Run: pip install groq",
        ) from exc
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY not configured",
        )
    return Groq(api_key=key)


def _chat_completion(messages: list[dict]) -> str:
    client = _groq_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    choice = completion.choices[0].message
    if not choice or not choice.content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Empty response from language model",
        )
    return choice.content.strip()


def _summary_completion(transcript: str) -> str:
    client = _groq_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize pharmaceutical training / HCP chat sessions for internal records. "
                    "Output a short headline (one line) then 3–6 bullet points. "
                    "Focus on topics discussed, products mentioned, objections, and any follow-ups. "
                    "Be factual; do not invent details not present in the transcript."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this conversation:\n\n{transcript}",
            },
        ],
        temperature=0.4,
        max_tokens=512,
    )
    choice = completion.choices[0].message
    if not choice or not choice.content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Empty summary from language model",
        )
    return choice.content.strip()


class SendMessageRequest(BaseModel):
    session_id: Optional[str] = None
    content: str = Field(..., min_length=1, max_length=16000)
    mode: Literal["physician_portal", "medrep_training"] = "physician_portal"


class SendMessageResponse(BaseModel):
    session_id: str
    reply: str


class FinalizeResponse(BaseModel):
    session_id: str
    summary: str


def _ensure_mode(mode: str) -> str:
    if mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mode")
    return mode


async def _get_owned_session(
    db: AsyncIOMotorDatabase, session_id: str, email: str
):
    try:
        oid = ObjectId(session_id)
    except InvalidId as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id") from exc
    doc = await db.conversations.find_one({"_id": oid})
    if not doc or doc.get("user_email") != email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return doc, oid


@router.post("/message", response_model=SendMessageResponse)
async def send_message(
    body: SendMessageRequest,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    now = datetime.utcnow()

    if body.session_id:
        doc, oid = await _get_owned_session(db, body.session_id, current_user.email)
        if doc.get("status") == "closed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation is closed; start a new chat",
            )
        mode = _ensure_mode(doc["mode"])
        messages = list(doc.get("messages") or [])
    else:
        doc = None
        oid = None
        mode = _ensure_mode(body.mode)
        messages = []

    user_entry = {"role": "user", "content": body.content.strip(), "at": now}
    groq_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]
    for m in messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})
    groq_messages.append({"role": "user", "content": user_entry["content"]})

    reply_text = _chat_completion(groq_messages)
    asst_time = datetime.utcnow()
    asst_entry = {"role": "assistant", "content": reply_text, "at": asst_time}

    messages.append(user_entry)
    messages.append(asst_entry)

    if doc is None:
        insert_doc = {
            "user_email": current_user.email,
            "mode": mode,
            "messages": messages,
            "summary": None,
            "summary_created_at": None,
            "status": "open",
            "created_at": now,
            "updated_at": asst_time,
        }
        result = await db.conversations.insert_one(insert_doc)
        session_id = str(result.inserted_id)
    else:
        await db.conversations.update_one(
            {"_id": oid},
            {"$set": {"messages": messages, "updated_at": asst_time}},
        )
        session_id = str(oid)

    return SendMessageResponse(session_id=session_id, reply=reply_text)


@router.post("/sessions/{session_id}/finalize", response_model=FinalizeResponse)
async def finalize_session(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    doc, oid = await _get_owned_session(db, session_id, current_user.email)
    if doc.get("summary"):
        return FinalizeResponse(session_id=session_id, summary=doc["summary"])

    msgs = doc.get("messages") or []
    if not msgs:
        summary = "No messages in this conversation."
        now = datetime.utcnow()
        await db.conversations.update_one(
            {"_id": oid},
            {"$set": {"summary": summary, "summary_created_at": now, "status": "closed", "updated_at": now}},
        )
        return FinalizeResponse(session_id=session_id, summary=summary)

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    summary = _summary_completion(transcript)
    now = datetime.utcnow()
    await db.conversations.update_one(
        {"_id": oid},
        {"$set": {"summary": summary, "summary_created_at": now, "status": "closed", "updated_at": now}},
    )
    return FinalizeResponse(session_id=session_id, summary=summary)


@router.get("/sessions", response_model=List[SessionListItem])
async def list_sessions(
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    limit: int = 50,
):
    limit = min(max(limit, 1), 100)
    cursor = (
        db.conversations.find({"user_email": current_user.email})
        .sort("updated_at", -1)
        .limit(limit)
    )
    items: list[SessionListItem] = []
    async for doc in cursor:
        summary = doc.get("summary")
        msgs = doc.get("messages") or []
        if summary:
            preview = summary[:220].replace("\n", " ")
        elif msgs:
            preview = (msgs[-1].get("content") or "")[:220]
        else:
            preview = ""
        items.append(
            SessionListItem(
                id=str(doc["_id"]),
                mode=doc.get("mode", ""),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                summary=summary,
                summary_created_at=doc.get("summary_created_at"),
                status=doc.get("status", "open"),
                preview=preview,
            )
        )
    return items


@router.get("/sessions/{session_id}", response_model=ConversationResponse)
async def get_session(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    doc, _ = await _get_owned_session(db, session_id, current_user.email)
    return ConversationResponse(**doc)
