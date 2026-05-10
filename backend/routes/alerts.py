"""Alerts / Knowledge Gaps — admin only."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..dependencies import get_database, require_roles
from ..models.alert import AlertCreate, AlertResponse, AlertSeverity, AlertStatus
from ..models.user import UserInDB
from ..utils.mongo import object_id

router = APIRouter(prefix="/alerts", tags=["alerts"])

_admin = Depends(require_roles("admin"))

# Realistic seed data used until the NLP pipeline produces real alerts
_SEED_ALERTS = [
    {
        "user_message": "What are the long-term side effects of Arkogelules for patients with renal impairment?",
        "detected_intent": "product_safety_query",
        "missing_entity": "renal_contraindications",
        "missing_info": "No renal-impairment dosing guidance found in product knowledge base.",
        "severity": "HIGH", "status": "open", "source_conversation_id": None,
        "timestamp": datetime(2026, 4, 28, 14, 32, 0), "resolved_at": None,
    },
    {
        "user_message": "Can I take Phyto-Extract with anticoagulants?",
        "detected_intent": "drug_interaction_query",
        "missing_entity": "drug_interaction_anticoagulants",
        "missing_info": "Drug interaction data with anticoagulants not present in the knowledge base.",
        "severity": "HIGH", "status": "open", "source_conversation_id": None,
        "timestamp": datetime(2026, 4, 29, 9, 15, 0), "resolved_at": None,
    },
    {
        "user_message": "What is the recommended dosage for elderly patients?",
        "detected_intent": "dosage_query",
        "missing_entity": "elderly_dosage_adjustment",
        "missing_info": "Geriatric dosing guidelines missing from product documentation.",
        "severity": "MEDIUM", "status": "open", "source_conversation_id": None,
        "timestamp": datetime(2026, 4, 30, 11, 5, 0), "resolved_at": None,
    },
    {
        "user_message": "Is there a pediatric formulation available?",
        "detected_intent": "product_availability_query",
        "missing_entity": "pediatric_formulation",
        "missing_info": "No pediatric formulation data available in catalogue.",
        "severity": "MEDIUM", "status": "resolved", "source_conversation_id": None,
        "timestamp": datetime(2026, 4, 27, 16, 45, 0), "resolved_at": datetime(2026, 5, 1, 10, 0, 0),
    },
    {
        "user_message": "How should the product be stored after opening?",
        "detected_intent": "storage_query",
        "missing_entity": "post_open_storage_instructions",
        "missing_info": "Post-opening storage instructions not documented.",
        "severity": "LOW", "status": "open", "source_conversation_id": None,
        "timestamp": datetime(2026, 5, 1, 8, 20, 0), "resolved_at": None,
    },
]


async def _seed_if_empty(db: AsyncIOMotorDatabase) -> None:
    if await db.alerts.count_documents({}) == 0:
        await db.alerts.insert_many(_SEED_ALERTS)


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    severity: Optional[AlertSeverity] = Query(None),
    alert_status: Optional[AlertStatus] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    await _seed_if_empty(db)

    query: dict = {}
    if severity:
        query["severity"] = severity.value
    if alert_status:
        query["status"] = alert_status.value

    cursor = db.alerts.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    return [AlertResponse(**a) for a in await cursor.to_list(length=limit)]


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    payload: AlertCreate,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    doc = {**payload.model_dump(), "timestamp": datetime.utcnow(), "resolved_at": None}
    result = await db.alerts.insert_one(doc)
    return AlertResponse(**await db.alerts.find_one({"_id": result.inserted_id}))


@router.patch("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: str,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    oid = object_id(alert_id, "alert")
    result = await db.alerts.update_one(
        {"_id": oid},
        {"$set": {"status": "resolved", "resolved_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    return AlertResponse(**await db.alerts.find_one({"_id": oid}))


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(
    alert_id: str,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    oid = object_id(alert_id, "alert")
    result = await db.alerts.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
