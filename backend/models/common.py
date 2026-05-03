"""Shared Pydantic/MongoDB helpers used across domain models."""

from typing import Any

from bson import ObjectId


def coerce_mongo_id(data: Any) -> Any:
    """
    Remap MongoDB's ``_id`` field to ``id`` so that FastAPI serialises the
    response with the key ``"id"`` rather than ``"_id"``.

    FastAPI calls ``jsonable_encoder(..., by_alias=True)`` internally, which
    means a ``Field(..., alias="_id")`` would produce ``"_id"`` in the JSON
    output — not what frontend code expects.  Using a ``model_validator``
    that calls this helper avoids aliases entirely.
    """
    if isinstance(data, dict) and "_id" in data:
        data = dict(data)
        raw = data.pop("_id")
        data["id"] = str(raw) if isinstance(raw, ObjectId) else str(raw)
    return data
