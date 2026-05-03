"""Shared MongoDB helpers used across route handlers."""

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException, status


def object_id(id_str: str, entity: str = "resource") -> ObjectId:
    """
    Convert a hex string to a BSON ObjectId, raising a 400 HTTPException on
    failure so every route gets a consistent, descriptive error message.

    Args:
        id_str: The 24-char hex string from the URL path parameter.
        entity: Human-readable resource name used in the error message.
    """
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {entity} ID",
        )
