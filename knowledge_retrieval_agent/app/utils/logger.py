"""Centralized logging configuration."""

import json
import logging
from datetime import datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one-line JSON logs suitable for local observability."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as a JSON string."""

        payload: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=True)


def get_logger(name: str = "knowledge_retrieval_agent") -> logging.Logger:
    """Create or return a configured logger instance."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
