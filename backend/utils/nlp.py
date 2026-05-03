"""Compatibility wrapper — re-exports from alia_nlp."""

from alia_nlp.data.taxonomy.loader import SUPPORTED_INTENTS
from alia_nlp.src.pipeline.online import analyze_message_nlp

__all__ = ["SUPPORTED_INTENTS", "analyze_message_nlp"]
