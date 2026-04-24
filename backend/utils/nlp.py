"""Compatibility wrapper for NLP pipeline imports.

Runtime code should keep importing from backend.utils.nlp while the source of truth
now lives in NLP/pipeline/nlp.py.
"""

from NLP.pipeline.nlp import SUPPORTED_INTENTS, analyze_message_nlp

__all__ = ["SUPPORTED_INTENTS", "analyze_message_nlp"]
