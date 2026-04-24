"""Compatibility wrapper for NLP evaluation imports.

Runtime code should keep importing from backend.utils.nlp_evaluator while the
source of truth now lives in NLP/evaluation/evaluator.py.
"""

from NLP.evaluation.evaluator import TAXONOMY, evaluate_conversation

__all__ = ["TAXONOMY", "evaluate_conversation"]
