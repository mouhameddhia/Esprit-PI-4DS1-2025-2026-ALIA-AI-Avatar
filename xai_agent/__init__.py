"""
XAI Agent - Intelligent Explainable AI for Medical Multi-Agent Systems

Real, executable Python module with JSON I/O and sophisticated analysis.

Quick Start:
    from xai_agent.pipeline import analyze_system
    
    result = analyze_system(json_input_string)
    output = json.loads(result)

Components:
- retrieval.py    - RAG quality analysis
- generation.py   - Answer consistency auditing
- judge.py        - Judge fairness validation
- safety.py       - Medical safety risk detection
- trainer.py      - Training feedback analysis
- pipeline.py     - Main orchestrator with JSON I/O
"""

from xai_agent.pipeline import XAIPipeline, analyze_system
from xai_agent.retrieval import RetrievalAnalyzer
from xai_agent.generation import GenerationAuditor
from xai_agent.judge import JudgeAnalyzer
from xai_agent.safety import SafetyAuditor
from xai_agent.trainer import TrainingAnalyzer
from xai_agent.shap_generation_explainer import SHAPGenerationExplainer, explain_generation_with_shap
from xai_agent.lime_retrieval_explainer import LIMERetrievalExplainer, explain_retrieval_with_lime

__version__ = "2.0.0"
__all__ = [
    "XAIPipeline",
    "analyze_system",
    "RetrievalAnalyzer",
    "GenerationAuditor",
    "JudgeAnalyzer",
    "SafetyAuditor",
    "TrainingAnalyzer",
    "SHAPGenerationExplainer",
    "explain_generation_with_shap",
    "LIMERetrievalExplainer",
    "explain_retrieval_with_lime",
]
