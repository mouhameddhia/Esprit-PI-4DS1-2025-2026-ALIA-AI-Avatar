"""
XAI Pipeline - Main orchestrator with JSON I/O
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from xai_agent.retrieval import RetrievalAnalyzer
from xai_agent.generation import GenerationAuditor
from xai_agent.judge import JudgeAnalyzer
from xai_agent.safety import SafetyAuditor
from xai_agent.trainer import TrainingAnalyzer
from xai_agent.shap_generation_explainer import explain_generation_with_shap
from xai_agent.lime_retrieval_explainer import explain_retrieval_with_lime


class XAIPipeline:
    """
    Main XAI pipeline that accepts structured JSON inputs and returns comprehensive analysis.
    
    Input JSON Schema:
    {
      "query": "str - user question",
      "retrieved_docs": [
        {
          "source": "str - document source",
          "content": "str - document text",
          "similarity_score": 0.0-1.0,
          "source_type": "str - type of source"
        }
      ],
      "answer": "str - generated answer",
      "judge_output": {
        "faithfulness": 0.0-1.0,
        "answer_relevance": 0.0-1.0,
        "context_utilization": 0.0-1.0,
        "medical_safety": 0.0-1.0,
        "clarity": 0.0-1.0,
        "mode_alignment": 0.0-1.0,
        "overall_score": 0.0-1.0,
        "verdict": "str"
      },
      "mode": "commercial|training",
      "training_scores": { optional - for training mode }
    }
    """
    
    def __init__(self):
        """Initialize all analyzers."""
        self.retrieval = RetrievalAnalyzer()
        self.generation = GenerationAuditor()
        self.judge = JudgeAnalyzer()
        self.safety = SafetyAuditor()
        self.trainer = TrainingAnalyzer()
        # XAI explainers
        self.shap_explainer = None  # Lazy loaded
        self.lime_explainer = None  # Lazy loaded
    
    def analyze(self, input_json: str) -> str:
        """
        Main analysis pipeline - accepts JSON string, returns JSON analysis.
        
        Args:
            input_json: JSON string with query, docs, answer, judge_output
        
        Returns:
            JSON string with 7 analysis reports
        """
        
        # Parse and validate input
        try:
            inputs = json.loads(input_json)
        except json.JSONDecodeError as e:
            return self._error_response(f"Invalid JSON input: {str(e)}")
        
        # Validate required fields
        validation = self._validate_inputs(inputs)
        if not validation["valid"]:
            return self._error_response(validation["error"])
        
        try:
            # Run analysis pipeline
            retrieval_analysis = self.retrieval.analyze(
                inputs.get("retrieved_docs", []),
                inputs.get("query", "")
            )
            
            generation_analysis = self.generation.audit(
                inputs.get("answer", ""),
                inputs.get("retrieved_docs", []),
                inputs.get("query", "")
            )
            
            judge_analysis = self.judge.analyze(
                inputs.get("judge_output", {}),
                retrieval_analysis.get("metrics", {}),
                generation_analysis
            )
            
            safety_analysis = self.safety.audit(
                inputs.get("answer", ""),
                inputs.get("retrieved_docs", []),
                generation_analysis
            )
            
            training_analysis = {}
            if inputs.get("training_scores"):
                training_analysis = self.trainer.analyze(
                    inputs.get("training_scores", {}),
                    judge_analysis.get("scores", {}),
                    generation_analysis,
                    safety_analysis
                )

            # Generate SHAP and LIME explanations for all modes
            shap_explanation = explain_generation_with_shap(
                inputs.get("query", ""),
                inputs.get("retrieved_docs", []),
                inputs.get("answer", ""),
                inputs.get("judge_output", {})
            )

            lime_explanation = explain_retrieval_with_lime(
                inputs.get("query", ""),
                inputs.get("retrieved_docs", [])
            )
            
            # Generate final verdict
            final_verdict = self._generate_verdict(
                retrieval_analysis,
                generation_analysis,
                judge_analysis,
                safety_analysis
            )
            
            # Compile all outputs
            output = {
                "timestamp": datetime.now().isoformat(),
                "status": "SUCCESS",
                "summary_report": self._generate_summary(
                    retrieval_analysis,
                    generation_analysis,
                    judge_analysis,
                    safety_analysis
                ),
                "retrieval_analysis": retrieval_analysis,
                "generation_audit": generation_analysis,
                "judge_analysis": judge_analysis,
                "safety_audit": safety_analysis,
                "training_feedback": training_analysis if training_analysis else None,
                "xai_explanations": {
                    "shap_generation_explanation": shap_explanation,
                    "lime_retrieval_explanation": lime_explanation
                },
                "final_verdict": final_verdict
            }
            
            return json.dumps(output, indent=2, default=str)
        
        except Exception as e:
            return self._error_response(f"Analysis pipeline error: {str(e)}")
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input structure."""
        required = ["query", "retrieved_docs", "answer", "judge_output"]
        
        for field in required:
            if field not in inputs:
                return {"valid": False, "error": f"Missing required field: {field}"}
        
        # Validate types
        if not isinstance(inputs.get("query"), str) or not inputs["query"].strip():
            return {"valid": False, "error": "Query must be non-empty string"}
        
        if not isinstance(inputs.get("retrieved_docs"), list):
            return {"valid": False, "error": "retrieved_docs must be list"}
        
        if not isinstance(inputs.get("answer"), str) or not inputs["answer"].strip():
            return {"valid": False, "error": "Answer must be non-empty string"}
        
        if not isinstance(inputs.get("judge_output"), dict):
            return {"valid": False, "error": "judge_output must be dict"}
        
        return {"valid": True}
    
    def _generate_verdict(self, retrieval: Dict, generation: Dict, judge: Dict, safety: Dict) -> Dict[str, Any]:
        """Generate final reliability verdict."""
        
        # Collect risk indicators
        risk_indicators = []
        
        # Check retrieval quality
        if retrieval.get("risk_level") in ["HIGH", "CRITICAL"]:
            risk_indicators.append({
                "indicator": "POOR_RETRIEVAL",
                "description": "Retrieved documents have low quality or credibility",
                "severity": retrieval.get("risk_level")
            })
        
        # Check generation consistency
        if generation.get("consistency_score", 1.0) < 0.65:
            risk_indicators.append({
                "indicator": "LOW_CONSISTENCY",
                "description": "Generated answer has unsupported or inconsistent claims",
                "severity": "HIGH"
            })
        
        # Check for hallucinations
        if generation.get("hallucinations_detected"):
            risk_indicators.append({
                "indicator": "HALLUCINATIONS",
                "description": f"{generation.get('hallucination_count', 0)} hallucinations detected",
                "severity": "CRITICAL" if generation.get("hallucination_count", 0) > 2 else "HIGH"
            })
        
        # Check judge alignment
        if judge.get("alignment_score", 1.0) < 0.65:
            risk_indicators.append({
                "indicator": "JUDGE_MISALIGNMENT",
                "description": "Judge scoring doesn't align with content quality",
                "severity": "MEDIUM"
            })
        
        # Check safety issues
        if safety.get("risk_level") in ["CRITICAL", "HIGH"]:
            risk_indicators.append({
                "indicator": "MEDICAL_SAFETY_CONCERN",
                "description": f"{len(safety.get('critical_claims', []))} critical safety claims detected",
                "severity": safety.get("risk_level")
            })
        
        if safety.get("missing_warnings"):
            risk_indicators.append({
                "indicator": "MISSING_SAFETY_INFO",
                "description": f"Missing {len(safety.get('missing_warnings', []))} critical warnings",
                "severity": "HIGH"
            })
        
        # Determine overall reliability
        reliability = self._determine_reliability(risk_indicators)
        
        return {
            "reliability": reliability,
            "confidence_level": self._calculate_confidence(
                judge.get("confidence", "LOW"),
                len(risk_indicators)
            ),
            "risk_indicators": risk_indicators,
            "recommendation": self._get_recommendation(reliability, risk_indicators),
            "clinical_review_required": reliability != "HIGH",
            "component_status": {
                "retrieval": retrieval.get("quality", "UNKNOWN"),
                "generation": generation.get("consistency_level", "UNKNOWN"),
                "judge": judge.get("alignment", {}).get("status", "UNKNOWN"),
                "safety": safety.get("safety_status", "UNKNOWN")
            }
        }
    
    def _determine_reliability(self, risk_indicators: list) -> str:
        """Determine reliability based on risk indicators."""
        if not risk_indicators:
            return "HIGH"
        
        critical_count = len([r for r in risk_indicators if r.get("severity") == "CRITICAL"])
        high_count = len([r for r in risk_indicators if r.get("severity") == "HIGH"])
        
        if critical_count > 0:
            return "LOW"
        elif high_count > 1:
            return "LOW"
        elif high_count == 1:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_confidence(self, judge_confidence: str, risk_count: int) -> str:
        """Calculate overall confidence."""
        if judge_confidence == "HIGH" and risk_count == 0:
            return "VERY_HIGH"
        elif judge_confidence == "HIGH" or risk_count <= 1:
            return "HIGH"
        elif judge_confidence == "MEDIUM" or risk_count <= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommendation(self, reliability: str, risk_indicators: list) -> str:
        """Generate recommendation."""
        if reliability == "HIGH":
            return "✓ SAFE TO USE - All quality checks passed"
        elif reliability == "MEDIUM":
            issues = ", ".join([r["indicator"] for r in risk_indicators[:2]])
            return f"⚠️  NEEDS VERIFICATION - Issues: {issues}"
        else:
            critical_issues = [r for r in risk_indicators if r["severity"] == "CRITICAL"]
            if critical_issues:
                return "🚨 NOT SAFE - Critical issues detected. Requires clinical expert review."
            else:
                return "❌ LOW RELIABILITY - Multiple concerns detected. Clinical review required."
    
    def _generate_summary(self, retrieval: Dict, generation: Dict, judge: Dict, safety: Dict) -> Dict[str, Any]:
        """Generate executive summary."""
        return {
            "system_status": "OPERATIONAL",
            "analysis_timestamp": datetime.now().isoformat(),
            "key_metrics": {
                "retrieval_quality": retrieval.get("quality", "UNKNOWN"),
                "generation_consistency": f"{generation.get('consistency_score', 0):.2%}",
                "judge_alignment": f"{judge.get('alignment_score', 0):.2%}",
                "safety_risk_level": safety.get("risk_level", "UNKNOWN")
            },
            "component_summary": {
                "retrieval": f"{len(retrieval.get('documents', []))} docs retrieved, avg similarity {retrieval.get('metrics', {}).get('avg_similarity', 0):.3f}",
                "generation": f"{generation.get('total_claims', 0)} claims made, {generation.get('unsupported_claims', 0)} unsupported",
                "judge": f"Overall score: {judge.get('overall_score', 0):.2f}, Confidence: {judge.get('confidence', 'UNKNOWN')}",
                "safety": f"Risk level: {safety.get('risk_level', 'UNKNOWN')}, {len(safety.get('critical_claims', []))} critical issues"
            },
            "highlights": self._generate_highlights(retrieval, generation, judge, safety)
        }
    
    def _generate_highlights(self, retrieval: Dict, generation: Dict, judge: Dict, safety: Dict) -> List[str]:
        """Generate key highlights."""
        highlights = []
        
        if retrieval.get("quality") == "EXCELLENT":
            highlights.append("✓ Excellent retrieval quality with credible sources")
        
        if generation.get("consistency_score", 0) > 0.90:
            highlights.append("✓ High answer consistency with retrieved evidence")
        
        if judge.get("confidence") == "HIGH":
            highlights.append(f"✓ High judge confidence (score: {judge.get('overall_score', 0):.2f})")
        
        if safety.get("risk_level") == "LOW":
            highlights.append("✓ No critical safety concerns detected")
        
        if generation.get("hallucinations_detected"):
            highlights.append(f"⚠️  {generation.get('hallucination_count', 0)} hallucinations need review")
        
        if safety.get("clinical_review_required"):
            highlights.append("⚠️  Clinical review recommended before use")
        
        return highlights
    
    def _error_response(self, error_message: str) -> str:
        """Generate error response in standard format."""
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "status": "ERROR",
            "error": error_message,
            "final_verdict": {
                "reliability": "UNKNOWN",
                "recommendation": "Cannot proceed - fix input errors"
            }
        }
        return json.dumps(error_response, indent=2)


# Convenience function for direct usage
def analyze_system(input_json: str) -> str:
    """Quick analysis wrapper."""
    pipeline = XAIPipeline()
    return pipeline.analyze(input_json)
