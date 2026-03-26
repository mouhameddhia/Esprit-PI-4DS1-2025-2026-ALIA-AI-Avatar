# evaluation_service.py
"""
Service Layer for Evaluation Agent

Provides high-level API for evaluation functionality
Handles request validation, agent orchestration, and response formatting
"""

from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.evaluation_agent.agent import EvaluationAgent
from agents.evaluation_agent.schemas import EvaluationRequest, EvaluationResponse


class EvaluationService:
    """
    Service layer for medical representative performance evaluation
    
    Provides a clean interface for the API layer to interact with the evaluation agent
    """
    
    def __init__(self, model_name: str = "llama3:8b", use_single_pass: bool = True):
        """
        Initialize the evaluation service
        
        Args:
            model_name: LLM model to use for evaluation
            use_single_pass: Whether to use single-pass evaluation (faster)
                           or multi-pass (more accurate)
        """
        self.agent = EvaluationAgent(
            model_name=model_name,
            use_single_pass=use_single_pass
        )
    
    def evaluate_transcript(
        self,
        transcript: str,
        product_name: Optional[str] = None,
        rep_name: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a dialogue transcript
        
        Args:
            transcript: Full dialogue transcript as string
            product_name: Optional product name discussed in dialogue
            rep_name: Optional representative name
            additional_metadata: Any additional context information
        
        Returns:
            Dictionary containing evaluation results
        
        Raises:
            ValueError: If transcript is empty or invalid
            Exception: If evaluation fails
        """
        
        # Validate input
        if not transcript or not isinstance(transcript, str):
            raise ValueError("Transcript must be a non-empty string")
        
        if len(transcript.strip()) < 50:
            raise ValueError("Transcript is too short to evaluate (minimum 50 characters)")
        
        # Build metadata
        metadata = {}
        if product_name:
            metadata["product_name"] = product_name
        if rep_name:
            metadata["rep_name"] = rep_name
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Process evaluation
        try:
            result = self.agent.process(
                transcript=transcript,
                metadata=metadata
            )
            
            # Add service metadata
            result["service_version"] = "1.0.0"
            result["evaluation_method"] = "single_pass" if self.agent.use_single_pass else "multi_pass"
            
            return result
            
        except Exception as e:
            raise Exception(f"Evaluation failed: {str(e)}")
    
    def evaluate_batch(
        self,
        transcripts: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Evaluate multiple transcripts in batch
        
        Args:
            transcripts: List of dictionaries containing:
                - transcript: str (required)
                - product_name: str (optional)
                - rep_name: str (optional)
                - metadata: dict (optional)
        
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, item in enumerate(transcripts):
            try:
                result = self.evaluate_transcript(
                    transcript=item.get("transcript", ""),
                    product_name=item.get("product_name"),
                    rep_name=item.get("rep_name"),
                    additional_metadata=item.get("metadata")
                )
                
                result["batch_index"] = i
                result["success"] = True
                results.append(result)
                
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def get_evaluation_summary(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a brief summary from a full evaluation result
        
        Args:
            evaluation_result: Full evaluation result dictionary
        
        Returns:
            Simplified summary dictionary
        """
        return {
            "overall_score": evaluation_result.get("overall_score"),
            "classification": evaluation_result.get("classification"),
            "top_strength": evaluation_result.get("strengths", ["N/A"])[0],
            "top_improvement": evaluation_result.get("improvement_areas", ["N/A"])[0],
            "compliance_issues": len(evaluation_result.get("compliance_flags", [])) > 0
        }
    
    def validate_compliance(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance status from evaluation result
        
        Args:
            evaluation_result: Full evaluation result dictionary
        
        Returns:
            Compliance validation result
        """
        compliance_flags = evaluation_result.get("compliance_flags", [])
        compliance_score = evaluation_result.get("dimension_scores", {}).get("compliance", 15)
        
        has_critical_violation = any(
            "CRITICAL" in flag or "off-label" in flag.lower()
            for flag in compliance_flags
        )
        
        return {
            "compliant": len(compliance_flags) == 0,
            "has_critical_violation": has_critical_violation,
            "compliance_score": compliance_score,
            "total_flags": len(compliance_flags),
            "flags": compliance_flags,
            "status": "FAIL" if has_critical_violation else ("PASS" if len(compliance_flags) == 0 else "WARNING")
        }
    
    def compare_evaluations(
        self,
        evaluation1: Dict[str, Any],
        evaluation2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two evaluation results (useful for tracking progress)
        
        Args:
            evaluation1: First evaluation result (typically older)
            evaluation2: Second evaluation result (typically newer)
        
        Returns:
            Comparison dictionary showing improvements/declines
        """
        score1 = evaluation1.get("overall_score", 0)
        score2 = evaluation2.get("overall_score", 0)
        
        dimensions1 = evaluation1.get("dimension_scores", {})
        dimensions2 = evaluation2.get("dimension_scores", {})
        
        dimension_changes = {}
        for dim in dimensions1.keys():
            change = dimensions2.get(dim, 0) - dimensions1.get(dim, 0)
            dimension_changes[dim] = {
                "previous": dimensions1.get(dim, 0),
                "current": dimensions2.get(dim, 0),
                "change": change,
                "improved": change > 0
            }
        
        return {
            "overall_change": score2 - score1,
            "improved": score2 > score1,
            "previous_score": score1,
            "current_score": score2,
            "previous_classification": evaluation1.get("classification"),
            "current_classification": evaluation2.get("classification"),
            "dimension_changes": dimension_changes,
            "improved_dimensions": [
                dim for dim, data in dimension_changes.items() if data["improved"]
            ],
            "declined_dimensions": [
                dim for dim, data in dimension_changes.items() if not data["improved"] and data["change"] < 0
            ]
        }


# ==================== Testing ====================

if __name__ == "__main__":
    """
    Test the evaluation service
    """
    
    print("=" * 80)
    print("🧪 Testing Evaluation Service")
    print("=" * 80)
    
    # Initialize service
    service = EvaluationService(use_single_pass=True)
    
    # Sample transcript
    sample_transcript = """
Rep: Good morning Dr. Smith! Thank you for seeing me today.
Doctor: Good morning. What can I help you with?
Rep: I'm here to discuss Cardiomax, our new cardiovascular medication that might benefit your patients with hypertension.
Doctor: Tell me about it.
Rep: Certainly. What are your current challenges with hypertensive patients?
Doctor: Compliance is always difficult.
Rep: I understand. Cardiomax is designed with once-daily dosing to help with that. It works by blocking angiotensin II receptors, providing effective blood pressure control. The recommended dose is 50mg daily.
Doctor: What about safety?
Rep: The safety profile is good. Common side effects include mild dizziness in about 5% of patients. It's contraindicated in pregnancy.
Doctor: Okay, I'll consider it.
Rep: Great! Let me leave some information and follow up next week. Thank you for your time!
"""
    
    print("\n📘 Test 1: Basic Evaluation")
    print("-" * 80)
    
    result = service.evaluate_transcript(
        transcript=sample_transcript,
        product_name="Cardiomax",
        rep_name="John Doe"
    )
    
    print(f"Overall Score: {result['overall_score']:.2f}")
    print(f"Classification: {result['classification']}")
    print(f"Service Version: {result['service_version']}")
    
    print("\n📗 Test 2: Evaluation Summary")
    print("-" * 80)
    
    summary = service.get_evaluation_summary(result)
    print(f"Summary: {summary}")
    
    print("\n📙 Test 3: Compliance Validation")
    print("-" * 80)
    
    compliance = service.validate_compliance(result)
    print(f"Compliance Status: {compliance['status']}")
    print(f"Compliant: {compliance['compliant']}")
    print(f"Compliance Score: {compliance['compliance_score']}")
    
    print("\n" + "=" * 80)
    print("✅ Service testing complete!")
    print("=" * 80)
