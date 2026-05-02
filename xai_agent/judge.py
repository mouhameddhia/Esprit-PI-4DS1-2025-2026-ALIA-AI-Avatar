"""
XAI Judge Analyzer - Intelligent judge evaluation validation
"""

from typing import Any, Dict, List
import statistics


class JudgeAnalyzer:
    """
    Advanced judge analysis with scoring consistency validation and alignment checking.
    """
    
    def analyze(self, judge_output: Dict[str, Any], retrieval_metrics: Dict, generation_metrics: Dict) -> Dict[str, Any]:
        """Intelligent judge analysis with cross-component validation."""
        
        # Extract judge scores
        scores = self._extract_scores(judge_output)
        
        # Validate scoring consistency
        consistency_check = self._validate_consistency(scores)
        
        # Check alignment with retrieval quality
        retrieval_alignment = self._check_retrieval_alignment(
            scores.get("faithfulness", 0),
            retrieval_metrics.get("metrics", {}).get("avg_similarity", 0)
        )
        
        # Check alignment with generation quality
        generation_alignment = self._check_generation_alignment(
            scores.get("answer_relevance", 0),
            generation_metrics.get("consistency_score", 0)
        )
        
        # Validate medical safety scoring
        safety_validation = self._validate_safety_scoring(
            scores.get("medical_safety", 0),
            generation_metrics.get("hallucinations_detected", False)
        )
        
        analysis = {
            "scores": scores,
            "overall_score": scores.get("overall_score", 0),
            "verdict": judge_output.get("verdict", "unknown"),
            "consistency": consistency_check,
            "alignment": {
                "retrieval": retrieval_alignment,
                "generation": generation_alignment,
                "safety": safety_validation
            },
            "alignment_score": round(
                (retrieval_alignment["score"] + 
                 generation_alignment["score"] + 
                 safety_validation["score"]) / 3,
                3
            ),
            "confidence": self._assess_judge_confidence(scores, consistency_check),
            "issues": self._identify_judge_issues(
                scores, 
                consistency_check,
                retrieval_alignment,
                generation_alignment,
                safety_validation
            )
        }
        
        return analysis
    
    def _extract_scores(self, judge_output: Dict[str, Any]) -> Dict[str, float]:
        """Extract and validate judge scores."""
        required_metrics = [
            "faithfulness",
            "answer_relevance",
            "context_utilization",
            "medical_safety",
            "clarity",
            "mode_alignment"
        ]
        
        scores = {}
        for metric in required_metrics:
            value = judge_output.get(metric, 0.0)
            # Clamp to 0.0-1.0
            scores[metric] = max(0.0, min(1.0, float(value)))
        
        # Calculate overall if not provided
        scores["overall_score"] = judge_output.get(
            "overall_score",
            statistics.mean(scores.values())
        )
        
        return scores
    
    def _validate_consistency(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Check internal scoring consistency."""
        core_scores = [
            scores.get("faithfulness", 0),
            scores.get("answer_relevance", 0),
            scores.get("medical_safety", 0)
        ]
        
        if not core_scores:
            return {"status": "INVALID", "variance": 0}
        
        variance = statistics.variance(core_scores) if len(core_scores) > 1 else 0
        stdev = statistics.stdev(core_scores) if len(core_scores) > 1 else 0
        
        consistency_status = "HIGH" if stdev < 0.2 else ("MEDIUM" if stdev < 0.4 else "LOW")
        
        return {
            "status": consistency_status,
            "variance": round(variance, 3),
            "stdev": round(stdev, 3),
            "mean": round(statistics.mean(core_scores), 3)
        }
    
    def _check_retrieval_alignment(self, faithfulness_score: float, retrieval_quality: float) -> Dict[str, Any]:
        """Validate faithfulness score against retrieval quality."""
        alignment_diff = abs(faithfulness_score - retrieval_quality)
        
        # If retrieval is poor, faithfulness should be lower
        # If retrieval is good, faithfulness can be higher
        is_aligned = alignment_diff < 0.25
        
        return {
            "score": 1.0 if is_aligned else 0.5,
            "aligned": is_aligned,
            "judge_score": faithfulness_score,
            "retrieval_quality": retrieval_quality,
            "difference": round(alignment_diff, 3),
            "status": "ALIGNED" if is_aligned else "MISALIGNED"
        }
    
    def _check_generation_alignment(self, relevance_score: float, generation_consistency: float) -> Dict[str, Any]:
        """Validate relevance score against generation consistency."""
        alignment_diff = abs(relevance_score - generation_consistency)
        is_aligned = alignment_diff < 0.25
        
        return {
            "score": 1.0 if is_aligned else 0.5,
            "aligned": is_aligned,
            "judge_score": relevance_score,
            "generation_consistency": generation_consistency,
            "difference": round(alignment_diff, 3),
            "status": "ALIGNED" if is_aligned else "MISALIGNED"
        }
    
    def _validate_safety_scoring(self, safety_score: float, has_hallucinations: bool) -> Dict[str, Any]:
        """Validate medical safety scoring."""
        # If hallucinations detected, safety should be low
        if has_hallucinations and safety_score > 0.7:
            return {
                "score": 0.3,
                "valid": False,
                "issue": "High safety score despite hallucinations detected",
                "status": "CONCERNING"
            }
        
        # If no hallucinations, safety can be high
        if not has_hallucinations and safety_score < 0.5:
            return {
                "score": 0.5,
                "valid": False,
                "issue": "Low safety score despite no hallucinations",
                "status": "QUESTIONABLE"
            }
        
        return {
            "score": 1.0,
            "valid": True,
            "issue": None,
            "status": "VALID"
        }
    
    def _assess_judge_confidence(self, scores: Dict[str, float], consistency: Dict) -> str:
        """Assess confidence in judge's verdict."""
        overall = scores.get("overall_score", 0.5)
        consistency_status = consistency.get("status", "LOW")
        
        # High confidence if: consistent scoring + clear signal (very high or very low)
        if consistency_status == "HIGH" and (overall > 0.85 or overall < 0.35):
            return "HIGH"
        elif consistency_status == "MEDIUM" or (0.4 < overall < 0.8):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_judge_issues(self, scores: Dict, consistency: Dict, ret_align: Dict, gen_align: Dict, safe_val: Dict) -> List[str]:
        """Identify judge evaluation problems."""
        issues = []
        
        # Consistency issues
        if consistency["status"] == "LOW":
            issues.append(f"Inconsistent scoring - high variance ({consistency['stdev']})")
        
        # Alignment issues
        if not ret_align["aligned"]:
            issues.append("Faithfulness score misaligned with retrieval quality")
        
        if not gen_align["aligned"]:
            issues.append("Relevance score misaligned with generation consistency")
        
        if not safe_val["valid"]:
            issues.append(f"Safety scoring concern: {safe_val['issue']}")
        
        # Extreme scores without justification
        if scores.get("overall_score", 0) > 0.95:
            issues.append("Very high overall score - may be overconfident")
        
        if scores.get("overall_score", 0) < 0.2:
            issues.append("Very low overall score - severe critique may be justified")
        
        return issues
