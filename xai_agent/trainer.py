"""
XAI Training Analyzer - Intelligent feedback for training mode
"""

from typing import Any, Dict, List


class TrainingAnalyzer:
    """
    Advanced training mode analysis with score breakdown and improvement suggestions.
    """
    
    def analyze(self, training_scores: Dict[str, Any], judge_scores: Dict[str, float], generation_metrics: Dict, safety_metrics: Dict) -> Dict[str, Any]:
        """Comprehensive training feedback analysis."""
        
        # Normalize scores
        normalized_scores = self._normalize_scores(training_scores)
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(normalized_scores)
        weaknesses = self._identify_weaknesses(normalized_scores)
        
        # Generate targeted improvement suggestions
        suggestions = self._generate_suggestions(
            weaknesses,
            generation_metrics,
            safety_metrics,
            judge_scores
        )
        
        # Calculate progression indicators
        progression = self._assess_progression(normalized_scores)
        
        analysis = {
            "total_score": round(sum(normalized_scores.values()) / len(normalized_scores), 3) if normalized_scores else 0.0,
            "score_breakdown": normalized_scores,
            "score_levels": self._categorize_scores(normalized_scores),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_suggestions": suggestions,
            "focus_areas": self._identify_focus_areas(weaknesses),
            "progression": progression,
            "training_status": self._determine_training_status(normalized_scores),
            "next_steps": self._generate_next_steps(normalized_scores, weaknesses)
        }
        
        return analysis
    
    def _normalize_scores(self, scores: Dict[str, Any]) -> Dict[str, float]:
        """Normalize training scores to 0.0-1.0 range."""
        normalized = {}
        
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                normalized[key] = max(0.0, min(1.0, float(value)))
        
        return normalized
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify strong areas (>0.80)."""
        strengths = []
        
        for metric, score in scores.items():
            if score > 0.80:
                level = "EXCELLENT" if score > 0.90 else "VERY GOOD"
                strengths.append(f"{metric}: {level} ({score:.2f})")
        
        return strengths
    
    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify weak areas (<0.70)."""
        weaknesses = []
        
        for metric, score in scores.items():
            if score < 0.70:
                severity = "URGENT" if score < 0.50 else "NEEDS_WORK"
                weaknesses.append({
                    "metric": metric,
                    "score": score,
                    "severity": severity
                })
        
        return sorted(weaknesses, key=lambda x: x["score"])
    
    def _generate_suggestions(self, weaknesses: List[Dict], generation: Dict, safety: Dict, judge: Dict) -> List[str]:
        """Generate targeted improvement suggestions."""
        suggestions = []
        
        for weakness in weaknesses:
            metric = weakness["metric"].lower()
            score = weakness["score"]
            
            if "correctness" in metric and score < 0.60:
                suggestions.append("⚡ CORRECTNESS: Verify answers against authoritative medical sources (FDA, clinical guidelines)")
                suggestions.append("   → Check for factual accuracy of all medical claims")
                suggestions.append("   → Cross-reference dosages, indications, contraindications")
            
            if "completeness" in metric and score < 0.60:
                suggestions.append("⚡ COMPLETENESS: Ensure answer covers all aspects of the question")
                suggestions.append("   → Include necessary warnings, contraindications, special populations")
                suggestions.append("   → Mention relevant interactions or alternative treatments")
            
            if "safety" in metric and score < 0.60:
                suggestions.append("⚡ SAFETY: Critical medical information gaps detected")
                suggestions.append("   → Add warnings for specific populations (pregnant, elderly, pediatric)")
                suggestions.append("   → Highlight adverse reactions and overdose risks")
                suggestions.append("   → Flag contraindications prominently")
            
            if "clarity" in metric and score < 0.60:
                suggestions.append("⚡ CLARITY: Improve answer structure and readability")
                suggestions.append("   → Use bullet points for key information")
                suggestions.append("   → Break into sections: indication, dosage, warnings, interactions")
                suggestions.append("   → Avoid jargon or define medical terms")
            
            if "evidence" in metric and score < 0.60:
                suggestions.append("⚡ EVIDENCE GROUNDING: Connect answer to retrieved sources")
                suggestions.append("   → Cite specific guidelines or studies")
                suggestions.append("   → Avoid unsupported claims")
                suggestions.append("   → Use language matching retrieved documents")
        
        # Add contextual suggestions from safety/generation metrics
        if safety.get("clinical_review_required"):
            suggestions.append("🔴 MEDICAL REVIEW: Answer requires clinical expert review before use")
        
        if generation.get("hallucinations_detected"):
            suggestions.append(f"⚠️  HALLUCINATIONS: Remove {generation.get('hallucination_count', 0)} unsupported claims")
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def _identify_focus_areas(self, weaknesses: List[Dict]) -> List[str]:
        """Identify primary areas for improvement."""
        focus = []
        
        if not weaknesses:
            focus.append("✓ No major weaknesses - focus on incremental improvement")
        elif len(weaknesses) >= 3:
            focus.append(f"Multiple areas need work ({len(weaknesses)}). Prioritize:")
            for w in weaknesses[:3]:
                focus.append(f"  1. {w['metric'].upper()} (current: {w['score']:.2f})")
        else:
            for w in weaknesses:
                focus.append(f"{w['metric'].upper()}: Improve from {w['score']:.2f} to 0.75+")
        
        return focus
    
    def _assess_progression(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess training progression."""
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        
        progression_level = "BEGINNER" if avg_score < 0.60 else (
            "INTERMEDIATE" if avg_score < 0.75 else (
                "ADVANCED" if avg_score < 0.88 else "EXPERT"
            )
        )
        
        return {
            "level": progression_level,
            "average_score": round(avg_score, 3),
            "estimated_gap_to_expert": round(1.0 - avg_score, 3),
            "next_milestone": "75% (INTERMEDIATE)" if avg_score < 0.75 else "88% (ADVANCED)" if avg_score < 0.88 else "95% (EXPERT)"
        }
    
    def _categorize_scores(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Categorize scores by level."""
        categories = {
            "EXCELLENT": [],
            "GOOD": [],
            "ACCEPTABLE": [],
            "NEEDS_WORK": [],
            "CRITICAL": []
        }
        
        for metric, score in scores.items():
            if score >= 0.90:
                categories["EXCELLENT"].append(metric)
            elif score >= 0.80:
                categories["GOOD"].append(metric)
            elif score >= 0.70:
                categories["ACCEPTABLE"].append(metric)
            elif score >= 0.50:
                categories["NEEDS_WORK"].append(metric)
            else:
                categories["CRITICAL"].append(metric)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _determine_training_status(self, scores: Dict[str, float]) -> str:
        """Determine overall training status."""
        avg = sum(scores.values()) / len(scores) if scores else 0
        
        if avg >= 0.88:
            return "READY_FOR_PRODUCTION"
        elif avg >= 0.75:
            return "ADVANCED_TRAINING"
        elif avg >= 0.60:
            return "INTERMEDIATE_TRAINING"
        else:
            return "BEGINNING_TRAINING"
    
    def _generate_next_steps(self, scores: Dict[str, float], weaknesses: List[Dict]) -> List[str]:
        """Generate next steps for trainee."""
        next_steps = []
        avg = sum(scores.values()) / len(scores) if scores else 0
        
        if avg < 0.60:
            next_steps.append("1. FOUNDATION: Focus on medical accuracy and evidence grounding")
            next_steps.append("2. STUDY: Review medical guidelines and clinical evidence")
            next_steps.append("3. PRACTICE: Generate answers with source verification")
        
        elif avg < 0.75:
            next_steps.append("1. TARGETED IMPROVEMENT: Address specific weaknesses")
            for w in weaknesses[:2]:
                next_steps.append(f"   - Improve {w['metric']}: {w['score']:.2f} → 0.80+")
            next_steps.append("2. QUALITY CHECK: Ensure completeness and safety coverage")
        
        elif avg < 0.88:
            next_steps.append("1. POLISH: Fine-tune clarity and structure")
            next_steps.append("2. EDGE CASES: Handle complex or rare scenarios")
            next_steps.append("3. CONSISTENCY: Maintain quality across diverse queries")
        
        else:
            next_steps.append("✓ READY FOR PRODUCTION: Proceed to live deployment")
            next_steps.append("→ Continue monitoring and refining based on feedback")
        
        return next_steps
