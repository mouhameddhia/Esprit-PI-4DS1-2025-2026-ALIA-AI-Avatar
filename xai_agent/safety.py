"""
XAI Safety Auditor - Intelligent medical risk detection
"""

from typing import Any, Dict, List
import re


class SafetyAuditor:
    """
    Advanced safety analysis with pattern-based risk detection and clinical concern flagging.
    """
    
    # Critical safety keywords
    CRITICAL_KEYWORDS = {
        "fatal": 1.0,
        "death": 1.0,
        "lethal": 1.0,
        "overdose": 0.9,
        "anaphylaxis": 0.9,
        "contraindicated": 0.8,
        "prohibited": 0.8,
        "forbidden": 0.8,
    }
    
    # High-risk keywords
    HIGH_RISK_KEYWORDS = {
        "severe": 0.7,
        "dangerous": 0.7,
        "toxic": 0.7,
        "allergy": 0.6,
        "adverse": 0.6,
        "interaction": 0.5,
    }
    
    # Missing safety information patterns
    REQUIRED_WARNINGS = {
        "pregnancy": ["pregnant", "pregnancy", "fetus", "trimester"],
        "children": ["child", "children", "pediatric", "infant", "neonate"],
        "elderly": ["elderly", "geriatric", "age", "senior"],
        "renal": ["kidney", "renal", "creatinine", "GFR"],
        "hepatic": ["liver", "hepatic", "cirrhosis", "fibrosis"],
    }
    
    def audit(self, answer: str, retrieved_docs: List[Dict[str, Any]], generation_metrics: Dict) -> Dict[str, Any]:
        """Comprehensive medical safety audit."""
        
        context_text = " ".join([d.get("content", "") for d in retrieved_docs])
        
        # Detect various safety issues
        critical_claims = self._find_critical_claims(answer)
        high_risk_claims = self._find_high_risk_claims(answer)
        missing_warnings = self._check_missing_warnings(answer, context_text)
        hallucinations = generation_metrics.get("hallucinations", [])
        
        # Calculate safety risk score
        risk_score = self._calculate_risk_score(
            critical_claims,
            high_risk_claims,
            missing_warnings,
            len(hallucinations)
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        analysis = {
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "critical_concerns": len(critical_claims),
            "critical_claims": critical_claims,
            "high_risk_concerns": len(high_risk_claims),
            "high_risk_claims": high_risk_claims,
            "missing_warnings": missing_warnings,
            "hallucinations_medical_impact": self._assess_hallucination_impact(hallucinations),
            "clinical_review_required": risk_level in ["CRITICAL", "HIGH"],
            "safety_status": self._determine_safety_status(risk_level),
            "recommendations": self._generate_recommendations(
                critical_claims,
                high_risk_claims,
                missing_warnings,
                risk_level
            ),
            "issues": self._compile_safety_issues(
                critical_claims,
                high_risk_claims,
                missing_warnings,
                hallucinations
            )
        }
        
        return analysis
    
    def _find_critical_claims(self, text: str) -> List[Dict[str, Any]]:
        """Find claims with critical safety keywords."""
        claims = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword, severity in self.CRITICAL_KEYWORDS.items():
                if keyword in sentence_lower:
                    claims.append({
                        "keyword": keyword,
                        "severity": "CRITICAL",
                        "score": severity,
                        "text": sentence.strip()
                    })
                    break
        
        return claims
    
    def _find_high_risk_claims(self, text: str) -> List[Dict[str, Any]]:
        """Find claims with high-risk keywords."""
        claims = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword, severity in self.HIGH_RISK_KEYWORDS.items():
                if keyword in sentence_lower:
                    claims.append({
                        "keyword": keyword,
                        "severity": "HIGH",
                        "score": severity,
                        "text": sentence.strip()
                    })
                    break
        
        return claims
    
    def _check_missing_warnings(self, answer: str, context: str) -> List[Dict[str, Any]]:
        """Check for missing critical safety warnings."""
        missing = []
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        for condition, keywords in self.REQUIRED_WARNINGS.items():
            condition_mentioned = any(kw in answer_lower for kw in keywords)
            
            if condition_mentioned:
                # Check if specific warning is in context
                warning_keywords = [
                    f"{condition} warning", 
                    f"avoid in {condition}",
                    f"caution in {condition}"
                ]
                warning_present = any(w in context_lower for w in warning_keywords)
                
                if not warning_present:
                    missing.append({
                        "condition": condition,
                        "message": f"Missing safety information for {condition}",
                        "severity": "HIGH"
                    })
        
        return missing
    
    def _calculate_risk_score(self, critical: List, high_risk: List, missing_warnings: List, hallucination_count: int) -> float:
        """Calculate overall safety risk score."""
        score = 0.0
        
        # Critical claims heavily increase risk
        score += len(critical) * 0.4
        
        # High-risk claims moderate risk
        score += len(high_risk) * 0.2
        
        # Missing warnings increase risk
        score += len(missing_warnings) * 0.15
        
        # Hallucinations in medical context are critical
        score += hallucination_count * 0.35
        
        return min(1.0, score)  # Clamp to 0.0-1.0
    
    def _determine_risk_level(self, score: float) -> str:
        """Map risk score to level."""
        if score >= 0.75:
            return "CRITICAL"
        elif score >= 0.50:
            return "HIGH"
        elif score >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_hallucination_impact(self, hallucinations: List[Dict]) -> Dict[str, Any]:
        """Assess medical impact of hallucinations."""
        if not hallucinations:
            return {"count": 0, "impact": "NONE", "severity": "LOW"}
        
        critical_hallucinations = [h for h in hallucinations if h.get("severity") == "CRITICAL"]
        
        if critical_hallucinations:
            return {
                "count": len(hallucinations),
                "critical_count": len(critical_hallucinations),
                "impact": "SEVERE",
                "severity": "CRITICAL",
                "message": "Hallucinations with medical safety implications"
            }
        
        return {
            "count": len(hallucinations),
            "impact": "MODERATE",
            "severity": "HIGH",
            "message": "Hallucinations present - clinical review needed"
        }
    
    def _determine_safety_status(self, risk_level: str) -> str:
        """Determine overall safety status."""
        status_map = {
            "CRITICAL": "NOT_SAFE",
            "HIGH": "NEEDS_REVIEW",
            "MEDIUM": "CAUTION_ADVISED",
            "LOW": "SAFE"
        }
        return status_map.get(risk_level, "UNKNOWN")
    
    def _generate_recommendations(self, critical: List, high_risk: List, missing: List, risk_level: str) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.append("🚨 CRITICAL: Do NOT use without clinical expert review")
            if critical:
                recommendations.append(f"Remove or verify {len(critical)} critical safety claims")
            if missing:
                recommendations.append(f"Add {len(missing)} missing safety warnings before use")
        
        elif risk_level == "HIGH":
            recommendations.append("⚠️  HIGH RISK: Requires clinical verification")
            if high_risk:
                recommendations.append(f"Verify {len(high_risk)} high-risk medical claims")
            if missing:
                recommendations.append(f"Supplement with {len(missing)} missing warnings")
        
        elif risk_level == "MEDIUM":
            recommendations.append("⚠️  Review recommended before clinical use")
            if missing:
                recommendations.append(f"Consider adding information for: {', '.join([m['condition'] for m in missing])}")
        
        else:
            recommendations.append("✓ Safety profile appears acceptable")
        
        return recommendations
    
    def _compile_safety_issues(self, critical: List, high_risk: List, missing: List, hallucinations: List) -> List[str]:
        """Compile all safety issues."""
        issues = []
        
        if critical:
            for c in critical:
                issues.append(f"CRITICAL: {c['text'][:80]}... (contains '{c['keyword']}')")
        
        if high_risk:
            for h in high_risk:
                issues.append(f"HIGH RISK: {h['text'][:80]}... (contains '{h['keyword']}')")
        
        if missing:
            for m in missing:
                issues.append(f"MISSING: {m['message']}")
        
        if hallucinations:
            for h in hallucinations:
                if h.get("severity") in ["CRITICAL", "HIGH"]:
                    issues.append(f"UNSUPPORTED CLAIM: {h['text'][:80]}...")
        
        return issues
