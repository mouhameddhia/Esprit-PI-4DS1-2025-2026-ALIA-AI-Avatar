"""
XAI Generation Auditor - Intelligent answer consistency analysis
"""

from typing import Any, Dict, List
import re


class GenerationAuditor:
    """
    Advanced generation analysis with semantic claim verification and hallucination detection.
    """
    
    # Medical keywords for claim extraction
    CLAIM_TRIGGERS = {
        "dosage_keywords": ["dose", "dosage", "mg", "microgram", "mcg", "unit", "IU"],
        "indication_keywords": ["treat", "treatment", "indicated", "indication", "use for", "used for"],
        "contraindication_keywords": ["contraindicated", "avoid", "not use", "don't use", "forbidden", "prohibited"],
        "side_effect_keywords": ["side effect", "adverse", "reaction", "complication", "symptom"],
        "interaction_keywords": ["interact", "interaction", "combined with", "together with"],
    }
    
    def audit(self, answer: str, retrieved_docs: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Intelligent generation audit with semantic verification."""
        
        context_text = self._build_context(retrieved_docs)
        
        # Extract claims from answer
        claims = self._extract_medical_claims(answer)
        
        # Verify each claim
        verified_claims = []
        unsupported_claims = []
        hallucinations = []
        
        for claim_type, claim_text in claims:
            supported = self._verify_claim(claim_text, context_text)
            severity = self._assess_claim_severity(claim_text)
            
            claim_obj = {
                "type": claim_type,
                "text": claim_text,
                "supported": supported,
                "severity": severity
            }
            
            if supported:
                verified_claims.append(claim_obj)
            else:
                unsupported_claims.append(claim_obj)
                if severity in ["CRITICAL", "HIGH"]:
                    hallucinations.append(claim_obj)
        
        # Calculate consistency metrics
        total_claims = len(claims)
        consistency_score = (len(verified_claims) / total_claims) if total_claims > 0 else 1.0
        
        analysis = {
            "total_claims": total_claims,
            "verified_claims": len(verified_claims),
            "unsupported_claims": len(unsupported_claims),
            "consistency_score": round(consistency_score, 3),
            "consistency_level": self._score_consistency(consistency_score),
            "hallucinations_detected": len(hallucinations) > 0,
            "hallucination_count": len(hallucinations),
            "hallucinations": hallucinations,
            "evidence_coverage": self._assess_evidence_coverage(answer, context_text),
            "logical_flow": self._assess_logical_flow(answer),
            "issues": self._identify_generation_issues(
                consistency_score, 
                len(hallucinations),
                self._assess_evidence_coverage(answer, context_text)
            )
        }
        
        return analysis
    
    def _extract_medical_claims(self, text: str) -> List[tuple]:
        """Extract medical claims with type classification."""
        claims = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Classify claim type
            claim_type = self._classify_claim(sentence)
            if claim_type:
                claims.append((claim_type, sentence))
        
        return claims
    
    def _classify_claim(self, sentence: str) -> str:
        """Classify the type of medical claim."""
        sentence_lower = sentence.lower()
        
        for claim_cat, keywords in self.CLAIM_TRIGGERS.items():
            if any(kw in sentence_lower for kw in keywords):
                return claim_cat.replace("_keywords", "").upper()
        
        return None
    
    def _verify_claim(self, claim: str, context: str) -> bool:
        """Verify if claim is supported by context."""
        # Extract key phrases from claim
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        # Calculate overlap
        overlap = claim_words & context_words
        overlap_ratio = len(overlap) / len(claim_words) if claim_words else 0
        
        # Semantic verification: if 40%+ of claim words appear in context, likely supported
        return overlap_ratio > 0.4
    
    def _assess_claim_severity(self, claim: str) -> str:
        """Assess severity of unsupported claim."""
        critical_words = ["fatal", "death", "lethal", "contraindicated", "prohibited"]
        high_words = ["severe", "dangerous", "risk", "toxic", "overdose", "allergy"]
        
        claim_lower = claim.lower()
        
        for word in critical_words:
            if word in claim_lower:
                return "CRITICAL"
        
        for word in high_words:
            if word in claim_lower:
                return "HIGH"
        
        return "MEDIUM"
    
    def _score_consistency(self, score: float) -> str:
        """Rate consistency level."""
        if score > 0.90:
            return "EXCELLENT"
        elif score > 0.75:
            return "GOOD"
        elif score > 0.60:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def _assess_evidence_coverage(self, answer: str, context: str) -> float:
        """Assess how well answer is backed by context."""
        answer_words = set(w for w in answer.lower().split() if len(w) > 3)
        context_words = set(w for w in context.lower().split() if len(w) > 3)
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        coverage = overlap / len(answer_words)
        
        return round(min(1.0, coverage), 3)
    
    def _assess_logical_flow(self, answer: str) -> str:
        """Assess logical structure of answer."""
        # Simple heuristics for logical flow
        sentences = len(re.split(r'[.!?]', answer))
        
        if sentences < 2:
            return "TOO_SHORT"
        elif sentences > 15:
            return "TOO_LONG"
        else:
            return "ADEQUATE"
    
    def _identify_generation_issues(self, consistency: float, hallucination_count: int, coverage: float) -> List[str]:
        """Identify specific generation problems."""
        issues = []
        
        if hallucination_count > 0:
            issues.append(f"HALLUCINATION: {hallucination_count} unsupported claims detected")
        
        if consistency < 0.6:
            issues.append(f"LOW CONSISTENCY: Only {int(consistency*100)}% of claims verified")
        
        if coverage < 0.4:
            issues.append(f"POOR EVIDENCE COVERAGE: Only {int(coverage*100)}% of answer grounded in context")
        
        if hallucination_count > 2:
            issues.append("CRITICAL: Multiple severe hallucinations - medical review required")
        
        return issues
    
    def _build_context(self, docs: List[Dict]) -> str:
        """Build full context string from documents."""
        return " ".join([d.get("content", "") for d in docs])
