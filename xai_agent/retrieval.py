"""
XAI Retrieval Analyzer - Intelligent RAG quality assessment
"""

from typing import Any, Dict, List
import statistics


class RetrievalAnalyzer:
    """
    Advanced RAG analysis with semantic scoring and source credibility tracking.
    """
    
    # Medical source credibility tiers
    CREDIBLE_SOURCES = {
        "fda": 1.0,
        "ema": 1.0,
        "nih": 1.0,
        "clinical_guideline": 0.95,
        "clinical_study": 0.9,
        "medical_journal": 0.9,
        "pharmaceutical_database": 0.85,
        "medical_textbook": 0.85,
    }
    
    QUESTIONABLE_SOURCES = {
        "blog": 0.3,
        "forum": 0.2,
        "social_media": 0.1,
        "uncited": 0.1,
    }
    
    def analyze(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Intelligent retrieval analysis with multiple metrics."""
        
        if not documents:
            return {
                "quality": "CRITICAL",
                "status": "FAILED",
                "count": 0,
                "reason": "No documents retrieved",
                "metrics": {}
            }
        
        # Calculate multiple quality dimensions
        similarity_scores = [d.get("similarity_score", 0.0) for d in documents]
        credibility_scores = [self._assess_credibility(d) for d in documents]
        completeness_scores = [self._assess_completeness(d) for d in documents]
        
        analysis = {
            "count": len(documents),
            "documents": [self._analyze_document(d, idx) for idx, d in enumerate(documents)],
            "metrics": {
                "avg_similarity": round(statistics.mean(similarity_scores), 3),
                "median_similarity": round(statistics.median(similarity_scores), 3),
                "similarity_stdev": round(statistics.stdev(similarity_scores), 3) if len(similarity_scores) > 1 else 0.0,
                "avg_credibility": round(statistics.mean(credibility_scores), 3),
                "avg_completeness": round(statistics.mean(completeness_scores), 3),
                "high_quality_docs": len([s for s in similarity_scores if s > 0.8]),
                "coverage_ratio": self._assess_coverage(documents),
            }
        }
        
        # Intelligent quality assessment
        analysis["quality"] = self._determine_quality(analysis["metrics"])
        analysis["risk_level"] = self._determine_risk(analysis["metrics"])
        
        # Flag potential issues
        analysis["issues"] = self._identify_issues(documents, similarity_scores, credibility_scores)
        
        return analysis
    
    def _analyze_document(self, doc: Dict, idx: int) -> Dict:
        """Deep analysis of individual document."""
        return {
            "index": idx,
            "source": doc.get("source", "unknown"),
            "similarity": doc.get("similarity_score", 0.0),
            "credibility": self._assess_credibility(doc),
            "completeness": self._assess_completeness(doc),
            "type": self._infer_source_type(doc),
            "evidence_strength": "HIGH" if doc.get("similarity_score", 0) > 0.85 else ("MEDIUM" if doc.get("similarity_score", 0) > 0.65 else "LOW")
        }
    
    def _assess_credibility(self, doc: Dict) -> float:
        """Score source credibility 0.0-1.0."""
        source = doc.get("source_type", "").lower()
        content_length = len(doc.get("content", ""))
        
        # Base credibility from source type
        credibility = 0.5  # Default neutral
        
        for cred_source, score in self.CREDIBLE_SOURCES.items():
            if cred_source in source:
                credibility = max(credibility, score)
        
        for quest_source, score in self.QUESTIONABLE_SOURCES.items():
            if quest_source in source:
                credibility = min(credibility, score)
        
        # Adjust based on content quality
        if content_length > 500:
            credibility = min(1.0, credibility + 0.1)
        elif content_length < 50:
            credibility = max(0.0, credibility - 0.2)
        
        return round(credibility, 2)
    
    def _assess_completeness(self, doc: Dict) -> float:
        """Score document completeness."""
        content = doc.get("content", "")
        length = len(content)
        
        # Heuristic: longer documents tend to be more complete
        if length > 1000:
            return 0.95
        elif length > 500:
            return 0.80
        elif length > 200:
            return 0.65
        elif length > 50:
            return 0.40
        else:
            return 0.10
    
    def _infer_source_type(self, doc: Dict) -> str:
        """Infer source type from metadata."""
        source_type = doc.get("source_type", "unknown").lower()
        if any(x in source_type for x in ["fda", "ema", "nih"]):
            return "OFFICIAL_GUIDELINE"
        elif any(x in source_type for x in ["clinical", "study", "journal"]):
            return "CLINICAL_EVIDENCE"
        elif any(x in source_type for x in ["database", "textbook"]):
            return "MEDICAL_REFERENCE"
        elif any(x in source_type for x in ["blog", "forum", "social"]):
            return "COMMUNITY_SOURCE"
        else:
            return "UNKNOWN"
    
    def _assess_coverage(self, documents: List[Dict]) -> float:
        """Assess topic coverage breadth."""
        sources = set(d.get("source", "") for d in documents)
        return min(1.0, len(sources) / 3.0)  # Normalize by expected diversity
    
    def _determine_quality(self, metrics: Dict) -> str:
        """Intelligent quality determination."""
        avg_sim = metrics["avg_similarity"]
        avg_cred = metrics["avg_credibility"]
        
        quality_score = (avg_sim * 0.6) + (avg_cred * 0.4)
        
        if quality_score > 0.80:
            return "EXCELLENT"
        elif quality_score > 0.65:
            return "GOOD"
        elif quality_score > 0.50:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def _determine_risk(self, metrics: Dict) -> str:
        """Determine retrieval risk level."""
        quality_score = (metrics["avg_similarity"] * 0.6) + (metrics["avg_credibility"] * 0.4)
        
        if quality_score > 0.75:
            return "LOW"
        elif quality_score > 0.55:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _identify_issues(self, documents: List[Dict], similarities: List[float], credibilities: List[float]) -> List[str]:
        """Identify potential retrieval problems."""
        issues = []
        
        # Low similarity issue
        if statistics.mean(similarities) < 0.6:
            issues.append("Low average similarity - retrieved docs may be weakly relevant")
        
        # High variability in similarity
        if len(similarities) > 2 and statistics.stdev(similarities) > 0.3:
            issues.append("High variance in document relevance - inconsistent retrieval quality")
        
        # Low credibility sources
        if statistics.mean(credibilities) < 0.5:
            issues.append("Retrieved sources have questionable credibility")
        
        # Few high-quality docs
        high_quality = len([s for s in similarities if s > 0.8])
        if high_quality == 0:
            issues.append("No high-confidence documents (>0.8 similarity)")
        
        return issues
