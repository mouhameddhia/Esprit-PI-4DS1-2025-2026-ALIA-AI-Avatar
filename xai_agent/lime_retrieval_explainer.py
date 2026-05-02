"""
LIME-Based Retrieval Explainer - Explains RAG Retrieval Decisions
"""

import math
import re
from typing import Any, Callable, Dict, List

import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
except Exception:  # pragma: no cover - fallback only if dependency import fails
    LimeTextExplainer = None


class LIMERetrievalExplainer:
    """
    Uses LIME (Local Interpretable Model-agnostic Explanations) to explain RAG retrieval.
    
    Explains which parts of the query matched which parts of retrieved documents
    and why certain documents were retrieved/ranked higher.
    """
    
    def __init__(self):
        """Initialize the official LIME explainer for retrieval."""
        self.explainer_type = "lime.lime_text.LimeTextExplainer"
        self.model_type = "retrieval"
        self._lime = (
            LimeTextExplainer(class_names=["irrelevant", "relevant"], random_state=42)
            if LimeTextExplainer is not None
            else None
        )
    
    def explain_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Explain RAG retrieval decisions using LIME approximation.
        
        Returns interpretable explanations for why each document was retrieved
        and how it relates to the query.
        """
        
        if not retrieved_docs:
            return {
                "explainer": self.explainer_type,
                "explanation_type": "retrieval_decisions",
                "query": query,
                "query_features": self._extract_query_features(query),
                "document_explanations": [],
                "retrieval_summary": {"summary": "No documents retrieved", "avg_relevance": 0.0},
                "ranking_rationale": "No documents retrieved"
            }

        # Extract query features used by the local relevance surrogate.
        query_features = self._extract_query_features(query)
        
        # For each document, explain why it was retrieved
        doc_explanations = []
        for doc in retrieved_docs:
            explanation = self._explain_single_document(query, query_features, doc)
            doc_explanations.append(explanation)
        
        # Aggregate explanations
        return {
            "explainer": self.explainer_type,
            "explanation_type": "retrieval_decisions",
            "query": query,
            "query_features": query_features,
            "document_explanations": doc_explanations,
            "retrieval_summary": self._generate_retrieval_summary(doc_explanations),
            "ranking_rationale": self._explain_ranking(retrieved_docs, query_features)
        }
    
    def _extract_query_features(self, query: str) -> Dict[str, List[str]]:
        """Extract interpretable features from query."""
        
        words = self._tokenize(query)
        
        # Categorize by length
        short_terms = [w for w in words if len(w) <= 5]
        long_terms = [w for w in words if len(w) > 5]
        
        # Medical keywords
        medical_keywords = self._identify_medical_keywords(query)
        
        return {
            "all_terms": words,
            "short_terms": short_terms,
            "long_terms": long_terms,
            "medical_keywords": medical_keywords,
            "query_length": len(words)
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into normalized terms for local relevance scoring."""

        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]

    def _source_bonus(self, doc: Dict[str, Any]) -> float:
        """Assign a small source prior for authoritative medical documents."""

        source_text = f"{doc.get('source', '')} {doc.get('source_type', '')}".lower()

        if any(token in source_text for token in ["fda", "ema", "nih", "guideline", "official"]):
            return 0.35
        if any(token in source_text for token in ["clinical", "journal", "study", "database"]):
            return 0.2
        if any(token in source_text for token in ["blog", "forum", "social"]):
            return -0.2

        return 0.0

    def _score_relevance(self, query_features: Dict[str, List[str]], text: str, doc: Dict[str, Any]) -> float:
        """Score relevance for a perturbed document text."""

        doc_terms = set(self._tokenize(text))
        query_terms = set(query_features.get("all_terms", []))
        medical_terms = set(query_features.get("medical_keywords", []))

        term_overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
        medical_overlap = len(medical_terms & doc_terms) / max(1, len(medical_terms)) if medical_terms else 0.0
        similarity_prior = float(doc.get("similarity_score", 0.0))
        source_bonus = self._source_bonus(doc)
        length_factor = min(1.0, len(doc_terms) / 120.0)

        raw_score = (
            -0.9
            + 2.1 * term_overlap
            + 1.25 * medical_overlap
            + 1.0 * similarity_prior
            + 0.45 * source_bonus
            + 0.15 * length_factor
        )

        return float(1.0 / (1.0 + math.exp(-raw_score)))

    def _build_prediction_function(
        self,
        query_features: Dict[str, List[str]],
        doc: Dict[str, Any]
    ) -> Callable[[List[str]], np.ndarray]:
        """Build the class-probability function used by LIME."""

        def predict(texts: List[str]) -> np.ndarray:
            probabilities = []
            for text in texts:
                relevance = self._score_relevance(query_features, text, doc)
                probabilities.append([1.0 - relevance, relevance])
            return np.asarray(probabilities, dtype=float)

        return predict

    def _generate_lime_explanation(self, doc_content: str, prediction_fn: Callable[[List[str]], np.ndarray]):
        """Create a LIME explanation for a single document."""

        if self._lime is None:
            raise RuntimeError("lime is not available in the current environment")

        num_tokens = max(3, len(self._tokenize(doc_content)))
        num_features = min(10, num_tokens)
        return self._lime.explain_instance(
            doc_content,
            prediction_fn,
            labels=(1,),
            num_features=num_features,
            num_samples=2000,
        )

    def _format_lime_features(self, explanation) -> List[Dict[str, Any]]:
        """Convert LIME output into JSON-friendly feature records."""

        return [
            {
                "feature": feature,
                "weight": float(weight),
                "direction": "positive" if weight >= 0 else "negative"
            }
            for feature, weight in explanation.as_list(label=1)
        ]
    
    def _identify_medical_keywords(self, text: str) -> List[str]:
        """Identify medical keywords in text."""
        
        medical_terms = [
            "dosage", "dose", "treatment", "medication", "drug",
            "symptom", "side effect", "contraindication", "indication",
            "patient", "therapy", "clinical", "disease", "condition",
            "aspirin", "ibuprofen", "acetaminophen", "mg", "tablet",
            "adverse", "reaction", "interaction", "pregnancy"
        ]
        
        text_lower = text.lower()
        found = [term for term in medical_terms if term in text_lower]
        return found
    
    def _explain_single_document(
        self,
        query: str,
        query_features: Dict,
        doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate LIME-style explanation for single document."""
        
        doc_content = doc.get("content", "")
        similarity_score = doc.get("similarity_score", 0.0)
        source = doc.get("source", "Unknown")
        
        # Find matching features in document
        matching_terms = self._find_matching_terms(query_features, doc_content)

        prediction_fn = self._build_prediction_function(query_features, doc)
        predicted_relevance = self._score_relevance(query_features, doc_content, doc)

        lime_features: List[Dict[str, Any]] = []
        lime_error = None
        if doc_content.strip():
            try:
                explanation = self._generate_lime_explanation(doc_content, prediction_fn)
                lime_features = self._format_lime_features(explanation)
                predicted_relevance = float(explanation.predict_proba[0][1])
            except Exception as exc:
                lime_error = str(exc)
        
        # Calculate local importance weights
        feature_weights = self._calculate_feature_weights(
            lime_features,
            matching_terms,
            similarity_score,
            doc_content,
            predicted_relevance
        )
        
        # Extract supporting evidence (snippets)
        supporting_snippets = self._extract_supporting_snippets(
            doc_content,
            matching_terms
        )
        
        return {
            "source": source,
            "similarity_score": similarity_score,
            "predicted_relevance": predicted_relevance,
            "matching_query_terms": matching_terms,
            "feature_weights": feature_weights,
            "lime_features": lime_features,
            "lime_status": "generated" if lime_features else "fallback",
            "lime_error": lime_error,
            "supporting_evidence": supporting_snippets,
            "retrieval_reason": self._generate_retrieval_reason(
                matching_terms,
                feature_weights,
                source,
                lime_features,
                predicted_relevance
            ),
            "relevance_explanation": self._explain_relevance(
                similarity_score,
                matching_terms,
                doc_content,
                predicted_relevance
            )
        }
    
    def _find_matching_terms(self, query_features: Dict, doc: str) -> Dict[str, List[str]]:
        """Find which query terms appear in document."""
        
        doc_lower = doc.lower()
        
        matching = {
            "matching_words": [],
            "matching_medical_keywords": [],
            "unmatched_query_terms": [],
            "all_query_terms": list(query_features.get("all_terms", []))
        }
        
        # Check word matches
        for word in query_features["all_terms"]:
            if word in doc_lower:
                matching["matching_words"].append(word)
            else:
                matching["unmatched_query_terms"].append(word)
        
        # Check medical keyword matches
        for keyword in query_features["medical_keywords"]:
            if keyword in doc_lower:
                matching["matching_medical_keywords"].append(keyword)
        
        return matching
    
    def _calculate_feature_weights(
        self,
        lime_features: List[Dict[str, Any]],
        matching_terms: Dict,
        similarity: float,
        doc_content: str,
        predicted_relevance: float
    ) -> Dict[str, float]:
        """Summarize the LIME explanation as aggregated weights."""
        
        weights = {}

        positive_weight = sum(item["weight"] for item in lime_features if item["weight"] > 0)
        negative_weight = sum(abs(item["weight"]) for item in lime_features if item["weight"] < 0)
        
        num_matching = len(matching_terms["matching_words"])
        num_medical = len(matching_terms["matching_medical_keywords"])
        
        weights["lime_positive"] = round(positive_weight, 4)
        weights["lime_negative"] = round(negative_weight, 4)
        weights["predicted_relevance"] = round(predicted_relevance, 4)
        weights["document_similarity"] = round(similarity, 4)
        weights["word_matches"] = round(num_matching / max(1, len(doc_content.split())), 4) if num_matching > 0 else 0.0
        weights["medical_keyword_matches"] = float(num_medical)
        weights["matching_coverage"] = round(
            len(matching_terms["matching_words"]) / max(1, len(matching_terms["all_query_terms"])),
            4,
        ) if "all_query_terms" in matching_terms else 0.0
        
        weights["overall_relevance"] = round(max(similarity, predicted_relevance), 4)
        
        # Content length consideration
        doc_length = len(doc_content.split())
        weights["document_comprehensiveness"] = min(1.0, doc_length / 200.0)
        
        return weights
    
    def _extract_supporting_snippets(
        self,
        doc_content: str,
        matching_terms: Dict
    ) -> List[str]:
        """Extract document snippets supporting the retrieval decision."""
        
        snippets = []
        doc_lower = doc_content.lower()
        
        # Find snippets containing matching terms
        sentences = doc_content.split(".")
        
        for term in matching_terms["matching_words"][:3]:  # Top 3 terms
            for sentence in sentences:
                if term in sentence.lower():
                    snippet = sentence.strip()
                    if len(snippet) > 20:  # Only meaningful snippets
                        snippets.append(f"'{snippet}'")
                    break
        
        return snippets[:3]  # Top 3 snippets
    
    def _generate_retrieval_reason(
        self,
        matching_terms: Dict,
        weights: Dict,
        source: str,
        lime_features: List[Dict[str, Any]],
        predicted_relevance: float
    ) -> str:
        """Generate human-readable reason for retrieval."""
        
        reasons = []
        
        if matching_terms["matching_medical_keywords"]:
            keywords = ", ".join(matching_terms["matching_medical_keywords"][:2])
            reasons.append(f"Contains medical keywords: {keywords}")
        
        if matching_terms["matching_words"]:
            num_matches = len(matching_terms["matching_words"])
            reasons.append(f"Matches {num_matches} query terms")
        
        if weights.get("overall_relevance", 0) > 0.8:
            reasons.append("High local relevance under the LIME surrogate")

        if lime_features:
            top_feature = lime_features[0]
            reasons.append(f"Top LIME token: {top_feature['feature']} ({top_feature['weight']:+.3f})")
        
        if source and any(x in source.lower() for x in ["fda", "nih", "clinical"]):
            reasons.append(f"From authoritative source: {source}")

        reasons.append(f"Predicted local relevance: {predicted_relevance:.3f}")
        
        return " | ".join(reasons) if reasons else "Matched query intent"
    
    def _explain_relevance(self, similarity: float, matching_terms: Dict, doc: str, predicted_relevance: float) -> str:
        """Explain relevance level."""
        
        match_count = len(matching_terms["matching_words"]) + len(matching_terms["matching_medical_keywords"])
        
        if predicted_relevance > 0.9:
            return f"VERY HIGH - LIME relevance {predicted_relevance:.3f} with strong token support"
        if predicted_relevance > 0.75:
            return f"HIGH - LIME relevance {predicted_relevance:.3f} with matching content"
        if predicted_relevance > 0.6:
            return f"MEDIUM - LIME relevance {predicted_relevance:.3f} with some matching terms"
        return f"LOW - LIME relevance {predicted_relevance:.3f}, may need context"
    
    def _generate_retrieval_summary(self, explanations: List[Dict]) -> Dict[str, Any]:
        """Generate summary of retrieval decisions."""
        
        if not explanations:
            return {"summary": "No documents retrieved", "avg_relevance": 0.0}
        
        avg_similarity = sum(e["similarity_score"] for e in explanations) / len(explanations)
        avg_relevance = sum(e.get("predicted_relevance", 0.0) for e in explanations) / len(explanations)
        top_doc = max(explanations, key=lambda item: item.get("predicted_relevance", 0.0))
        
        return {
            "total_documents": len(explanations),
            "average_similarity": avg_similarity,
            "average_lime_relevance": round(avg_relevance, 4),
            "best_match_source": top_doc.get("source", "N/A"),
            "retrieval_quality": "EXCELLENT" if avg_relevance > 0.85 else "GOOD" if avg_relevance > 0.7 else "FAIR"
        }
    
    def _explain_ranking(self, docs: List[Dict], query_features: Dict) -> str:
        """Explain ranking order of documents."""
        
        if len(docs) <= 1:
            return "Only one document retrieved"
        
        ranked_docs = sorted(docs, key=lambda item: item.get("predicted_relevance", 0.0), reverse=True)
        top_doc = ranked_docs[0]
        second_doc = ranked_docs[1] if len(ranked_docs) > 1 else None
        
        explanation = (
            f"Top document '{top_doc.get('source', 'Unknown')}' ranked first due to the highest local LIME relevance "
            f"({top_doc.get('predicted_relevance', 0):.2f}) and similarity score ({top_doc.get('similarity_score', 0):.2f}). "
        )
        
        if second_doc:
            explanation += (
                f"Second document '{second_doc.get('source', 'Unknown')}' with LIME relevance "
                f"{second_doc.get('predicted_relevance', 0):.2f} and similarity {second_doc.get('similarity_score', 0):.2f}."
            )
        
        return explanation


def explain_retrieval_with_lime(
    query: str,
    retrieved_docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convenience function for LIME retrieval explanation."""
    explainer = LIMERetrievalExplainer()
    return explainer.explain_retrieval(query, retrieved_docs)
