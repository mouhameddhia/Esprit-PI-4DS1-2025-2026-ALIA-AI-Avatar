"""
SHAP-Based Generation Explainer - Explains LLM Generation Output
"""

import math
import re
from typing import Any, Dict, List

import numpy as np

try:
    import shap
except Exception:  # pragma: no cover - fallback only if dependency import fails
    shap = None


class SHAPGenerationExplainer:
    """
    Uses SHAP (SHapley Additive exPlanations) to explain LLM generation outputs.
    
    Explains which parts of the input (query + context) contributed most to the generated answer.
    """
    
    def __init__(self):
        """Initialize the official SHAP explainer for generation."""
        self.explainer_type = "shap.KernelExplainer"
        self.model_type = "generation"
    
    def explain_generation(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        generated_answer: str,
        judge_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Explain LLM generation using SHAP value approximation.
        
        Returns which input features (query terms, document excerpts) contributed 
        most to the generated answer.
        """
        
        # Build input features for the local surrogate model.
        input_features = self._build_input_features(query, retrieved_docs, generated_answer)

        feature_names = list(input_features.keys())
        feature_vector = np.asarray([list(input_features.values())], dtype=float)
        
        # Calculate SHAP values using the official library, with a deterministic fallback.
        shap_values = self._calculate_shap_values(
            feature_names,
            feature_vector,
            generated_answer,
            judge_scores
        )
        
        # Identify top contributing features
        top_contributors = self._identify_top_contributors(shap_values, input_features)

        base_value = getattr(self, "_last_expected_value", self._calculate_base_value(judge_scores))
        prediction_score = self._predict_generation_support(feature_vector, feature_names, judge_scores)
        
        return {
            "explainer": self.explainer_type,
            "explanation_type": "generation_output",
            "base_value": base_value,
            "shap_values": shap_values,
            "top_contributors": top_contributors,
            "feature_importance": self._rank_feature_importance(shap_values),
            "model_confidence": judge_scores.get("overall_score", 0.0),
            "predicted_support": prediction_score,
            "insight": self._generate_insight(top_contributors, generated_answer)
        }
    
    def _build_input_features(self, query: str, docs: List[Dict], answer: str) -> Dict[str, float]:
        """Extract input features from query and documents."""
        features = {}

        answer_tokens = set(self._tokenize(answer))
        context_tokens = set()
        if docs:
            context_tokens = set(self._tokenize(" ".join(doc.get("content", "") for doc in docs)))
        
        # Query terms as features
        query_terms = self._tokenize(query)
        for term in query_terms:
            if len(term) > 3:  # Skip small words
                features[f"query:{term}"] = 1.0 if term in answer_tokens else 0.0
        
        # Document snippets as features
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            source = doc.get("source", "")
            similarity = doc.get("similarity_score", 0.0)

            doc_tokens = set(self._tokenize(content))
            overlap = len(doc_tokens & context_tokens) / max(1, len(context_tokens)) if context_tokens else 0.0
            features[f"doc:{i}:{source}"] = round(min(1.0, 0.7 * float(similarity) + 0.3 * overlap), 4)

        if docs:
            features["context_alignment"] = round(
                len(context_tokens & answer_tokens) / max(1, len(answer_tokens)),
                4,
            ) if answer_tokens else 0.0
        else:
            features["context_alignment"] = 0.0

        features["answer_length"] = min(1.0, len(answer_tokens) / 120.0) if answer_tokens else 0.0
        
        return features
    
    def _calculate_shap_values(
        self,
        feature_names: List[str],
        feature_vector: np.ndarray,
        answer: str,
        judge_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate SHAP values for each feature using the official SHAP library.

        The explanation is attached to a local surrogate model that predicts answer support
        from query and retrieved-document features.
        """
        if shap is None:
            return self._calculate_shap_values_fallback(feature_names, feature_vector, answer, judge_scores)

        try:
            predict_fn = self._build_prediction_function(feature_names, judge_scores)
            background = self._build_background_matrix(feature_vector)
            explainer = shap.KernelExplainer(predict_fn, background, link="identity")
            self._last_expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

            shap_result = explainer.shap_values(feature_vector, nsamples=min(200, max(50, len(feature_names) * 20)))
            shap_array = self._normalize_shap_output(shap_result)

            return {
                feature_names[idx]: float(shap_array[idx])
                for idx in range(len(feature_names))
            }
        except Exception:
            self._last_expected_value = self._calculate_base_value(judge_scores)
            return self._calculate_shap_values_fallback(feature_names, feature_vector, answer, judge_scores)

    def _calculate_shap_values_fallback(
        self,
        feature_names: List[str],
        feature_vector: np.ndarray,
        answer: str,
        judge_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Fallback attribution logic if SHAP cannot be executed."""

        shap_values = {}
        base_score = judge_scores.get("overall_score", 0.0)
        context_weight = judge_scores.get("context_utilization", 0.8)

        for idx, feature in enumerate(feature_names):
            feature_score = float(feature_vector[0][idx])
            contribution = feature_score * base_score

            if feature.startswith("query:"):
                contribution *= 0.6
            elif feature.startswith("doc:"):
                contribution *= context_weight

            shap_values[feature] = contribution

        return shap_values

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into normalized terms."""

        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]

    def _build_background_matrix(self, feature_vector: np.ndarray) -> np.ndarray:
        """Create a small background matrix for KernelExplainer."""

        zeros = np.zeros_like(feature_vector)
        return np.vstack([zeros, feature_vector * 0.5])

    def _build_prediction_function(self, feature_names: List[str], judge_scores: Dict[str, float]):
        """Build the local surrogate that SHAP explains."""

        query_names = [name for name in feature_names if name.startswith("query:")]
        doc_names = [name for name in feature_names if name.startswith("doc:")]
        context_index = feature_names.index("context_alignment") if "context_alignment" in feature_names else None
        length_index = feature_names.index("answer_length") if "answer_length" in feature_names else None

        query_weight = 0.35 / max(1, len(query_names))
        doc_weight = 0.45 / max(1, len(doc_names))
        bias = -0.2 + 0.35 * judge_scores.get("overall_score", 0.0) + 0.25 * judge_scores.get("context_utilization", 0.0)

        def predict(matrix: np.ndarray) -> np.ndarray:
            outputs = []
            matrix = np.atleast_2d(matrix)
            for row in matrix:
                raw_score = bias

                for idx, feature_name in enumerate(feature_names):
                    feature_value = float(row[idx])
                    if feature_name.startswith("query:"):
                        raw_score += query_weight * feature_value
                    elif feature_name.startswith("doc:"):
                        raw_score += doc_weight * feature_value

                if context_index is not None:
                    raw_score += 0.15 * float(row[context_index])
                if length_index is not None:
                    raw_score += 0.05 * float(row[length_index])

                outputs.append(1.0 / (1.0 + math.exp(-4.0 * (raw_score - 0.5))))

            return np.asarray(outputs, dtype=float)

        return predict

    def _normalize_shap_output(self, shap_result: Any) -> np.ndarray:
        """Convert SHAP output to a flat numeric vector."""

        if isinstance(shap_result, list):
            shap_result = shap_result[0]

        shap_array = np.asarray(shap_result, dtype=float)
        if shap_array.ndim == 2:
            shap_array = shap_array[0]
        return shap_array
    
    def _identify_top_contributors(self, shap_values: Dict, features: Dict) -> List[Dict]:
        """Identify top contributing features (SHAP explanations)."""
        
        ranked = sorted(
            [
                {
                    "feature": feat,
                    "shap_value": shap_values.get(feat, 0.0),
                    "feature_type": feat.split(":")[0],
                    "feature_value": features.get(feat, 0.0),
                    "impact": "HIGH" if abs(shap_values.get(feat, 0.0)) > 0.15 else "MEDIUM" if abs(shap_values.get(feat, 0.0)) > 0.05 else "LOW"
                }
                for feat in shap_values.keys()
            ],
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )
        
        return ranked[:5]  # Top 5 contributors
    
    def _rank_feature_importance(self, shap_values: Dict) -> Dict[str, float]:
        """Rank features by absolute SHAP values (feature importance)."""
        
        total_impact = sum(abs(v) for v in shap_values.values())
        
        if total_impact == 0:
            total_impact = 1.0
        
        importance = {}
        for feature, value in shap_values.items():
            importance[feature] = abs(value) / total_impact
        
        # Return top 5 by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def _calculate_base_value(self, judge_scores: Dict) -> float:
        """Calculate a baseline value for the explanation."""

        return judge_scores.get("overall_score", 0.5) * 0.5

    def _predict_generation_support(self, feature_vector: np.ndarray, feature_names: List[str], judge_scores: Dict[str, float]) -> float:
        """Return the surrogate model's support score for the current instance."""

        predict_fn = self._build_prediction_function(feature_names, judge_scores)
        return float(predict_fn(feature_vector)[0])
    
    def _generate_insight(self, top_contributors: List[Dict], answer: str) -> str:
        """Generate human-readable insight from SHAP explanation."""
        
        if not top_contributors:
            return "Unable to identify key contributing factors to generation."
        
        top_feature = top_contributors[0]
        feature_name = top_feature["feature"]
        
        if feature_name.startswith("query:"):
            return f"Generation heavily influenced by query term '{feature_name.split(':')[1]}' with SHAP value {top_feature['shap_value']:.3f}"
        elif feature_name.startswith("doc:"):
            doc_id = feature_name.split(":")[1]
            return f"Retrieved document #{doc_id} is the strongest contributor to answer generation with SHAP value {top_feature['shap_value']:.3f}"
        
        return f"Primary contributor: {feature_name} (SHAP value: {top_feature['shap_value']:.3f})"


def explain_generation_with_shap(
    query: str,
    retrieved_docs: List[Dict],
    answer: str,
    judge_scores: Dict
) -> Dict[str, Any]:
    """Convenience function for SHAP generation explanation."""
    explainer = SHAPGenerationExplainer()
    return explainer.explain_generation(query, retrieved_docs, answer, judge_scores)
