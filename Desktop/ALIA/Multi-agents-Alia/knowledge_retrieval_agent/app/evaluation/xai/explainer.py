"""Deterministic XAI report generator for hybrid RAG + judge decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
except Exception:  # pragma: no cover - optional dependency
    LimeTextExplainer = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


@dataclass
class XAIInputs:
    question: str
    answer: str
    context: str
    judge_scores: dict[str, float] | None = None
    crag_decision: str | None = None


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _split_context_chunks(context: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", context) if block.strip()]
    return blocks if blocks else ([context.strip()] if context.strip() else [])


def _split_claims(answer: str) -> list[str]:
    text = _normalize_text(answer)
    if not text:
        return []
    parts = [segment.strip(" .") for segment in re.split(r"[.;]\s+|\n+", text) if segment.strip()]
    return parts if parts else [text]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}


def _find_best_support(claim: str, chunks: list[str]) -> tuple[int | None, float]:
    claim_tokens = _tokenize(claim)
    if not claim_tokens or not chunks:
        return None, 0.0

    best_index = None
    best_overlap = 0.0
    for idx, chunk in enumerate(chunks, start=1):
        chunk_tokens = _tokenize(chunk)
        if not chunk_tokens:
            continue
        overlap = len(claim_tokens.intersection(chunk_tokens)) / max(1, len(claim_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = idx

    return best_index, best_overlap


def _support_probability(text: str, chunks: list[str]) -> float:
    tokens = _tokenize(text)
    if not tokens or not chunks:
        return 0.0
    best = 0.0
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk)
        if not chunk_tokens:
            continue
        overlap = len(tokens.intersection(chunk_tokens)) / max(1, len(tokens))
        if overlap > best:
            best = overlap
    return float(min(1.0, max(0.0, best)))


def _lime_local_attribution(claim: str, chunks: list[str]) -> str:
    if LimeTextExplainer is None:
        return "LIME unavailable (package not installed)."
    if not claim.strip() or not chunks:
        return "LIME unavailable (insufficient claim/context data)."

    explainer = LimeTextExplainer(class_names=["unsupported", "supported"], random_state=42)

    def _predict_proba(texts: list[str]) -> np.ndarray:
        probs = []
        for txt in texts:
            support = _support_probability(txt, chunks)
            probs.append([1.0 - support, support])
        return np.asarray(probs, dtype=float)

    try:
        explanation = explainer.explain_instance(claim, _predict_proba, labels=[1], num_features=5)
        items = explanation.as_list(label=1)
        positive = [f"{token} ({weight:.3f})" for token, weight in items if weight > 0][:3]
        if not positive:
            return "LIME ran but found no strong positive local tokens."
        return "Top local tokens: " + ", ".join(positive)
    except Exception as exc:  # pragma: no cover - robustness fallback
        return f"LIME failed: {exc}"


def _claim_feature_vector(claim: str, chunk: str, overlap: float) -> np.ndarray:
    claim_lower = claim.lower()
    chunk_lower = chunk.lower()
    number_overlap = len(set(re.findall(r"\d+", claim_lower)).intersection(set(re.findall(r"\d+", chunk_lower))))
    dosage_terms = ("dosage", "dose", "ml", "mg", "capsule", "tablet")
    warning_terms = ("warning", "contraindication", "pregnancy", "risk", "adverse")

    dosage_match = 1.0 if any(term in claim_lower and term in chunk_lower for term in dosage_terms) else 0.0
    warning_match = 1.0 if any(term in claim_lower and term in chunk_lower for term in warning_terms) else 0.0
    numeric_match = 1.0 if number_overlap > 0 else 0.0

    return np.asarray([float(overlap), numeric_match, dosage_match, warning_match], dtype=float)


def _shap_global_attribution(claim_support: list[tuple[str, int | None, float]], chunks: list[str]) -> str:
    if shap is None:
        return "SHAP unavailable (package not installed)."
    if not claim_support:
        return "SHAP unavailable (no claims to explain)."

    rows = []
    for claim, chunk_idx, overlap in claim_support:
        chunk = chunks[chunk_idx - 1] if chunk_idx is not None and 0 < chunk_idx <= len(chunks) else ""
        rows.append(_claim_feature_vector(claim, chunk, overlap))
    x = np.vstack(rows)

    feature_names = ["overlap", "numeric_match", "dosage_match", "warning_match"]
    weights = np.asarray([0.55, 0.15, 0.2, 0.1], dtype=float)

    def _model(arr: np.ndarray) -> np.ndarray:
        return np.dot(arr, weights)

    try:
        background = np.zeros((1, x.shape[1]), dtype=float)
        explainer = shap.Explainer(_model, background)
        shap_values = explainer(x)
        values = np.asarray(shap_values.values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        mean_abs = np.mean(np.abs(values), axis=0)
        ranked = sorted(zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True)
        top = [f"{name} ({score:.3f})" for name, score in ranked[:3]]
        return "Top global features: " + ", ".join(top)
    except Exception as exc:  # pragma: no cover - robustness fallback
        return f"SHAP failed: {exc}"


def _format_judge_explanation(judge_scores: dict[str, float] | None) -> tuple[str, str, str]:
    if not judge_scores:
        return (
            "- Score reasoning: Judge scores were not provided.",
            "- Penalties: Not computable without judge metrics.",
            "- Strengths: Not computable without judge metrics.",
        )

    score_parts = []
    penalties = []
    strengths = []

    for key in (
        "overall_score",
        "faithfulness",
        "answer_relevance",
        "context_utilization",
        "medical_safety",
        "clarity",
        "mode_alignment",
    ):
        if key in judge_scores:
            score_parts.append(f"{key}={float(judge_scores[key]):.3f}")

    for key, value in judge_scores.items():
        try:
            score = float(value)
        except Exception:
            continue
        if score < 0.5:
            penalties.append(f"{key} below threshold ({score:.3f})")
        elif score >= 0.8:
            strengths.append(f"{key} strong ({score:.3f})")

    penalties_text = "; ".join(penalties) if penalties else "No major penalties detected from provided scores."
    strengths_text = "; ".join(strengths) if strengths else "No high-confidence strengths detected from provided scores."

    return (
        f"- Score reasoning: {'; '.join(score_parts) if score_parts else 'No valid numeric scores supplied.'}",
        f"- Penalties: {penalties_text}",
        f"- Strengths: {strengths_text}",
    )


def _format_crag_explanation(crag_decision: str | None) -> tuple[str, str]:
    decision = (crag_decision or "UNKNOWN").strip().upper()
    if decision == "CORRECT":
        reason = "High retrieval confidence and evidence consistency indicate no corrective fallback required."
    elif decision == "AMBIGUOUS":
        reason = "Retrieval confidence is moderate; multiple plausible evidence paths suggest expansion or clarification."
    elif decision == "INCORRECT":
        reason = "Low confidence and/or conflicting retrieval signals indicate evidence quality failure."
    else:
        reason = "CRAG decision not provided; corrective routing rationale cannot be fully determined."
    return f"- Decision: {decision}", f"- Reason: {reason}"


def _format_safety(claim_support: list[tuple[str, int | None, float]], judge_scores: dict[str, float] | None) -> tuple[str, str, str]:
    unsupported_count = sum(1 for _, idx, _ in claim_support if idx is None)
    if unsupported_count > 0:
        hallucination = f"Elevated ({unsupported_count} unsupported claim(s))."
    else:
        hallucination = "Low from lexical support check (all claims mapped to at least one context chunk)."

    medical_safety_score = None
    if judge_scores and "medical_safety" in judge_scores:
        try:
            medical_safety_score = float(judge_scores["medical_safety"])
        except Exception:
            medical_safety_score = None

    if medical_safety_score is None:
        risk_level = "Unknown (judge safety score not provided)."
    elif medical_safety_score < 0.5:
        risk_level = f"High (medical_safety={medical_safety_score:.3f})."
    elif medical_safety_score < 0.75:
        risk_level = f"Moderate (medical_safety={medical_safety_score:.3f})."
    else:
        risk_level = f"Low (medical_safety={medical_safety_score:.3f})."

    missing_warning = "Potentially missing if context contains contraindication/warning terms absent in answer."
    return (
        f"- Hallucination risk: {hallucination}",
        f"- Medical risk level: {risk_level}",
        f"- Missing warnings: {missing_warning}",
    )


def build_xai_report(
    *,
    question: str,
    answer: str,
    context: str,
    judge_scores: dict[str, float] | None = None,
    crag_decision: str | None = None,
) -> str:
    """Return strict formatted XAI report for RAG + Judge outcomes."""

    data = XAIInputs(
        question=_normalize_text(question),
        answer=_normalize_text(answer),
        context=str(context or "").strip(),
        judge_scores=judge_scores,
        crag_decision=crag_decision,
    )

    chunks = _split_context_chunks(data.context)
    claims = _split_claims(data.answer)

    claim_support: list[tuple[str, int | None, float]] = []
    evidence_lines: list[str] = []
    for index, claim in enumerate(claims, start=1):
        chunk_idx, overlap = _find_best_support(claim, chunks)
        if chunk_idx is None or overlap < 0.2:
            claim_support.append((claim, None, overlap))
            evidence_lines.append(f"- Claim {index}: {claim} -> NO EVIDENCE FOUND")
        else:
            claim_support.append((claim, chunk_idx, overlap))
            evidence_lines.append(f"- Claim {index}: {claim} -> Context Chunk {chunk_idx}")

    if not evidence_lines:
        evidence_lines.append("- Missing evidence -> NO EVIDENCE FOUND")

    lime_summary = _lime_local_attribution(claims[0], chunks) if claims else "LIME unavailable (no claims found)."
    shap_summary = _shap_global_attribution(claim_support, chunks)

    judge_reason, judge_penalties, judge_strengths = _format_judge_explanation(data.judge_scores)
    crag_decision_line, crag_reason_line = _format_crag_explanation(data.crag_decision)
    safety_hallu, safety_risk, safety_missing = _format_safety(claim_support, data.judge_scores)

    unsupported_count = sum(1 for _, idx, _ in claim_support if idx is None)
    final_sentence = (
        f"- The system answer maps {len(claims) - unsupported_count}/{max(1, len(claims))} claims to retrieved context, "
        f"with CRAG={((data.crag_decision or 'UNKNOWN').strip().upper())} and safety requiring review if unsupported claims exist."
    )

    sections = [
        "=== XAI REPORT ===",
        "",
        "1. ANSWER REASONING",
        "- Explanation of how answer was formed: The answer was generated from retrieved context chunks after retrieval and ranking; lexical claim-to-context matching was used to validate traceability.",
        f"- Local attribution (LIME): {lime_summary}",
        f"- Global attribution (SHAP): {shap_summary}",
        "",
        "2. EVIDENCE MAPPING",
        *evidence_lines,
        "- Missing evidence -> NONE" if unsupported_count == 0 else "- Missing evidence -> SPECIFY (see claims marked NO EVIDENCE FOUND)",
        "",
        "3. JUDGE EXPLANATION",
        judge_reason,
        judge_penalties,
        judge_strengths,
        "",
        "4. CRAG EXPLANATION",
        crag_decision_line,
        crag_reason_line,
        "",
        "5. SAFETY ANALYSIS",
        safety_hallu,
        safety_risk,
        safety_missing,
        "",
        "6. FINAL INTERPRETATION",
        final_sentence,
    ]

    return "\n".join(sections)
