"""4-Level Medical Rep Competency Framework for ALIA Training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CompetencyLevel(Enum):
    """Competency levels for medical rep training."""

    BEGINNER = 1
    JUNIOR = 2
    CONFIRMED = 3
    EXPERT = 4


def coerce_competency_level(value: Any) -> CompetencyLevel:
    """Coerce a UI/config value into a competency level."""

    if isinstance(value, CompetencyLevel):
        return value

    if isinstance(value, int):
        try:
            return CompetencyLevel(value)
        except ValueError:
            return CompetencyLevel.JUNIOR

    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned.isdigit():
            try:
                return CompetencyLevel(int(cleaned))
            except ValueError:
                return CompetencyLevel.JUNIOR
        try:
            return CompetencyLevel[cleaned]
        except KeyError:
            return CompetencyLevel.JUNIOR

    return CompetencyLevel.JUNIOR


@dataclass
class CompetencyProfile:
    """Profile defining rules for each competency level."""

    level: CompetencyLevel
    name: str
    profile_description: str
    # Visit structure
    visit_structure: str
    # Questioning
    min_questions: int
    max_questions: int
    question_depth: str  # simple, intermediate, complex
    # Objection handling
    objections_to_handle: int
    objection_types: list[str]
    # Argumentation
    argument_format: str
    args_per_product: int
    # Knowledge scope
    product_portfolio_size: str
    product_knowledge_depth: str
    proof_count: int
    proof_handling: str
    # Limits
    limitations: list[str]
    # KPI targets
    kpi_score: str
    objection_success_rate: str
    engagement_rate: str
    completion_rate: str


# Define the 4 levels
COMPETENCY_LEVELS = {
    CompetencyLevel.BEGINNER: CompetencyProfile(
        level=CompetencyLevel.BEGINNER,
        name="DÉBUTANT",
        profile_description="ALIA peut tenir une visite courte, polite et structurée, mais reste 'scriptée'. Elle applique le process sans grande adaptation.",
        visit_structure="6 étapes strictes: Introduction → Sondage → Synthèse → Objections → Argumentation → Conclusion",
        min_questions=1,
        max_questions=2,
        question_depth="simple",
        objections_to_handle=1,
        objection_types=["standard_prepared_responses"],
        argument_format="basic_prepared",
        args_per_product=2,
        product_portfolio_size="basique",
        product_knowledge_depth="noms, indications, 2 bénéfices clés, posologie simple, précautions principales",
        proof_count=1,
        proof_handling="phrase courte, sans chiffres complexes",
        limitations=[
            "Ne gère pas les objections complexes (prix, études, interactions)",
            "Ne segmente pas par SONCAS",
            "Ne répond pas à une question scientifique pointue",
        ],
        kpi_score="≥ 90%",
        objection_success_rate="N/A",
        engagement_rate="≥ 50%",
        completion_rate="structure respectée ≥ 90%",
    ),
    CompetencyLevel.JUNIOR: CompetencyProfile(
        level=CompetencyLevel.JUNIOR,
        name="JUNIOR",
        profile_description="ALIA devient plus interactive: elle questionne mieux, écoute, et adapte légèrement le discours selon le médecin.",
        visit_structure="3 formats: Flash / Standard / Approfondie",
        min_questions=2,
        max_questions=4,
        question_depth="intermediate",
        objections_to_handle=2,
        objection_types=["habitudes", "pas le temps", "pas convaincu", "tolérance"],
        argument_format="besoin → avantage → preuve → usage",
        args_per_product=3,
        product_portfolio_size="portefeuille restreint (5-10 produits)",
        product_knowledge_depth="fiche complète, composition/mécanisme simplifié, profils patients, conseils pratiques",
        proof_count=2,
        proof_handling="1-2 preuves citées, sans bibliographie complète",
        limitations=[
            "Ne rentre pas dans discussions scientifiques avancées",
            "Ne négocie pas le prix (renvoie aux canaux appropriés)",
            "Si demande guideline détaillée: propose d'envoyer",
        ],
        kpi_score="≥ 7/10",
        objection_success_rate="≥ 70% avec validation A-C-R-V",
        engagement_rate="≥ 60%",
        completion_rate="CRM complet ≥ 80%",
    ),
    CompetencyLevel.CONFIRMED: CompetencyProfile(
        level=CompetencyLevel.CONFIRMED,
        name="CONFIRMÉ",
        profile_description="ALIA simule un VM autonome: elle personnalise fortement, gère la pression, et conduit des visites 'vraie vie' avec interruptions et objections variées.",
        visit_structure="Adaptation fluide au profil médecin + découverte structurée",
        min_questions=3,
        max_questions=6,
        question_depth="complex",
        objections_to_handle=3,
        objection_types=["malentendu", "préjugé", "objection de valeur", "objection test"],
        argument_format="segmenté par profils patients",
        args_per_product=4,
        product_portfolio_size="portefeuille large (15-30 références)",
        product_knowledge_depth="mécanisme/composition précis, tolérance/contre-indications, messages par spécialité",
        proof_count=3,
        proof_handling="lecture et synthèse 2-3 études par produit, capable résumé 20 sec",
        limitations=[
            "Ne sort jamais du cadre réglementaire",
            "Ne promet pas de résultat patient",
            "Si hors périmètre: 'je note et je reviens avec réponse confirmée'",
        ],
        kpi_score="≥ 8/10",
        objection_success_rate="≥ 80%",
        engagement_rate="≥ 70%",
        completion_rate="Zéro 'mots tueurs', zéro surpromesse",
    ),
    CompetencyLevel.EXPERT: CompetencyProfile(
        level=CompetencyLevel.EXPERT,
        name="EXPERT",
        profile_description="ALIA devient 'Top performer': elle transforme une visite en valeur clinique + relation durable, et agit comme coach et référence tout en restant humble et conforme.",
        visit_structure="Diagnostic relationnel instantané + conduite stratégique",
        min_questions=4,
        max_questions=8,
        question_depth="complex_strategic",
        objections_to_handle=4,
        objection_types=["hostile", "multiple_objections", "conflit_croyances", "manque_dispo"],
        argument_format="haute précision + bénéfice patient",
        args_per_product=5,
        product_portfolio_size="portefeuille complet + inter-gammes",
        product_knowledge_depth="maîtrise inter-gammes, comparer niveaux preuve, résumer méthodologie",
        proof_count=5,
        proof_handling="capacité comparer niveaux preuve, citer référence à partager",
        limitations=[
            "Même expert, n'affirme JAMAIS sans source",
            "Sujet médical complexe = cadre: 'information générale' + renvoi recommandations",
            "Respect TOTAL règles internes (échantillons, cadeaux, claims)",
        ],
        kpi_score="≥ 9/10",
        objection_success_rate="≥ 80%",
        engagement_rate="≥ 70%",
        completion_rate="Qualité suivi (2e cycle) ≥ 80%, feedback coaching ≥ 8/10",
    ),
}


def get_competency_profile(level: CompetencyLevel) -> CompetencyProfile:
    """Get the profile for a given competency level."""
    return COMPETENCY_LEVELS[level]


def get_competency_system_prompt(level: CompetencyLevel) -> str:
    """Generate a system prompt for a given competency level."""
    profile = COMPETENCY_LEVELS[level]

    prompt = f"""You are a senior medical doctor training a pharmaceutical representative at the {profile.name} level.

COMPETENCY LEVEL: {profile.name}
PROFILE: {profile.profile_description}

VISIT STRUCTURE:
{profile.visit_structure}

QUESTIONING REQUIREMENTS:
- Ask between {profile.min_questions}-{profile.max_questions} questions per interaction
- Question depth level: {profile.question_depth}
- Build progressively on previous answers
- Never ask the same question twice

OBJECTION HANDLING:
- Expected to handle up to {profile.objections_to_handle} objections
- Types to expect: {', '.join(profile.objection_types)}
- Use validated objection handling (ACR-V method)

ARGUMENTATION:
- Format: {profile.argument_format}
- Approximately {profile.args_per_product} arguments per product
- Focus on benefits + evidence + usage

KNOWLEDGE SCOPE:
- Product portfolio size: {profile.product_portfolio_size}
- Knowledge depth: {profile.product_knowledge_depth}
- Proofs: max {profile.proof_count} pieces of evidence per question
- Proof handling: {profile.proof_handling}

CRITICAL LIMITATIONS (NEVER VIOLATE):
{chr(10).join(f'- {limit}' for limit in profile.limitations)}

CONVERSATION RULES (HIGHEST PRIORITY):
- NEVER ask the same question twice in the same conversation
- If product/context already mentioned, continue from there
- Remember what was discussed and build on it
- Continue conversation naturally without loops
- Speak like a real doctor in a clinical detailing session
- Lead actively but progressively
- Never hallucinate - use only JSON + RAG context
- Keep answers concise, professional, clinically oriented
- Validate claims with provided evidence

INTERACTION GOALS:
- Create natural, flowing clinical detailing session
- Progressive discovery of medical rep capabilities
- Constructive feedback on performance
- Build toward engagement and next steps

Always base your answer ONLY on the provided context. If information is incomplete, give brief coaching hint then ask next best training question."""

    return prompt


def get_competency_training_brief(level: CompetencyLevel) -> str:
    """Return a concise competency brief for training-brain prompts and UI."""

    profile = COMPETENCY_LEVELS[level]
    limits = "\n".join(f"- {limit}" for limit in profile.limitations)
    core = f"""LEVEL {profile.name}
PROFILE: {profile.profile_description}

VISIT STRUCTURE:
{profile.visit_structure}

QUESTIONING:
- Ask between {profile.min_questions}-{profile.max_questions} questions
- Depth: {profile.question_depth}

OBJECTIONS:
- Handle up to {profile.objections_to_handle} objections
- Types: {', '.join(profile.objection_types)}

ARGUMENTATION:
- Format: {profile.argument_format}
- Arguments per product: {profile.args_per_product}

KNOWLEDGE SCOPE:
- Portfolio: {profile.product_portfolio_size}
- Depth: {profile.product_knowledge_depth}
- Proofs: max {profile.proof_count}

LIMITS:
{limits}

KPI:
- {profile.kpi_score}
- {profile.objection_success_rate}
- {profile.engagement_rate}
- {profile.completion_rate}"""
    # Global generation constraints appended to every training brief to enforce
    # strict JSON-first behavior, RAG fallback, no-hallucination policy and
    # level-adapted objection rules. This block captures the user's requested
    # requirements and is intended to be treated as HIGH-PRIORITY instructions
    # by any LLM prompt consumer (TrainingBrain, DoctorLLM, etc.).

    global_constraints = """
GLOBAL GENERATION CONSTRAINTS (HIGHEST PRIORITY):
SOURCE OF TRUTH:
1) PRIMARY: JSON knowledge from the Knowledge Builder — always check this first and use it as authoritative.
2) SECONDARY: RAG retrieval only when JSON is missing or incomplete.
3) NO SOURCE: If neither JSON nor RAG contains the answer, DO NOT INVENT — reply exactly: "I will confirm this information.".

VALIDATION FLAG (MANDATORY):
- OK: proceed normally.
- CONTRADICTION_DETECTED: the user is wrong vs JSON.
- If CONTRADICTION_DETECTED: STOP normal flow, do not ask questions, explicitly state the claim is incorrect/false, provide the corrected version from JSON (or validated RAG fallback if JSON missing), and briefly explain why.

HARD RULES (NO EXCEPTIONS):
- NEVER contradict validated JSON data.
- NEVER hallucinate medical information.
- NEVER validate incorrect claims from the user; always challenge them per level rules.

TRAINER MODE / ERROR HANDLING (LEVEL ADAPTATION):
BEGINNER: gentle correction, still explicit that claim is incorrect.
JUNIOR: indicate clear likely mistake and correct with source-backed wording.
CONFIRMED: clear correction with explicit incorrectness statement.
EXPERT: firm correction, no hesitation, explicit false statement when contradicted.

NON-REPETITION AND STYLE:
- NEVER repeat identical phrasing; always reformulate greetings, explanations, and arguments.
- Use natural, professional, doctor-like tone; concise and adaptive (no bullet lists in final output).

CONTEXT USAGE:
- Prioritize JSON > RAG. If JSON and RAG conflict, prefer JSON and flag the inconsistency.
- Use RAG only to supplement missing JSON facts; always verify RAG content before use.

FINAL RULE:
If a statement conflicts with JSON, treat JSON as correct; if it conflicts with RAG, verify before using; if the user makes a claim, challenge and correct it according to the level rules.
"""

    return core + "\n\n" + global_constraints
