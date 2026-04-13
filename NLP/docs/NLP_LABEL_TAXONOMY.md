# NLP Label Taxonomy for ALIA

This taxonomy is derived from the training manuals, the competency reference, and the progression matrix.
It is designed for message classification, safety routing, retrieval steering, and training analytics.

## 1. Top-Level Intents

Use one primary intent per user message.

| Intent | Meaning | Typical Examples |
|---|---|---|
| `product_information_request` | User asks about a product, indication, positioning, or feature | "Tell me about Product X" |
| `dosage_question` | User asks about dosage, schedule, administration, or posology | "How should it be dosed?" |
| `safety_question` | User asks about contraindications, tolerance, interactions, adverse effects, or caution | "What are the risks in renal patients?" |
| `objection_handling` | User raises a resistance, doubt, or pushback | "I already use another product" |
| `training_simulation` | User wants role-play, visit simulation, or physician/rep scenario | "Act like a skeptical doctor" |
| `crm_follow_up` | User asks for follow-up, next step, relance, or visit planning | "What should I do after the visit?" |
| `competency_assessment` | User asks to be scored, evaluated, or leveled | "Am I Junior or Confirmed?" |
| `visit_format_request` | User asks for Flash / Standard / Approfondie structure | "Give me a standard visit script" |
| `sales_methodology_request` | User asks about process, objection method, argumentation, or closing | "How do I handle objections?" |
| `general_greeting` | Greeting or small talk with no domain task | "Hello" |
| `other` | Anything not covered above | - |

## 2. Secondary Tags

Secondary tags can be multi-label and are useful for analytics and routing.

### 2.1 Visit Phase Tags

- `opening_permission`
- `discovery_sondage`
- `summary_reformulation`
- `objection_handling`
- `argumentation`
- `closing_commitment`
- `crm_followup`
- `second_visit_cycle`

### 2.2 Visit Format Tags

- `flash_visit`
- `standard_visit`
- `deep_visit`

### 2.3 Objection Tags

- `no_time`
- `habitual_use`
- `not_convinced`
- `too_expensive`
- `needs_proof`
- `safety_concern`
- `tolerance_concern`
- `access_problem`

### 2.4 Safety and Compliance Tags

- `patient_specific_advice_request`
- `diagnosis_request`
- `off_label_request`
- `high_risk_interaction`
- `contraindication_query`
- `overclaim_risk`
- `source_required`

### 2.5 Competency Tags

- `beginner_level`
- `junior_level`
- `confirmed_level`
- `expert_level`
- `needs_coaching`
- `low_confidence`

## 3. Entity Types

Extract these entity types when present:

| Entity Type | Examples |
|---|---|
| `product_name` | Product names, brands, references |
| `active_ingredient` | Molecules or components |
| `indication` | Disease, syndrome, or target use |
| `patient_profile` | Pediatric, adult, renal patient, special population |
| `dosage` | Dose, frequency, route, duration |
| `benefit` | Efficacy, tolerance, observance, convenience |
| `contraindication` | Known contraindication or restriction |
| `adverse_event` | Side effect, tolerance issue, interaction |
| `competency_level` | Debutant, Junior, Confirme, Expert |
| `visit_format` | Flash, Standard, Approfondie |
| `objection_phrase` | Exact user objection |
| `proof_reference` | Study, evidence, source, repere |

## 4. Structured Output Schema

The NLP layer should return a strict JSON object:

```json
{
  "intent": "safety_question",
  "secondary_tags": ["objection_handling", "source_required"],
  "entities": {
    "product_name": ["Product X"],
    "patient_profile": ["renal patient"],
    "contraindication": [],
    "adverse_event": ["interaction risk"]
  },
  "topics": ["safety", "dosage"],
  "objections": ["I already use another product"],
  "action_items": ["check official label", "prepare follow-up source"],
  "safety_flags": ["patient_specific_advice_request"],
  "rewritten_query": "Product X safety dosage renal patient interaction",
  "confidence": 0.92
}
```

## 5. Tagging Rules

- Always assign exactly one primary intent.
- Use multiple secondary tags when relevant.
- Prefer explicit domain language from the manuals over generic tags.
- Safety tags must override convenience tags if both are present.
- If confidence is low, keep the primary intent but add `low_confidence`.

## 6. Mapping From Teacher Files

- The 6-step visit process drives the visit phase tags.
- The objection and closing sections drive objection and action-item tags.
- The competency reference drives level tags and evaluation labels.
- The Excel matrix drives scoring thresholds and checklist signals.
