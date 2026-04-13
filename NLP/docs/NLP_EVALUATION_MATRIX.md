# NLP Evaluation Matrix for ALIA

This matrix maps the four competency levels from the teacher files to measurable NLP and simulation behaviors.

## 1. Evaluation Dimensions

Evaluate each conversation or simulation across the following dimensions:

- opening and permission
- discovery quality
- synthesis / reformulation
- objection handling
- argumentation quality
- closing and commitment
- CRM completeness
- safety and compliance
- adaptation to doctor style
- retrieval grounding

## 2. Level-Based Matrix

| Dimension | Débutant | Junior | Confirmé | Expert |
|---|---|---|---|---|
| Opening / Permission | Says hello, asks for time, follows script | Handles brief refusal or urgency | Adapts to stress, secretary, timing | Creates positive climate in tension |
| Discovery | 1 to 2 simple questions | 2 to 4 relevant questions with follow-ups | Identifies main and secondary need | Uses discovery strategically for cycle long |
| Synthesis | Basic summary | Reformulates clearly | Validates need and aligns message | Summarizes with precision and pace control |
| Objections | Handles 1 standard objection | Handles 2 common objections with A-C-R-V | Handles 3 varied objections and distinguishes type | Handles difficult objections and pressure calmly |
| Argumentation | 1 simple product message | Need -> advantage -> proof -> usage with 2-3 arguments | Segment-based argumentation, relevant proof | High precision, balanced benefit and limit framing |
| Closing | Gets a micro-commitment | Closes at the right time | Closes after signals and interruptions | Closes strategically with cycle management |
| CRM | Minimal note | Structured CRM and follow-up plan | Complete traceability and next step | Clean, reliable, coachable CRM discipline |
| Safety / Compliance | Avoids obvious errors | Uses cautious phrasing | Respects regulatory limits consistently | Zero overclaim, zero compliance errors |
| Adaptation | Mostly scripted | Basic adaptation to context | Adapts to doctor profile and SONCAS | Real-time adaptation across styles |
| Retrieval Grounding | Uses only basic knowledge | Retrieves and cites key supports | Grounded in broad portfolio knowledge | Grounded in full portfolio and limits |

## 3. Scoring Guidance

Use a 0 to 10 score per dimension.

### 3.1 Débutant

- score target: 0 to 6
- must respect the 6-step structure at a basic level
- may handle only one standard objection
- must avoid compliance mistakes

### 3.2 Junior

- score target: 7 to 7.9
- must handle the standard visit formats
- must ask relevant discovery questions
- must handle common objections using A-C-R-V
- CRM should be structured

### 3.3 Confirmé

- score target: 8 to 8.9
- must adapt to different doctor styles
- must distinguish objection types
- must segment argumentation by patient profile
- must manage interruptions and maintain closing discipline

### 3.4 Expert

- score target: 9 to 10
- must handle difficult visits and cycle-long scenarios
- must maintain highly precise, safe, and compliant language
- must show strong evidence handling and coaching-level mastery

## 4. KPI Thresholds

| Transition | Required Thresholds |
|---|---|
| Débutant -> Junior | score >= 7/10, structure OK >= 80%, engagement >= 60%, compliance errors = 0 |
| Junior -> Confirmé | score >= 8/10, A-C-R-V validated >= 70%, profile adaptation >= 60%, CRM complete >= 80%, compliance errors = 0 |
| Confirmé -> Expert | score >= 9/10, difficult visits success >= 70%, cycle-long success >= 60%, clean language >= 95%, compliance errors = 0 |

## 5. NLP Evaluation Labels

For each conversation, store:

- `primary_intent`
- `visit_phase_detected`
- `objection_type`
- `doctor_style_detected`
- `soncas_tag`
- `safety_flag`
- `retrieval_hit`
- `closing_success`
- `crm_complete`
- `confidence`

## 6. Pass/Fail Rules

- pass only if compliance errors are zero
- if safety flags are present and not handled, the run should fail
- if the wrong visit phase is used, mark the run as a partial fail
- if the model makes unsupported claims, fail the run

## 7. Recommended Test Set

Build test prompts for:

- short visit requests
- long discovery scenarios
- objection-heavy scenarios
- safety-sensitive questions
- level assessment prompts
- CRM follow-up prompts

## 8. Automation Idea

Each time a conversation is finalized, run an evaluator that checks:

1. the detected intent
2. the detected visit phase
3. the objection type
4. the safety flags
5. the completeness of the CRM summary
6. whether the retrieved context was actually used
