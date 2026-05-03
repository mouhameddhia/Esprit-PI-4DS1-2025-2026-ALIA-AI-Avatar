"""Safety detection term sets — centralised constants."""

from typing import List

# Patient-specific advice triggers
PATIENT_CONTEXT_TERMS: List[str] = [
    "my patient", "for this patient", "safe for", "suitable for", "use in",
    "given to", "prescribe", "administer", "recommend for", "appropriate for",
    "contraindicated in", "can i give", "can she take", "can he take",
    "is it safe", "is product safe",
]

HIGH_RISK_POPULATIONS: List[str] = [
    "pregnan", "renal impairment", "renal failure", "renal disease", "renal",
    "pediatric", "child", "elderly", "geriatric", "hepatic", "liver disease",
    "cardiac", "heart failure", "nursing", "breastfeed", "lactation", "dialysis",
    "neonatal", "infant", "trimester",
]

# Teratogenicity → patient_specific_advice_request even without explicit patient framing
TERATOGEN_TERMS: List[str] = [
    "teratogen", "teratogenic", "embryotox", "fetal risk", "fetal safety",
    "category c", "category d", "category x", "pregnancy category",
    "birth defect", "congenital", "placental", "perinatal",
]

# Clinical toxicity properties → contraindication_query
TOXICITY_FLAGS: List[str] = [
    "nephrotox", "hepatotox", "cardiotox", "qt prolongation", "torsades",
    "photosensitivity", "black box", "boxed warning", "serious adverse",
    "rhabdomyolysis", "serotonin syndrome",
]

CONTRAINDICATION_TERMS: List[str] = [
    "contraindicated", "avoid in", "not recommended in",
    "should not use", "must not",
]

HIGH_RISK_INTERACTION_TERMS: List[str] = [
    "drug interaction", "drug-drug", "drug–drug", "concurrent use",
    "co-administration", "concomitant", "combined with", "given with",
    "interacts with", "combination with",
]

SOURCE_REQUIRED_TERMS: List[str] = [
    "need a source", "show me evidence", "give me a reference",
    "study supporting", "where is the proof", "cite your source",
]
