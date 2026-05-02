from pathlib import Path
import sys
import json

# Ensure package imports from agent/ and rag_knowledge_builder
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

try:
    from hybrid_medical_agent.agent.json_retriever import JSONRetriever
except Exception as e:
    print(f"IMPORT_ERROR: {e}")
    raise


def check_payload(payload: dict) -> dict:
    issues = []
    keys = set(payload.keys())
    # Basic required fields
    if not payload.get('drug_name') and not payload.get('document_name'):
        issues.append('missing_drug_or_document_name')

    # Topics to check
    def has_topic(p, names):
        for n in names:
            v = p.get(n)
            if v not in (None, "", {}, []):
                return True
        return False

    if not has_topic(payload, ['indications']):
        issues.append('missing_indications')
    if not has_topic(payload, ['dosage', 'posology']):
        issues.append('missing_dosage')
    if not has_topic(payload, ['warnings', 'precautions', 'side_effects']):
        issues.append('missing_warnings_precautions')

    # Check for benefits heuristics in any string fields
    def find_benefits_in_text(obj):
        texts = []
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                texts.extend(find_benefits_in_text(v))
        elif isinstance(obj, list):
            for v in obj:
                texts.extend(find_benefits_in_text(v))
        return texts

    all_text = "\n".join(find_benefits_in_text(payload))
    benefits_count = 0
    for line in all_text.splitlines():
        l = line.strip().lower()
        if 'benefit' in l or 'bénéf' in l or 'avantage' in l:
            benefits_count += 1
    if benefits_count < 2:
        issues.append(f'low_benefits_count:{benefits_count}')

    # Check product-level entries
    products = payload.get('products')
    if isinstance(products, dict) and products:
        for pname, pval in products.items():
            sub_issues = []
            if not has_topic(pval, ['indications']):
                sub_issues.append('missing_indications')
            if not has_topic(pval, ['dosage', 'posology']):
                sub_issues.append('missing_dosage')
            if not has_topic(pval, ['warnings', 'precautions', 'side_effects']):
                sub_issues.append('missing_warnings_precautions')
            p_text = "\n".join(find_benefits_in_text(pval))
            p_benefits = sum(1 for line in p_text.splitlines() if 'benefit' in line.lower() or 'bénéf' in line.lower() or 'avantage' in line.lower())
            if p_benefits < 2:
                sub_issues.append(f'low_benefits_count:{p_benefits}')
            if sub_issues:
                issues.append(f'product:{pname}:' + ";".join(sub_issues))

    return {
        'source_file': payload.get('__source_file'),
        'aliases': payload.get('__aliases', []),
        'issues': issues,
    }


if __name__ == '__main__':
    retriever = JSONRetriever(workspace_root=ROOT)
    results = []
    for payload in retriever.payloads:
        r = check_payload(payload)
        results.append(r)
    out = {
        'workspace_root': str(ROOT),
        'total_payloads': len(retriever.payloads),
        'sample_drugs': retriever.sample_drugs,
        'results': results,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
