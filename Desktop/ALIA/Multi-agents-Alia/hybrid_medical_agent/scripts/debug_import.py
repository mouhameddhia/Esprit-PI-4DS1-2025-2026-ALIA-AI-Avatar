from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
print('ROOT=', ROOT)
print('exists', ROOT.exists())
print('listdir:', [p.name for p in ROOT.iterdir()])
sys.path.insert(0, str(ROOT))
print('sys.path[0]=', sys.path[0])
try:
    import hybrid_medical_agent
    print('imported hybrid_medical_agent at', hybrid_medical_agent.__file__)
except Exception as e:
    print('IMPORT_ERROR:', e)
