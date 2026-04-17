import os
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv()

from rag.translation import translate_suggestions_batch

test_data = [{"name": "Mukhyamantri Mahila Utkarsh Yojana", "category": "Women Welfare"}]
print("Translating to Hindi...")
hi = translate_suggestions_batch(test_data, "hi")
print("Hindi:", hi)

print("Translating to Gujarati...")
gu = translate_suggestions_batch(test_data, "gu")
print("Gujarati:", gu)
