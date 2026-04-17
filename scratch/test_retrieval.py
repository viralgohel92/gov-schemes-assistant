import os
import sys
from dotenv import load_dotenv

# Ensure repo root is on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

load_dotenv()

from rag.retriever import fetch_schemes
from rag.llm import get_vector_db

def test_retrieval(query):
    print(f"\nQuery: {query}")
    
    # 1. Test Vector Search directly
    db = get_vector_db()
    if db:
        print("--- Top 3 Vector Matches ---")
        docs = db.as_retriever(search_kwargs={"k": 3}).invoke(query)
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.page_content[:150]}...")
    
    # 2. Test full fetch_schemes
    print("\n--- fetch_schemes output ---")
    results = fetch_schemes(query, [])
    for r in results:
        print(f"Result Name: {r.scheme_name}")

if __name__ == "__main__":
    # Test 1: Exact Match (Success)
    test_retrieval("Mukhyamantri Matrushakti Yojana show me full detail of the scheme")
    
    # Test 2: Similar sounding name but doesn't exist (Should be discarded)
    # We want to see if Tier 4 "Discarding result" triggers.
    test_retrieval("Mukhyamantri Moon Landing Yojana details please")

