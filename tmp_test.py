"""
Quick diagnostic test for Supabase Vector DB retrieval.
Tests:
1. Documents table has data
2. match_documents RPC works with current parameters
"""
import os
from dotenv import load_dotenv
load_dotenv(".env")

try:
    from supabase.client import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("❌ SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in .env")
        print("   Add them to test locally. They should match your GitHub secrets.")
        exit(1)
    
    client = create_client(url, key)
    print("✅ Supabase client connected")
    
    # Test 1: Check documents table count
    res = client.table("documents").select("id", count="exact").limit(1).execute()
    print(f"\n📊 Documents table count: {res.count}")
    
    if res.count == 0:
        print("❌ PROBLEM: Documents table is EMPTY! Vector embeddings were never inserted.")
        print("   The sync script may have failed to add vectors, or the migration was never run.")
        exit(1)
    
    # Test 2: Check a sample document
    sample = client.table("documents").select("id, content").limit(1).execute()
    if sample.data:
        content_preview = sample.data[0].get("content", "")[:200]
        print(f"\n📝 Sample document content:\n   {content_preview}...")
    
    # Test 3: Test match_documents RPC with BOTH 2-param and 3-param calls
    from langchain_mistralai import MistralAIEmbeddings
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    
    test_query = "education schemes"
    print(f"\n🔍 Testing vector search for: '{test_query}'")
    embed = embeddings.embed_query(test_query)
    print(f"   Embedding dimension: {len(embed)}")
    
    # Test A: 2-param call (what the code currently does)
    print("\n--- Test A: 2-param RPC call (no match_threshold) ---")
    try:
        res_a = client.rpc("match_documents", {
            "query_embedding": embed,
            "match_count": 3
        }).execute()
        print(f"   ✅ SUCCESS! Got {len(res_a.data)} results")
        for r in res_a.data[:2]:
            print(f"   📄 {r.get('content', '')[:100]}...")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test B: 3-param call (with match_threshold)
    print("\n--- Test B: 3-param RPC call (with match_threshold) ---")
    try:
        res_b = client.rpc("match_documents", {
            "query_embedding": embed,
            "match_threshold": 0.3,
            "match_count": 3
        }).execute()
        print(f"   ✅ SUCCESS! Got {len(res_b.data)} results")
        for r in res_b.data[:2]:
            print(f"   📄 {r.get('content', '')[:100]}...")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
