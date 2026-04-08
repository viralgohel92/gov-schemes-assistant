import os
from supabase.client import create_client
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def verify():
    print("🔍 Yojana AI — Supabase Cloud Diagnostic")
    print("=" * 45)
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    db_url = os.getenv("DATABASE_URL")
    
    # 1. Test REST API Connection
    print("📡 Testing REST API Connection...", end=" ", flush=True)
    try:
        supabase = create_client(url, key)
        # Try to list tables (implicitly via any query)
        supabase.table("schemes").select("count", count="exact").limit(1).execute()
        print("✅ SUCCESS")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # 2. Test PostgreSQL Connection
    print("🐘 Testing PostgreSQL Connection...", end=" ", flush=True)
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ SUCCESS")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # 3. Check for Vector Extension
    print("🧠 Checking for pgvector extension...", end=" ", flush=True)
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).fetchone()
            if res:
                print("✅ FOUND")
            else:
                print("❌ MISSING (Run 'CREATE EXTENSION IF NOT EXISTS vector;' in Supabase SQL Editor)")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # 4. Check for 'match_documents' function
    print("🔎 Checking for 'match_documents' function...", end=" ", flush=True)
    try:
        with engine.connect() as conn:
            query = """
            SELECT 1 FROM pg_proc JOIN pg_namespace ON pg_proc.pronamespace = pg_namespace.oid 
            WHERE proname = 'match_documents' AND nspname = 'public';
            """
            res = conn.execute(text(query)).fetchone()
            if res:
                print("✅ FOUND")
            else:
                print("❌ MISSING (Needed for AI search to work)")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    print("=" * 45)

if __name__ == "__main__":
    verify()
