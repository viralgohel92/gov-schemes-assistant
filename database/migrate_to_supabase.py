import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from database.db import DATABASE_URL
from database.models import Base, Scheme
from sqlalchemy.orm import sessionmaker
from langchain_mistralai import MistralAIEmbeddings
from supabase.client import create_client
import time

load_dotenv()
print(f"  Current Working Directory: {os.getcwd()}")
print(f"  Checking for .env: {os.path.exists('.env')}")

# --- Config ---
PROCESSED_CSV = os.path.join(os.getcwd(), "data", "processed", "scraped_schemes.csv")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def migrate():
    print("  Starting Migration to Supabase...")
    
    db_target = DATABASE_URL.split("@")[-1] if DATABASE_URL else "None"
    print(f"  Target: {db_target}")

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # 1. Clear and Prepare Tables
        print("   Resetting tables and enabling pgvector...")
        with engine.connect() as conn:
            # Drop old tables
            conn.execute(text("DROP TABLE IF EXISTS documents CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS schemes CASCADE;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create DOCUMENTS table specifically for LangChain VectorStore
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id bigserial PRIMARY KEY,
                    content text,
                    metadata jsonb,
                    embedding vector(1024)
                );
            """))
            
            # Create SEARCH function for LangChain similarity search
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION match_documents (
                    query_embedding vector(1024),
                    match_threshold float,
                    match_count int
                )
                RETURNS TABLE (
                    id bigint,
                    content text,
                    metadata jsonb,
                    similarity float
                )
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RETURN QUERY
                    SELECT
                        documents.id,
                        documents.content,
                        documents.metadata,
                        1 - (documents.embedding <=> query_embedding) AS similarity
                    FROM documents
                    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
                    ORDER BY similarity DESC
                    LIMIT match_count;
                END;
                $$;
            """))
            
            conn.commit()
            print("   Base tables and SQL functions ready.")
            # Give Supabase API a moment to refresh schema cache
            print("  Waiting for API sync...")
            time.sleep(3)

        print("   Creating tables from definitions...")
        Base.metadata.create_all(bind=engine)

        # 2. Ingest Relational Data
        print(f"  Reading schemes from: {PROCESSED_CSV}")
        if not os.path.exists(PROCESSED_CSV):
            print(f"  Error: {PROCESSED_CSV} not found!")
            return

        df = pd.read_csv(PROCESSED_CSV)
        print(f"  Loaded {len(df)} schemes from local CSV.")

        print("  Inserting schemes into Cloud DB...")
        for index, row in df.iterrows():
            scheme = Scheme(
                scheme_name=row['scheme_name'],
                application_link=row['scheme_link'],
                state=row['state'],
                category=row['category'],
                description=str(row['details']), 
                benefits=str(row['benefits']),
                eligibility=str(row['eligibility']),
                application_process=str(row['application_process']),
                documents_required=str(row['documents_required']),
                missing_count=0
            )
            session.add(scheme)
        
        session.commit()
        print(f"  Relational database sync complete ({len(df)} schemes).")

        # 3. Vector Store Initialization
        print("\n  Generating AI Vector Index (using Mistral API)...")
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("   Skipping vector sync: SUPABASE_URL or SUPABASE_KEY missing.")
            return

        from langchain_community.vectorstores import SupabaseVectorStore
        
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        embeddings = MistralAIEmbeddings(model="mistral-embed")

        # Prepare documents
        docs = []
        for _, row in df.iterrows():
            content = f"Scheme: {row['scheme_name']}\nDescription: {row['details']}\nBenefits: {row['benefits']}\nEligibility: {row['eligibility']}"
            docs.append(content)

        # Batch indexing to prevent timeouts
        BATCH_SIZE = 50
        print(f"  Uploading {len(docs)} AI embeddings in batches of {BATCH_SIZE}...")
        
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            try:
                SupabaseVectorStore.from_texts(
                    texts=batch,
                    embedding=embeddings,
                    client=supabase_client,
                    table_name="documents",
                    query_name="match_documents"
                )
                print(f"   - Progress: {min(i+BATCH_SIZE, len(docs))}/{len(docs)}")
            except Exception as inner_e:
                print(f"      Batch failed: {inner_e}")
                continue

        print("\n  SUPABASE MIGRATION COMPLETE!")

    except Exception as e:
        print(f"  FATAL ERROR: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    migrate()
