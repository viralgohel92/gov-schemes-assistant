import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from database.db import DATABASE_URL
from database.models import Base, Scheme
from sqlalchemy.orm import sessionmaker
from langchain_mistralai import MistralAIEmbeddings
from supabase.client import create_client

load_dotenv()
print(f"📍 Current Working Directory: {os.getcwd()}")
print(f"📂 Checking for .env: {os.path.exists('.env')}")

# --- Config ---
PROCESSED_CSV = os.path.join(os.getcwd(), "data", "processed", "scraped_schemes.csv")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def migrate():
    print("🚀 Starting Migration to Supabase...")
    
    db_target = DATABASE_URL.split("@")[-1] if DATABASE_URL else "None"
    print(f"🔗 Connecting to Database: {db_target}")

    # 1. Connect to PostgreSQL
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # 2. Clear Existing Tables
        print("🗑️ Clearing existing tables...")
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS documents CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS notifications CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS chat_history CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS users CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS schemes CASCADE;"))
            conn.commit()

        # 3. Enable pgvector extension
        print("⚙️ Enabling pgvector extension...")
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        # 4. Create Tables using SQLAlchemy
        print("🏗️ Creating tables from models...")
        Base.metadata.create_all(bind=engine)

        # 5. Load Data from CSV
        print(f"📖 Reading schemes from {PROCESSED_CSV}...")
        if not os.path.exists(PROCESSED_CSV):
            print(f"❌ Error: {PROCESSED_CSV} not found!")
            return

        df = pd.read_csv(PROCESSED_CSV)
        print(f"✅ Loaded {len(df)} schemes.")

        schemes_to_add = []
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
            schemes_to_add.append(scheme)

        print(f"💾 Inserting {len(schemes_to_add)} schemes into 'schemes' table...")
        session.bulk_save_objects(schemes_to_add)
        session.commit()
        print("✅ Schemes inserted.")

        # 6. Initialize Vector Store (using Supabase API for indexing)
        print("🌐 Indexing documents for Vector Search (Mistral API)...")
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("❌ Skipping vector indexing: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing.")
            return

        from langchain_community.vectorstores import SupabaseVectorStore
        
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        embeddings = MistralAIEmbeddings(model="mistral-embed")

        # Prepare documents for vector search
        documents = []
        for _, row in df.iterrows():
            content = f"""
            Scheme name: {row['scheme_name']}
            Description: {row['details']}
            Category: {row['category']}
            Benefits: {row['benefits']}
            Eligibility: {row['eligibility']}
            Application Process: {row['application_process']}
            Required Documents: {row['documents_required']}
            State: {row['state']}
            Link: {row['scheme_link']}
            """
            documents.append(content)

        # Batch indexing to avoid timeouts
        batch_size = 50
        print(f"📦 Indexing {len(documents)} documents in batches of {batch_size}...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            SupabaseVectorStore.from_texts(
                texts=batch,
                embedding=embeddings,
                client=supabase_client,
                table_name="documents",
                query_name="match_documents"
            )
            print(f"   - Processed {min(i+batch_size, len(documents))}/{len(documents)}")

        print("🎉 Migration and Vector Indexing complete!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    migrate()
