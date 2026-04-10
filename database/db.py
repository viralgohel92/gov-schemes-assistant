from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Prioritize a single DATABASE_URL (standard for Supabase/Vercel)
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    db_username = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_hostname = os.getenv("DB_HOSTNAME")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
    if db_username and db_password and db_hostname:
        DATABASE_URL = f"postgresql://{db_username}:{db_password}@{db_hostname}:{db_port}/{db_name}"

# Fix legacy 'postgres://' schema often used by cloud providers
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL or "None" in str(DATABASE_URL):
    raise ConnectionError("FATAL: DATABASE_URL is missing or invalid in environment variables.")

try:
    # Use connection pooling and pre-ping to handle serverless connections/restarts
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    # Quick connectivity test
    with engine.connect() as conn:
        print("Database connection verified (Supabase Cloud).")
except Exception as e:
    # Stop execution if connection fails
    raise ConnectionError(f"FATAL: Failed to connect to Supabase: {e}")

#    Auto-create tables and pgvector extension on first run                     
def init_db():
    """Creates tables if they don't exist. Safe to run multiple times."""
    from database.models import Base
    try:
        with engine.connect() as conn:
            # Enable pgvector for AI search (no-op if already enabled)
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        # Create all ORM-defined tables (schemes, users, etc.)
        Base.metadata.create_all(bind=engine)
        print("Database schema initialized (tables ready).")
    except Exception as e:
        print(f"Schema init warning: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)