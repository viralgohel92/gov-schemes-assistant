from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
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

engine = None
try:
    if not DATABASE_URL or "None" in str(DATABASE_URL):
        print("⚠️ Warning: DATABASE_URL is missing. Using in-memory fallback.")
        engine = create_engine("sqlite:///:memory:")
    else:
        # Use simple connection for serverless/Supabase compatibility
        engine = create_engine(
            DATABASE_URL, 
            pool_pre_ping=True,
            pool_recycle=3600
        )
        # Quick connectivity test
        with engine.connect() as conn:
            pass
        print("✅ Database engine initialized.")
except Exception as e:
    print(f"⚠️ Warning: Database connection failed. Using fallback: {e}")
    engine = create_engine("sqlite:///:memory:")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)