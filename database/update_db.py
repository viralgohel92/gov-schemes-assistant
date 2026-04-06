import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Ensure repo root is on PYTHONPATH
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

load_dotenv()

# Build connection string
db_user = os.getenv("DB_USERNAME")
db_pass = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOSTNAME")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_url)

def update_schema():
    with engine.connect() as conn:
        print("🔗 Connecting to database to update schema...")
        # Add columns if they don't exist
        # PostgreSQL syntax for adding multiple columns if they don't exist is tricky without a complex block
        # We'll do them one by one
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS telegram_chat_id VARCHAR;"))
            print("✅ Added 'telegram_chat_id' column.")
        except Exception as e:
            print(f"⚠️  Note (TG Column): {e}")
            
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS whatsapp_number VARCHAR;"))
            print("✅ Added 'whatsapp_number' column.")
        except Exception as e:
            print(f"⚠️  Note (WA Column): {e}")

        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS otp VARCHAR;"))
            print("✅ Added 'otp' column.")
        except Exception as e:
            print(f"⚠️  Note (OTP Column): {e}")

        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS otp_expiry TIMESTAMP;"))
            print("✅ Added 'otp_expiry' column.")
        except Exception as e:
            print(f"⚠️  Note (OTP Expiry Column): {e}")

        conn.commit()
        print("✨ Database successfully updated!")

if __name__ == "__main__":
    try:
        update_schema()
    except Exception as e:
        print(f"❌ Error during update: {e}")
