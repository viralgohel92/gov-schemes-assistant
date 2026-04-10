import psycopg2
from sqlalchemy import create_engine
import os

url = "postgresql://postgres.iimsgupngsmceieklejh:Yojana%40123%23@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"

print(f"Testing connection to Mumbai Pooler...")
try:
    engine = create_engine(url)
    with engine.connect() as conn:
        print("  Connection successful to ap-south-1 pooler!")
except Exception as e:
    print(f"  Failed ap-south-1: {e}")

url_sg = "postgresql://postgres.iimsgupngsmceieklejh:Yojana%40123%23@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
print(f"Testing connection to Singapore Pooler...")
try:
    engine = create_engine(url_sg)
    with engine.connect() as conn:
        print("  Connection successful to ap-southeast-1 pooler!")
except Exception as e:
    print(f"  Failed ap-southeast-1: {e}")
