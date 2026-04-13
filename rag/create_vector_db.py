import os
import sys
import pandas as pd
from langchain_core.documents import Document

# Ensure repo root is on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv()

from rag.llm import get_vector_db, get_embedding_model

def populate_vector_db(csv_path="data/processed/scraped_schemes.csv", target="cloud"):
    """
    Populates the vector database (Chroma or Supabase) from the processed CSV.
    
    Args:
        csv_path: Path to the processed schemes CSV.
        target: "cloud" for Supabase, "local" for Chroma.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading schemes from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Optional: Fill NaNs to avoid string conversion issues
    df = df.fillna("Not found")
    
    documents = []
    texts = []

    for _, row in df.iterrows():
        # Consistent formatting with sync_schemes.py
        text = f"""
    Scheme name :{row['scheme_name']}
    Description :{row['details']}
    category : {row['category']}
    benefits : {row['benefits']}
    eligibility : {row['eligibility']}
    application_process : {row['application_process']}
    required_documents : {row['documents_required']}
    state : {row['state']}
    Link : {row['scheme_link']}
    """
        texts.append(text)
        documents.append(Document(page_content=text, metadata={"source": row['scheme_link']}))

    print(f"Prepared {len(texts)} documents.")

    # Initialize the correct DB
    # If target is cloud, ensure SUPABASE environment variables are set
    if target == "cloud":
        os.environ["USE_SUPABASE"] = "true" # Hint for get_vector_db if needed
    else:
        os.environ["USE_SUPABASE"] = "false"

    vector_db = get_vector_db()
    
    if vector_db is None:
        print("Error: Could not initialize Vector DB.")
        return

    print(f"Indexing {len(texts)} documents into {type(vector_db).__name__} ({target})...")
    
    # Batch processing for cloud to avoid timeouts
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        vector_db.add_texts(batch)

    print(f"Successfully indexed {len(texts)} schemes.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Populate Yojana AI Vector Database")
    parser.add_argument("--csv", type=str, default="data/processed/scraped_schemes.csv", help="Path to CSV")
    parser.add_argument("--target", type=str, choices=["local", "cloud"], default="cloud", help="Target DB")
    
    args = parser.parse_args()
    
    populate_vector_db(csv_path=args.csv, target=args.target)
