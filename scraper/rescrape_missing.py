"""
rescrape_missing.py
                                                            
Finds all schemes in ChromaDB that have "Not found" for
benefits/eligibility/etc and refetches their detail pages.

Can be run:
  - Manually:   python scraper/rescrape_missing.py
  - Automatic:  called by scheduler.py every day at 3 AM

Requirements:
    pip install playwright langchain-chroma langchain-huggingface
    pip install langchain-mistralai sentence-transformers
    playwright install chromium
"""

import os
import re
import json
import time
import sqlite3
import sys

# Ensure 'database' and 'utils' are in the path
sys.path.append(os.path.abspath(os.curdir))

from dotenv import load_dotenv
from database.db import SessionLocal
from database.models import Scheme
from rag.llm import get_vector_db, get_embedding_model, SchemeOutput
from utils.notifier import broadcast_new_schemes
load_dotenv()

#    Config                                                                     

# Always find vector_db/ relative to project root
# (works whether run from scraper/ or project root)
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")

BATCH_DELAY    = 5    # seconds between Playwright fetches
MAX_RETRIES    = 3

#                                                                              


def get_llm():
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(model="mistral-small-latest", temperature=0.1)


def fetch_with_playwright(url: str, timeout: int = 25000) -> str:
    """Fetch JS-rendered page text using Playwright."""
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                ),
                java_script_enabled=True,
            )
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            try:
                page.wait_for_selector(
                    "section, .scheme-detail, main, article", timeout=8000
                )
            except Exception:
                pass
            page.wait_for_timeout(4000)
            text    = page.evaluate("document.body.innerText")
            browser.close()
            cleaned = re.sub(r'\s{2,}', ' ', text or "").strip()
            return cleaned[:10000]
    except Exception as e:
        print(f"      Playwright error: {e}")
        return ""


def extract_fields_with_llm(page_text: str, scheme_name: str) -> dict:
    """Use LLM to extract structured scheme fields from page text."""
    if not page_text or len(page_text) < 150:
        return {}

    # Use structured output for robustness
    from rag.llm import get_llm
    llm    = get_llm().with_structured_output(SchemeOutput)
    
    prompt = f"""Extract information about the government scheme "{scheme_name}" from this webpage text.

Webpage text:
\"\"\"
{page_text[:7000]}
\"\"\"

Extract these fields:
- description: Overall description of the scheme
- benefits: Benefits provided (financial, material, services)
- eligibility: Who can apply   age, income, caste, occupation, state
- documents_required: List of required documents (semicolon-separated)
- application_process: Step-by-step how to apply (format as "Step 1: ... Step 2: ...")

Rules:
- Copy text closely from the page. Do not invent anything.
- If a field is truly not present, use "Not available".
- Safety: If your description or benefits contain double quotes, you MUST escape them with a backslash (\").
- Reply ONLY with a valid JSON object.
"""

    for attempt in range(1, 4):
        try:
            # Mistral handles the JSON formatting and escaping automatically with structured output
            res = llm.invoke(prompt)
            if res:
                return res.dict()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "capacity" in err_str or "rate limit" in err_str:
                sleep_time = 15 * attempt
                print(f"      Rate limit (429) hit. Sleeping for {sleep_time}s... (Attempt {attempt}/3)")
                time.sleep(sleep_time)
            else:
                print(f"      Structured Extraction error: {e}")
                
    return {}


def build_document_text(name, description, category, benefits, eligibility,
                         application_process, documents_required, state, link) -> str:
    """Build document string in same format as original scraper."""
    return (
        f"\n    Scheme name :{name}\n"
        f"    Description :{description}\n"
        f"    category : {category}\n"
        f"    benefits : {benefits}\n"
        f"    eligibility : {eligibility}\n"
        f"    application_process : {application_process}\n"
        f"    required_documents : {documents_required}\n"
        f"    state : {state}\n"
        f"    Link : {link}\n\n    "
    )


def get_missing_schemes() -> list:
    """Query Supabase for schemes with missing or 'Not found' fields."""
    session = SessionLocal()
    try:
        # Search for schemes that have 'Not available' or 'Not found' in benefits or eligibility
        missing_schemes = session.query(Scheme).filter(
            (Scheme.benefits.ilike('%Not available%')) | 
            (Scheme.benefits.ilike('%Not found%')) |
            (Scheme.eligibility.ilike('%Not available%')) |
            (Scheme.eligibility.ilike('%Not found%'))
        ).all()
        
        results = []
        for s in missing_schemes:
            results.append({
                "id": s.id,
                "name": s.scheme_name,
                "url": s.application_link,
                "category": s.category,
                "state": s.state or "Gujarat"
            })
        return results
    except Exception as e:
        print(f"  Error querying missing schemes: {e}")
        return []
    finally:
        session.close()


def update_cloud_db(scheme_id: int, fields: dict, doc_text: str):
    """Update both relational and vector stores on Supabase."""
    session = SessionLocal()
    try:
        # 1. Update Relational DB
        scheme = session.query(Scheme).filter(Scheme.id == scheme_id).first()
        if scheme:
            # Always stringify   LLM may return dicts/lists for some fields
            def _to_str(val, fallback):
                if val is None:
                    return fallback
                if isinstance(val, (dict, list)):
                    return json.dumps(val, ensure_ascii=False)
                return str(val)

            scheme.description        = _to_str(fields.get("description"),        scheme.description)
            scheme.benefits           = _to_str(fields.get("benefits"),            scheme.benefits)
            scheme.eligibility        = _to_str(fields.get("eligibility"),         scheme.eligibility)
            scheme.application_process= _to_str(fields.get("application_process"),scheme.application_process)
            scheme.documents_required = _to_str(fields.get("documents_required"),  scheme.documents_required)
            scheme.missing_count      = 0
            session.commit()
            print(f"    Relational DB updated")

        # 2. Update Vector DB (Delete old + add new)
        try:
            vector_db = get_vector_db()
            if vector_db:
                from supabase.client import create_client
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                supabase_client = create_client(supabase_url, supabase_key)

                # Delete old vectors for this scheme (best-effort, ignore errors)
                try:
                    supabase_client.table("documents").delete().filter(
                        "content", "ilike", f"%Scheme name :{scheme.scheme_name}%"
                    ).execute()
                except Exception as del_e:
                    pass  # Non-critical   old vector will just become stale

                # Add updated document
                vector_db.add_texts([doc_text])
                print(f"    Vector Index updated")
        except Exception as vec_e:
            print(f"      Vector update skipped: {vec_e}")

    except Exception as e:
        print(f"      Cloud DB update error: {e}")
        session.rollback()
    finally:
        session.close()


def main():
    """Main function   called by GitHub Actions or run manually."""
    print("\n  Yojana AI   Cloud Re-scraper for missing scheme details")
    print("=" * 60)

    #    Find schemes with missing data                                        
    missing = get_missing_schemes()
    success       = 0
    failed        = []
    updated_names = []

    print(f"  Found {len(missing)} schemes with missing details in Cloud DB")

    if not missing:
        print("  Nothing to fix! All schemes have complete data.")
        return

    for i, scheme in enumerate(missing, 1):
        name = scheme["name"]
        url  = scheme["url"]

        print(f"[{i}/{len(missing)}] {name}")
        print(f"    {url}")

        #    Fetch page                                                        
        page_text = ""
        for attempt in range(1, MAX_RETRIES + 1):
            page_text = fetch_with_playwright(url)
            if len(page_text) > 200:
                break
            print(f"      Attempt {attempt} got too little content, retrying...")
            time.sleep(3)

        if len(page_text) < 200:
            print(f"    Could not fetch page after {MAX_RETRIES} attempts   skipping\n")
            failed.append(name)
            continue

        #    Extract fields with LLM                                           
        print(f"    Got {len(page_text)} chars   extracting fields with LLM...")
        fields = extract_fields_with_llm(page_text, name)

        if not fields or all(
            v in ("Not available", "", "Not found")
            for v in fields.values()
        ):
            print(f"    LLM could not extract fields   skipping\n")
            failed.append(name)
            continue

        #    Build updated document and save to ChromaDB                       
        new_doc = build_document_text(
            name                = name,
            description         = fields.get("description",       "Not available"),
            category            = scheme["category"],
            benefits            = fields.get("benefits",          "Not available"),
            eligibility         = fields.get("eligibility",       "Not available"),
            application_process = fields.get("application_process","Not available"),
            documents_required  = fields.get("documents_required", "Not available"),
            state               = scheme["state"],
            link                = url,
        )

        print(f"    Updating Cloud Database...")
        update_cloud_db(scheme["id"], fields, new_doc)
        success += 1
        updated_names.append(name)

        print(f"    Done\n")
        time.sleep(BATCH_DELAY)

    #    Summary                                                               
    print("=" * 60)
    print(f"  Successfully updated: {success}/{len(missing)} schemes")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for f in failed:
            print(f"   - {f}")
            
    # Broadcast notification if any schemes were recovered
    if updated_names:
        from utils.notifier import broadcast_new_schemes
        broadcast_new_schemes(updated_names, is_update=True)
        
    print("  Re-scraping complete!")


if __name__ == "__main__":
    main()