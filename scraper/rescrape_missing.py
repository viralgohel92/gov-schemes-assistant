"""
rescrape_missing.py
────────────────────────────────────────────────────────────
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

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

# Always find vector_db/ relative to project root
# (works whether run from scraper/ or project root)
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")

BATCH_DELAY    = 2    # seconds between Playwright fetches
MAX_RETRIES    = 2

# ─────────────────────────────────────────────────────────────────────────────


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
        print(f"  ⚠️  Playwright error: {e}")
        return ""


def extract_fields_with_llm(page_text: str, scheme_name: str) -> dict:
    """Use LLM to extract structured scheme fields from page text."""
    if not page_text or len(page_text) < 150:
        return {}

    llm    = get_llm()
    prompt = f"""Extract information about the government scheme "{scheme_name}" from this webpage text.

Webpage text:
\"\"\"
{page_text[:7000]}
\"\"\"

Extract these fields:
- description: Overall description of the scheme
- benefits: Benefits provided (financial, material, services)
- eligibility: Who can apply — age, income, caste, occupation, state
- documents_required: List of required documents (semicolon-separated)
- application_process: Step-by-step how to apply (format as "Step 1: ... Step 2: ...")

Rules:
- Copy text closely from the page. Do not invent anything.
- If a field is truly not present, use "Not available".
- Reply ONLY with a valid JSON object, no markdown, no explanation.

JSON:"""

    try:
        response = llm.invoke(prompt)
        raw      = response.content.strip()
        raw      = re.sub(r'^```(?:json)?\s*', '', raw)
        raw      = re.sub(r'\s*```$', '', raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  ⚠️  LLM parse error: {e}")
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


def get_missing_schemes(db_path: str) -> list:
    """Read ChromaDB SQLite to find schemes with empty/missing fields."""
    sqlite_path = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        print(f"❌ ChromaDB not found at: {sqlite_path}")
        return []

    conn = sqlite3.connect(sqlite_path)
    cur  = conn.cursor()

    # Find schemes where benefits OR eligibility is "Not found"
    cur.execute("""
        SELECT string_value FROM embedding_fulltext_search
        WHERE string_value LIKE '%benefits : Not found%'
           OR string_value LIKE '%eligibility : Not found%'
    """)
    rows = cur.fetchall()
    conn.close()

    missing = []
    seen    = set()

    for (text,) in rows:
        name_m  = re.search(r'Scheme name\s*:(.*)', text)
        link_m  = re.search(r'Link\s*:(.*)',        text)
        cat_m   = re.search(r'category\s*:(.*)',    text)
        state_m = re.search(r'state\s*:(.*)',       text)

        if not (name_m and link_m):
            continue

        name = name_m.group(1).strip()
        url  = link_m.group(1).strip()

        # Skip duplicates
        if url in seen:
            continue
        seen.add(url)

        missing.append({
            "name":     name,
            "url":      url,
            "category": cat_m.group(1).strip()   if cat_m   else "",
            "state":    state_m.group(1).strip()  if state_m else "Gujarat",
        })

    return missing


def update_chromadb(vector_db, scheme_name: str, new_doc_text: str):
    """Delete old embedding and add updated one."""
    try:
        results = vector_db._collection.get(
            where_document={"$contains": scheme_name[:40]}
        )
        if results and results.get("ids"):
            vector_db._collection.delete(ids=results["ids"])
            print(f"  🗑️  Deleted {len(results['ids'])} old embedding(s)")

        vector_db.add_texts([new_doc_text])
        print(f"  ✅ Added updated embedding")

    except Exception as e:
        print(f"  ⚠️  ChromaDB update error: {e}")
        try:
            vector_db.add_texts([new_doc_text])
            print(f"  ✅ Added (fallback)")
        except Exception as e2:
            print(f"  ❌ Failed completely: {e2}")


def main():
    """Main function — called by scheduler.py at 3 AM or run manually."""
    print("\n🚀 Yojana AI — Re-scraper for missing scheme details")
    print("=" * 60)

    # ── Load ChromaDB ────────────────────────────────────────────────────────
    print("⏳ Loading ChromaDB and embedding model...")
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )
    print("✅ ChromaDB loaded\n")

    # ── Find schemes with missing data ───────────────────────────────────────
    missing = get_missing_schemes(VECTOR_DB_PATH)
    print(f"📋 Found {len(missing)} schemes with missing details\n")

    if not missing:
        print("🎉 Nothing to fix! All schemes have complete data.")
        return

    success = 0
    failed  = []

    for i, scheme in enumerate(missing, 1):
        name = scheme["name"]
        url  = scheme["url"]

        print(f"[{i}/{len(missing)}] {name}")
        print(f"  🌐 {url}")

        # ── Fetch page ───────────────────────────────────────────────────────
        page_text = ""
        for attempt in range(1, MAX_RETRIES + 1):
            page_text = fetch_with_playwright(url)
            if len(page_text) > 200:
                break
            print(f"  ⚠️  Attempt {attempt} got too little content, retrying...")
            time.sleep(3)

        if len(page_text) < 200:
            print(f"  ❌ Could not fetch page after {MAX_RETRIES} attempts — skipping\n")
            failed.append(name)
            continue

        # ── Extract fields with LLM ──────────────────────────────────────────
        print(f"  📄 Got {len(page_text)} chars — extracting fields with LLM...")
        fields = extract_fields_with_llm(page_text, name)

        if not fields or all(
            v in ("Not available", "", "Not found")
            for v in fields.values()
        ):
            print(f"  ❌ LLM could not extract fields — skipping\n")
            failed.append(name)
            continue

        # ── Build updated document and save to ChromaDB ──────────────────────
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

        print(f"  💾 Updating ChromaDB...")
        update_chromadb(vector_db, name, new_doc)
        success += 1

        print(f"  ✅ Done\n")
        time.sleep(BATCH_DELAY)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"✅ Successfully updated: {success}/{len(missing)} schemes")
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for f in failed:
            print(f"   - {f}")
    print("\n🎉 Re-scraping complete!")


if __name__ == "__main__":
    main()