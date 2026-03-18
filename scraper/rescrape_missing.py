"""
rescrape_missing.py
────────────────────────────────────────────────────────────
Run this ONCE to fix all schemes in your ChromaDB that have
"Not found" for benefits/eligibility/etc.

It uses Playwright to fetch each myscheme.gov.in page (JS-rendered),
extracts the real data with the LLM, then UPDATES the ChromaDB entries.

Usage:
    python rescrape_missing.py

Requirements:
    pip install playwright langchain-chroma langchain-huggingface
    pip install langchain-mistralai sentence-transformers
    playwright install chromium
"""

import os, re, json, time, sqlite3
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

# ── Config ────────────────────────────────────────────────────────────────────
VECTOR_DB_PATH = "vector_db"      # path to your ChromaDB folder
BATCH_DELAY    = 2                 # seconds between Playwright fetches (be polite)
MAX_RETRIES    = 2
# ─────────────────────────────────────────────────────────────────────────────

llm = ChatMistralAI(model="mistral-small-latest", temperature=0.1)


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
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
                java_script_enabled=True,
            )
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            # Wait for content sections to appear
            try:
                page.wait_for_selector("section, .scheme-detail, main, article", timeout=8000)
            except Exception:
                pass
            page.wait_for_timeout(4000)
            text = page.evaluate("document.body.innerText")
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
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  ⚠️  LLM parse error: {e}")
        return {}


def build_document_text(name, description, category, benefits, eligibility,
                         application_process, documents_required, state, link) -> str:
    """Build the document string in the same format as your original scraper."""
    return f"""
    Scheme name :{name}
    Description :{description}
    category : {category}
    benefits : {benefits}
    eligibility : {eligibility}
    application_process : {application_process}
    required_documents : {documents_required}
    state : {state}
    Link : {link}

    """


def get_missing_schemes(db_path: str) -> list:
    """Read ChromaDB SQLite to find schemes with empty fields."""
    sqlite_path = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        print(f"❌ ChromaDB not found at: {sqlite_path}")
        return []

    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("SELECT string_value FROM embedding_fulltext_search WHERE string_value LIKE '%benefits : Not found%'")
    rows = cur.fetchall()
    conn.close()

    missing = []
    for r in rows:
        text = r[0]
        name_m    = re.search(r'Scheme name\s*:(.*)', text)
        link_m    = re.search(r'Link\s*:(.*)', text)
        cat_m     = re.search(r'category\s*:(.*)', text)
        state_m   = re.search(r'state\s*:(.*)', text)
        if name_m and link_m:
            missing.append({
                "name":     name_m.group(1).strip(),
                "url":      link_m.group(1).strip(),
                "category": cat_m.group(1).strip() if cat_m else "",
                "state":    state_m.group(1).strip() if state_m else "Gujarat",
            })
    return missing


def update_chromadb(vector_db: Chroma, scheme_name: str, new_doc_text: str):
    """
    Delete the old embedding for this scheme and add the updated one.
    ChromaDB doesn't support in-place update, so we delete + re-add.
    """
    try:
        # Find existing doc by similarity search (name is unique enough)
        results = vector_db._collection.get(
            where_document={"$contains": scheme_name[:40]}
        )
        if results and results.get("ids"):
            ids_to_delete = results["ids"]
            vector_db._collection.delete(ids=ids_to_delete)
            print(f"  🗑️  Deleted {len(ids_to_delete)} old embedding(s)")

        # Add updated document
        vector_db.add_texts([new_doc_text])
        print(f"  ✅ Added updated embedding")
    except Exception as e:
        print(f"  ⚠️  ChromaDB update error: {e}")
        # Fallback: just add without deleting (creates a duplicate, but still searchable)
        try:
            vector_db.add_texts([new_doc_text])
            print(f"  ✅ Added (fallback, may be duplicate)")
        except Exception as e2:
            print(f"  ❌ Failed completely: {e2}")


def main():
    print("🚀 Yojana AI — Re-scraper for missing scheme data")
    print("=" * 60)

    # Load ChromaDB
    print("⏳ Loading ChromaDB and embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    print("✅ ChromaDB loaded\n")

    # Get missing schemes
    missing = get_missing_schemes(VECTOR_DB_PATH)
    print(f"📋 Found {len(missing)} schemes with missing data\n")

    if not missing:
        print("🎉 Nothing to fix! All schemes have data.")
        return

    success = 0
    failed  = []

    for i, scheme in enumerate(missing, 1):
        name = scheme["name"]
        url  = scheme["url"]
        print(f"[{i}/{len(missing)}] {name}")
        print(f"  🌐 {url}")

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

        print(f"  📄 Got {len(page_text)} chars — extracting fields...")
        fields = extract_fields_with_llm(page_text, name)

        if not fields or all(v in ("Not available", "", "Not found") for v in fields.values()):
            print(f"  ❌ LLM could not extract fields — skipping\n")
            failed.append(name)
            continue

        # Build updated document
        new_doc = build_document_text(
            name              = name,
            description       = fields.get("description", "Not available"),
            category          = scheme["category"],
            benefits          = fields.get("benefits", "Not available"),
            eligibility       = fields.get("eligibility", "Not available"),
            application_process = fields.get("application_process", "Not available"),
            documents_required  = fields.get("documents_required", "Not available"),
            state             = scheme["state"],
            link              = url,
        )

        print(f"  💾 Updating ChromaDB...")
        update_chromadb(vector_db, name, new_doc)
        success += 1

        print(f"  ✅ Done\n")
        time.sleep(BATCH_DELAY)

    # Summary
    print("=" * 60)
    print(f"✅ Successfully updated: {success}/{len(missing)} schemes")
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for f in failed:
            print(f"   - {f}")
    print("\n🎉 Re-scraping complete! Restart your Flask app to use updated data.")


if __name__ == "__main__":
    main()
