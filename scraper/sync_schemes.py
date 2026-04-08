"""
sync_schemes.py
════════════════════════════════════════════════════════════════
Automatic sync script for Yojana AI — run via cron job.

What it does (in order):
  1. Scrapes current list of Gujarat schemes from myscheme.gov.in
  2. Validates scrape got enough schemes (Fix 1)
  3. Compares with what is already in ChromaDB
  4. NEW schemes    → scrapes details + adds to ChromaDB + CSVs
  5. MISSING schemes → tracks in missing_tracker.json
                       only deletes after missing 3 runs in a row (Fix 4)
  6. Writes sync log to logs/sync_log.txt

Fixes applied:
  Fix 1 — Validate minimum scrape count before syncing
  Fix 2 — MAX_RETRIES increased 3 → 5
  Fix 3 — API_WAIT increased 20 → 30 seconds
  Fix 4 — Grace period: delete only after missing 3 consecutive runs

Grace period tracker:
  logs/missing_tracker.json  ← auto created, tracks missing counts per slug

Run manually to test:
    python sync_schemes.py
════════════════════════════════════════════════════════════════
"""

import os
import re
import csv
import sys
import json
import time
import sqlite3
import logging
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# Ensure repo root is on PYTHONPATH
import sys
REPO_ROOT_FOR_AUTH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT_FOR_AUTH not in sys.path:
    sys.path.insert(0, REPO_ROOT_FOR_AUTH)

from utils.notifier import broadcast_new_schemes
from database.db import SessionLocal, engine
from database.models import Scheme
from rag.llm import get_vector_db, get_embedding_model
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VECTOR_DB_PATH  = os.path.join(PROJECT_ROOT, "vector_db")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "logs")
RAW_CSV         = os.path.join(PROJECT_ROOT, "data", "raw", "gujarat_schemes.csv")
PROCESSED_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "scraped_schemes.csv")

# ── Fix 4: How many consecutive runs before deleting ─────────────────────────
GRACE_PERIOD    = 3

BASE_URL        = "https://www.myscheme.gov.in/search/state/Gujarat"
SCHEME_BASE     = "https://www.myscheme.gov.in/schemes/"
STATE_NAME      = "Gujarat"
PAGE_SIZE       = 10
NAV_TIMEOUT     = 60_000

# ── Fix 3: Increased from 20 → 30 ────────────────────────────────────────────
API_WAIT        = 30

# ── Fix 2: Increased from 3 → 5 ──────────────────────────────────────────────
MAX_RETRIES     = 5

BATCH_DELAY     = 2
DETAIL_SECTIONS = ["Details", "Benefits", "Eligibility",
                   "Application Process", "Documents Required"]

# ── Fix 1: Minimum scrape threshold (95% of 643) ─────────────────────────────
SCRAPE_MIN_THRESHOLD = 610

# CSV column names
RAW_CSV_FIELDS       = ["scheme_name", "scheme_link", "state", "category", "description"]
PROCESSED_CSV_FIELDS = ["scheme_name", "scheme_link", "state", "category",
                        "details", "benefits", "eligibility",
                        "application_process", "documents_required", "error"]

# ─────────────────────────────────────────────────────────────────────────────


# ── Logging setup ─────────────────────────────────────────────────────────────

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
log_file = os.path.join(LOGS_DIR, "sync_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sync")

# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 4 — Missing tracker helpers (Database version)
# ══════════════════════════════════════════════════════════════════════════════

def update_missing_tracker_db(session, removed_slugs: set, live_slugs: set) -> tuple:
    """
    Update tracker counts in the database for this run.
    """
    confirmed_delete = set()
    still_waiting    = set()

    for slug in removed_slugs:
        # Search for scheme by slug in its application_link
        scheme = session.query(Scheme).filter(Scheme.application_link.contains(slug)).first()
        if not scheme: continue
        
        scheme.missing_count += 1
        if scheme.missing_count >= GRACE_PERIOD:
            confirmed_delete.add(slug)
            log.info(f"   🚨 '{slug}' missing {scheme.missing_count}/{GRACE_PERIOD} runs → confirmed delete")
        else:
            still_waiting.add(slug)
            log.info(f"   ⏳ '{slug}' missing {scheme.missing_count}/{GRACE_PERIOD} runs → waiting...")

    # Reset count for slugs that REAPPEARED
    # (Any scheme with missing_count > 0 that is NOT in the removed_slugs list)
    reappeared_schemes = session.query(Scheme).filter(Scheme.missing_count > 0).all()
    
    for scheme in reappeared_schemes:
        match = False
        for s in live_slugs:
            if s in scheme.application_link:
                match = True
                break
        
        if match:
            log.info(f"   ✅ '{scheme.scheme_name}' reappeared → reset missing count")
            scheme.missing_count = 0

    session.commit()
    return confirmed_delete, still_waiting


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Scrape current scheme list from myscheme.gov.in
# ══════════════════════════════════════════════════════════════════════════════

def _extract_schemes_from_api(data: dict) -> list:
    schemes = []
    try:
        items = data["data"]["hits"]["items"]
    except (KeyError, TypeError):
        return schemes

    for item in items:
        fields = item.get("fields", {})
        name   = (fields.get("schemeName") or "").strip()
        slug   = (fields.get("slug")       or "").strip()
        if not name:
            continue

        raw_state = fields.get("state") or fields.get("stateName") or []
        if isinstance(raw_state, list):
            state_val = ", ".join(raw_state).strip() or STATE_NAME
        else:
            state_val = (raw_state or STATE_NAME).strip()

        schemes.append({
            "scheme_name": name,
            "scheme_link": f"{SCHEME_BASE}{slug}" if slug else "",
            "slug":        slug,
            "state":       state_val,
            "category":    ", ".join(fields.get("schemeCategory") or []),
            "description": (fields.get("briefDescription") or "").strip(),
        })
    return schemes


def _wait_for(holder: dict, key: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if holder.get(key):
            return True
        time.sleep(0.05)
    return False


def scrape_live_scheme_list() -> dict:
    """Returns { slug: scheme_dict } for every Gujarat scheme on website."""
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    log.info("🌐 Scraping live scheme list from myscheme.gov.in …")
    all_schemes  = {}
    failed_pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        ctx   = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        bpage = ctx.new_page()
        state = {"target_offset": 0, "api_body": None, "api_done": False}

        def handle_route(route):
            url = route.request.url
            if re.search(r"api\.myscheme", url) and "schemes" in url:
                parsed     = urlparse(url)
                qs         = parse_qs(parsed.query, keep_blank_values=True)
                qs["from"] = [str(state["target_offset"])]
                new_qs     = urlencode({k: v[0] for k, v in qs.items()})
                route.continue_(url=urlunparse(parsed._replace(query=new_qs)))
            else:
                route.continue_()

        def on_response(resp):
            if re.search(r"api\.myscheme", resp.url) and "schemes" in resp.url:
                try:
                    state["api_body"] = resp.json()
                    state["api_done"] = True
                except Exception:
                    pass

        bpage.route("**/*", handle_route)
        bpage.on("response", on_response)

        # ── Page 1 (with retry) ───────────────────────────────────────────
        page1_ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            state.update(target_offset=0, api_body=None, api_done=False)
            try:
                bpage.goto(BASE_URL, wait_until="networkidle", timeout=NAV_TIMEOUT)
            except PWTimeout:
                try:
                    bpage.goto(BASE_URL, timeout=NAV_TIMEOUT)
                except Exception:
                    pass

            if _wait_for(state, "api_done", API_WAIT) and state["api_body"]:
                page1_ok = True
                break

            wait = 2 ** attempt
            log.warning(f"   ⚠️  Page 1 attempt {attempt}/{MAX_RETRIES} failed — retrying in {wait}s")
            time.sleep(wait)

        if not page1_ok:
            log.error("❌ No API response on page 1 after all retries. Aborting.")
            browser.close()
            return {}

        total       = state["api_body"].get("data", {}).get("summary", {}).get("total", 0)
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        log.info(f"   Total schemes on website: {total}  ({total_pages} pages)")

        for s in _extract_schemes_from_api(state["api_body"]):
            all_schemes[s["slug"]] = s

        # ── Pages 2..N ────────────────────────────────────────────────────
        for page_no in range(2, total_pages + 1):
            offset  = (page_no - 1) * PAGE_SIZE
            state.update(target_offset=offset, api_body=None, api_done=False)
            page_ok = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    bpage.reload(wait_until="networkidle", timeout=NAV_TIMEOUT)
                except PWTimeout:
                    try:
                        bpage.reload(timeout=NAV_TIMEOUT)
                    except Exception:
                        pass

                _wait_for(state, "api_done", API_WAIT)

                if state["api_done"] and state["api_body"]:
                    batch = _extract_schemes_from_api(state["api_body"])
                    if batch:
                        for s in batch:
                            all_schemes[s["slug"]] = s
                        page_ok = True
                        break

                wait = 2 ** attempt
                log.warning(f"   ⚠️  Page {page_no} attempt {attempt}/{MAX_RETRIES} — retrying in {wait}s")
                time.sleep(wait)

            if not page_ok:
                failed_pages.append(page_no)

            time.sleep(0.5)

        browser.close()

    if failed_pages:
        log.warning(f"   ⚠️  {len(failed_pages)} pages failed: {failed_pages}")

    log.info(f"   ✅ Scraped {len(all_schemes)} unique schemes from website")
    return all_schemes


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Read what is already in ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

def get_db_scheme_slugs():
    """Returns { slug: doc_text } for every scheme in the Supabase relational DB."""
    session = SessionLocal()
    try:
        schemes = session.query(Scheme).all()
        db_schemes = {}
        for s in schemes:
            if not s.application_link: continue
            slug = s.application_link.rstrip('/').split("/")[-1]
            # Use description/details as proxy for doc_text
            db_schemes[slug] = s.description or ""
        
        log.info(f"   📦 Supabase DB currently has {len(db_schemes)} schemes")
        return db_schemes
    except Exception as e:
        log.error(f"❌ Error reading Supabase DB: {e}")
        return {}
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Scrape detail page for a single scheme
# ══════════════════════════════════════════════════════════════════════════════

TRAILING_NOISE = re.compile(
    r"(Frequently Asked Questions.*|Sources And References.*|Feedback.*|Was this helpful.*)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_sections_js(page) -> dict:
    return page.evaluate(
        """(sectionNames) => {
            const allH3 = Array.from(document.querySelectorAll('h3'));
            const sectionH3s = allH3.filter(h => {
                const cls = h.className || '';
                return cls.includes('font-semibold') || cls.includes('text-darkblue-900');
            });
            if (sectionH3s.length === 0) return {};
            const results = {};
            sectionH3s.forEach((h3, idx) => {
                const label = (h3.innerText || '').trim();
                if (!sectionNames.includes(label)) return;
                const nextH3 = sectionH3s[idx + 1];
                const parts  = [];
                let node = h3.nextElementSibling;
                while (node) {
                    if (node === nextH3) break;
                    const nc = node.className || '';
                    if (node.tagName === 'H3' &&
                        (nc.includes('font-semibold') || nc.includes('text-darkblue-900'))) break;
                    const text = (node.innerText || '').trim();
                    if (text) parts.push(text);
                    node = node.nextElementSibling;
                }
                results[label] = parts.join('\\n').trim();
            });
            return results;
        }""",
        DETAIL_SECTIONS,
    )


def scrape_scheme_detail(bpage, scheme: dict) -> dict:
    from playwright.sync_api import TimeoutError as PWTimeout

    url    = scheme["scheme_link"]
    name   = scheme["scheme_name"]
    result = {k: "Not found" for k in
              ["details", "benefits", "eligibility",
               "application_process", "documents_required"]}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bpage.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT)
            bpage.wait_for_timeout(2000)

            is_404 = bpage.evaluate("""() => {
                const b = document.body?.innerText || '';
                return b.includes('Page not found') && !b.includes('Details');
            }""")
            if is_404:
                log.warning(f"   404 for {url}")
                return result

            try:
                bpage.wait_for_selector(
                    "h3[class*='font-semibold'], h3[class*='text-darkblue-900']",
                    timeout=12000,
                )
            except PWTimeout:
                pass

            bpage.wait_for_timeout(1500)
            sections = _extract_sections_js(bpage)

            for sec in DETAIL_SECTIONS:
                text = sections.get(sec, "Not found")
                if text:
                    text = TRAILING_NOISE.sub("", text).strip()
                key = sec.lower().replace(" ", "_")
                result[key] = text or "Not found"

            return result

        except PWTimeout:
            log.warning(f"   Timeout on {url} (attempt {attempt})")
            time.sleep(2 ** attempt)
        except Exception as e:
            log.warning(f"   Error scraping {name}: {e}")
            return result

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(filepath: str) -> list:
    if not os.path.exists(filepath):
        log.warning(f"   CSV not found: {filepath} — will create fresh.")
        return []
    rows = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def write_csv(filepath: str, fields: list, rows: list):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"   💾 Saved {len(rows)} rows → {os.path.basename(filepath)}")
    except PermissionError:
        log.warning(f"   ⚠️  CSV is open somewhere — close it: {os.path.basename(filepath)}")


def add_to_raw_csv(scheme: dict):
    rows = read_csv(RAW_CSV)
    if any(r.get("scheme_link") == scheme["scheme_link"] for r in rows):
        log.info(f"   ⚠️  Already in raw CSV: {scheme['scheme_name']}")
        return
    rows.append({
        "scheme_name": scheme["scheme_name"],
        "scheme_link": scheme["scheme_link"],
        "state":       scheme.get("state", STATE_NAME),
        "category":    scheme.get("category", ""),
        "description": scheme.get("description", ""),
    })
    write_csv(RAW_CSV, RAW_CSV_FIELDS, rows)


def add_to_processed_csv(scheme: dict, details: dict):
    rows = read_csv(PROCESSED_CSV)
    if any(r.get("scheme_link") == scheme["scheme_link"] for r in rows):
        log.info(f"   ⚠️  Already in processed CSV: {scheme['scheme_name']}")
        return
    rows.append({
        "scheme_name":         scheme["scheme_name"],
        "scheme_link":         scheme["scheme_link"],
        "state":               scheme.get("state", STATE_NAME),
        "category":            scheme.get("category", ""),
        "details":             details.get("details", "Not found"),
        "benefits":            details.get("benefits", "Not found"),
        "eligibility":         details.get("eligibility", "Not found"),
        "application_process": details.get("application_process", "Not found"),
        "documents_required":  details.get("documents_required", "Not found"),
        "error":               "",
    })
    write_csv(PROCESSED_CSV, PROCESSED_CSV_FIELDS, rows)


def delete_from_raw_csv(slug: str, scheme_name: str):
    rows   = read_csv(RAW_CSV)
    before = len(rows)
    rows   = [r for r in rows if f"/schemes/{slug}" not in r.get("scheme_link", "")]
    if len(rows) < before:
        write_csv(RAW_CSV, RAW_CSV_FIELDS, rows)
        log.info(f"   🗑️  Deleted from raw CSV: {scheme_name}")
    else:
        log.warning(f"   ⚠️  Not found in raw CSV: {scheme_name}")


def delete_from_processed_csv(slug: str, scheme_name: str):
    rows   = read_csv(PROCESSED_CSV)
    before = len(rows)
    rows   = [r for r in rows if f"/schemes/{slug}" not in r.get("scheme_link", "")]
    if len(rows) < before:
        write_csv(PROCESSED_CSV, PROCESSED_CSV_FIELDS, rows)
        log.info(f"   🗑️  Deleted from processed CSV: {scheme_name}")
    else:
        log.warning(f"   ⚠️  Not found in processed CSV: {scheme_name}")


# ══════════════════════════════════════════════════════════════════════════════
#  CHROMADB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_doc_text(scheme: dict, details: dict) -> str:
    return (
        f"\n    Scheme name :{scheme['scheme_name']}\n"
        f"    Description :{details.get('details', 'Not found')}\n"
        f"    category : {scheme.get('category', '')}\n"
        f"    benefits : {details.get('benefits', 'Not found')}\n"
        f"    eligibility : {details.get('eligibility', 'Not found')}\n"
        f"    application_process : {details.get('application_process', 'Not found')}\n"
        f"    required_documents : {details.get('documents_required', 'Not found')}\n"
        f"    state : {scheme.get('state', STATE_NAME)}\n"
        f"    Link : {scheme.get('scheme_link', '')}\n\n    "
    )


def add_to_cloud_db(scheme: dict, details: dict):
    """Saves scheme to Relational DB and Vector DB on Supabase."""
    session = SessionLocal()
    try:
        # 1. Add/Update in Relational DB
        existing = session.query(Scheme).filter(Scheme.application_link == scheme['scheme_link']).first()
        if existing:
            existing.description = str(details.get('details', ''))
            existing.benefits = str(details.get('benefits', ''))
            existing.eligibility = str(details.get('eligibility', ''))
            existing.application_process = str(details.get('application_process', ''))
            existing.documents_required = str(details.get('documents_required', ''))
            existing.missing_count = 0
            existing.last_updated_at = datetime.utcnow()
        else:
            new_scheme = Scheme(
                scheme_name=scheme['scheme_name'],
                application_link=scheme['scheme_link'],
                state=scheme.get('state', STATE_NAME),
                category=scheme.get('category', ''),
                description=str(details.get('details', '')),
                benefits=str(details.get('benefits', '')),
                eligibility=str(details.get('eligibility', '')),
                application_process=str(details.get('application_process', '')),
                documents_required=str(details.get('documents_required', '')),
                missing_count=0
            )
            session.add(new_scheme)
        
        session.commit()
        log.info(f"   ✅ Relational DB updated: {scheme['scheme_name']}")

        # 2. Add to Vector DB (Cloud Index)
        try:
            vector_db = get_vector_db()
            if vector_db:
                doc_text = _build_doc_text(scheme, details)
                vector_db.add_texts([doc_text])
                log.info(f"   ✅ Vector Index updated: {scheme['scheme_name']}")
        except Exception as vec_e:
            log.warning(f"   ⚠️ Vector Sync failed for {scheme['scheme_name']}: {vec_e}")
            log.warning("      (Continuing with relational data only)")

    except Exception as e:
        log.error(f"   ❌ FATAL DB error for {scheme['scheme_name']}: {e}")
        session.rollback()
    finally:
        session.close()


def delete_from_cloud_db(slug: str, scheme_name: str):
    """Deletes scheme from Relational DB and Vector Index."""
    session = SessionLocal()
    try:
        # 1. Delete from Relational DB
        session.query(Scheme).filter(Scheme.application_link.contains(slug)).delete(synchronize_session=False)
        session.commit()
        log.info(f"   🗑️  Deleted from Relational DB: {scheme_name}")

        # 2. Delete from Vector Index
        vector_db = get_vector_db()
        if vector_db:
            from supabase.client import create_client
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            supabase_client = create_client(supabase_url, supabase_key)
            
            # Using direct SQL delete via RPC if possible or via the client
            # Langchain's SupabaseVectorStore delete is IDs-based, so we filter by metadata
            supabase_client.table("documents").delete().filter("content", "ilike", f"%{slug}%").execute()
            log.info(f"   🗑️  Deleted from Vector Index: {scheme_name}")
    except Exception as e:
        log.error(f"   ❌ Failed to delete {scheme_name}: {e}")
        session.rollback()
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SYNC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_sync():
    start_time = datetime.now()
    log.info("=" * 65)
    log.info(f"🚀 Yojana AI — Cloud Sync started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    # ── Step 1: Scrape live scheme list ──────────────────────────────────────
    live_schemes = scrape_live_scheme_list()
    if not live_schemes:
        log.error("❌ Could not fetch live scheme list. Aborting sync.")
        return

    # ── Fix 1: Validate scrape count ─────────────────────────────────────────
    scraped_count = len(live_schemes)
    if scraped_count < SCRAPE_MIN_THRESHOLD:
        log.warning("=" * 65)
        log.warning(f"⚠️  SCRAPE INCOMPLETE — only got {scraped_count} schemes!")
        log.warning(f"   Skipping sync to avoid false deletes.")
        log.warning("=" * 65)
        return

    log.info(f"   ✅ Scrape validated: {scraped_count} schemes")

    # ── Step 2: Get DB scheme slugs (from Supabase) ──────────────────────────
    db_slugs    = get_db_scheme_slugs()
    live_slugs  = set(live_schemes.keys())
    db_slug_set = set(db_slugs.keys())

    new_slugs     = live_slugs - db_slug_set
    removed_slugs = db_slug_set - live_slugs

    log.info(f"\n📊 Sync summary:")
    log.info(f"   Live on website : {len(live_slugs)}")
    log.info(f"   In Cloud DB     : {len(db_slug_set)}")
    log.info(f"   ➕ New to add   : {len(new_slugs)}")
    log.info(f"   ➖ Missing       : {len(removed_slugs)}")

    # ── Fix 4: Grace period — check database tracker ──────────────────────────
    session = SessionLocal()
    confirmed_delete, still_waiting = update_missing_tracker_db(
        session, removed_slugs, live_slugs
    )

    log.info(f"\n⏳ Grace period status:")
    log.info(f"   🚨 Confirmed delete (missing {GRACE_PERIOD}x) : {len(confirmed_delete)}")
    log.info(f"   ⏳ Still waiting (missing < {GRACE_PERIOD}x)  : {len(still_waiting)}")

    # ── Step 3: Delete only CONFIRMED removed schemes ────────────────────────
    if confirmed_delete:
        log.info(f"\n🗑️  Deleting {len(confirmed_delete)} confirmed removed scheme(s) …")
        for slug in confirmed_delete:
            scheme_name = slug.replace("-", " ").title()
            delete_from_cloud_db(slug, scheme_name)
    else:
        log.info("\n✅ No confirmed deletions this run.")

    # ── Step 4: Add new schemes ───────────────────────────────────────────────
    if new_slugs:
        log.info(f"\n➕ Adding {len(new_slugs)} new scheme(s) …")

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            bpage = browser.new_page()

            added  = 0
            failed = []

            for i, slug in enumerate(sorted(new_slugs), 1):
                scheme = live_schemes[slug]
                log.info(f"\n  [{i}/{len(new_slugs)}] {scheme['scheme_name']}")

                if not scheme.get("scheme_link"):
                    log.warning("   ⚠️  No link — skipping")
                    continue

                details = scrape_scheme_detail(bpage, scheme)
                if details:
                    add_to_cloud_db(scheme, details)
                    added += 1
                else:
                    failed.append(scheme["scheme_name"])
                
                time.sleep(BATCH_DELAY)

            browser.close()

        log.info(f"\n   ✅ Added: {added}  |  ❌ Failed: {len(failed)}")
        
        if added > 0:
            new_names = [live_schemes[s]["scheme_name"] for s in sorted(new_slugs) if s in live_schemes]
            broadcast_new_schemes(new_names)
    else:
        log.info("\n✅ No new schemes to add.")

    session.close()
    
    # ── Done ─────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info(f"🎉 Cloud Sync complete")
    log.info("=" * 65)


if __name__ == "__main__":
    run_sync()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_sync()