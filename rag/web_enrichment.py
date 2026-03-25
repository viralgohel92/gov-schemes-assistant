import re
import json
import requests
from rag.llm import get_llm
from rag.utils import is_missing

def _fetch_page_text_requests(url: str, timeout: int = 8) -> str:
    """Try fetching page text with requests (works for static HTML sites)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<[^>]+>', ' ', html)
        html = re.sub(r'&nbsp;', ' ', html)
        html = re.sub(r'&[a-zA-Z]+;', '', html)
        html = re.sub(r'\s{2,}', ' ', html)
        return html.strip()[:8000]
    except Exception as e:
        print(f"[requests_fetch] Failed: {e}")
        return ""


def _fetch_page_text_playwright(url: str, timeout: int = 20000) -> str:
    """Fetch JS-rendered page text using Playwright (for sites like myscheme.gov.in)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(4000)   # wait for JS to render content
            text = page.evaluate("document.body.innerText")
            browser.close()
            cleaned = re.sub(r'\s{2,}', ' ', text or "")
            return cleaned.strip()[:8000]
    except Exception as e:
        print(f"[playwright_fetch] Failed: {e}")
        return ""


def _fetch_page_text(url: str) -> str:
    """
    Try requests first (fast). If it returns too little content (JS-rendered site),
    fall back to Playwright automatically.
    """
    text = _fetch_page_text_requests(url)
    if len(text) < 300:
        print(f"[fetch] requests got too little content, trying Playwright for {url}")
        text = _fetch_page_text_playwright(url)
    return text


def enrich_scheme_from_web(url: str, scheme_name: str, missing_fields: list) -> dict:
    """
    Fetch the official scheme page and use the LLM to extract only the missing fields.
    Returns a dict of extracted values. Falls back gracefully on any error.
    """
    print(f"🌐 Fetching live data for '{scheme_name}' → {url}")
    page_text = _fetch_page_text(url)
    if not page_text or len(page_text) < 100:
        print(f"[enrich] Could not get usable content from {url}")
        return {}

    fields_desc = {
        "description":         "A brief overall description of what this scheme is about",
        "benefits":            "Benefits provided to beneficiaries (financial amount, subsidies, services, etc.)",
        "eligibility":         "Who is eligible — age, income, caste, occupation, state restrictions",
        "documents_required":  "List of documents needed to apply",
        "application_process": "Step-by-step instructions on how to apply",
    }
    fields_to_extract = {f: fields_desc[f] for f in missing_fields if f in fields_desc}
    fields_json = json.dumps(fields_to_extract, indent=2)

    prompt = f"""You are extracting information about the government scheme "{scheme_name}" from a webpage.

Webpage content:
\"\"\"
{page_text}
\"\"\"

Extract ONLY the following fields from the content above:
{fields_json}

Rules:
- Copy relevant text exactly as found on the page.
- For application_process, number each step clearly: "Step 1: ... Step 2: ..."
- For documents_required, list each document on a new line or separated by semicolons.
- If a field is not present in the content at all, set its value to "Not available".
- Do NOT invent or hallucinate any information.
- Reply ONLY with a valid JSON object using the exact field names as keys. No markdown, no explanation.

JSON:"""

    try:
        response = get_llm().invoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        print(f"✅ Enriched {list(result.keys())} for '{scheme_name}'")
        return result
    except Exception as e:
        print(f"[enrich_scheme] Parse error for '{scheme_name}': {e}")
        return {}


def apply_visit_site_fallback(d: dict) -> dict:
    """
    For any missing fields, fetch live data from the official link.
    Falls back to a clickable link message if fetch fails or page is unreachable.
    """
    link = d.get("official_link", "")
    missing_fields = [
        f for f in ["description", "benefits", "eligibility", "documents_required", "application_process"]
        if is_missing(d.get(f, ""))
    ]
    if not missing_fields:
        return d  # all fields present, nothing to do

    if link and not is_missing(link):
        enriched = enrich_scheme_from_web(link, d.get("scheme_name", ""), missing_fields)
        for f in missing_fields:
            val = enriched.get(f, "")
            d[f] = val if val and not is_missing(val) else f"Not available. 👉 Visit: {link}"
    else:
        for f in missing_fields:
            d[f] = "Not available."
    return d
