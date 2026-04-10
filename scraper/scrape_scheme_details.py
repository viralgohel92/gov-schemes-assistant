"""
Gujarat Schemes Detail Scraper  (FINAL)
-----------------------------------------
Reads gujarat_schemes.csv and scrapes Details, Benefits, Eligibility,
Application Process, Documents Required from each scheme page.

Supports BOTH h3 class variants found on myscheme.gov.in:
  - class="...font-semibold..."       (most schemes)
  - class="...text-darkblue-900..."   (some schemes)

Output: data/processed/scraped_schemes.csv
"""

import os, csv, time, re
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

#    Paths                                                                       
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE   = os.path.join(DATA_DIR, "raw", "gujarat_schemes.csv")
OUTPUT_FILE  = os.path.join(DATA_DIR, "processed", "scraped_schemes.csv")

SCHEME_BASE  = "https://www.myscheme.gov.in/schemes/"

SECTIONS = [
    "Details",
    "Benefits",
    "Eligibility",
    "Application Process",
    "Documents Required",
]

def col(label: str) -> str:
    return label.lower().replace(" ", "_")

TRAILING_NOISE = re.compile(
    r"(Frequently Asked Questions.*|Sources And References.*|Feedback.*|Was this helpful.*)",
    re.DOTALL | re.IGNORECASE
)


def is_404(page) -> bool:
    return page.evaluate("""() => {
        const body = document.body?.innerText || '';
        return body.includes('Page not found') && !body.includes('Details');
    }""")


def name_to_slug(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def try_load_page(page, name: str, url: str) -> tuple:
    candidates = [url]
    name_no_suffix = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
    candidates.append(f"{SCHEME_BASE}{name_to_slug(name)}")
    candidates.append(f"{SCHEME_BASE}{name_to_slug(name_no_suffix)}")

    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    for candidate_url in unique:
        try:
            page.goto(candidate_url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)
            if not is_404(page):
                return True, candidate_url
            else:
                print(f"    404: {candidate_url}")
        except PWTimeout:
            print(f"    timeout: {candidate_url}")

    return False, url


#    FIXED: extract sections supporting BOTH h3 class variants                  
def extract_sections(page) -> dict:
    return page.evaluate("""(sectionNames) => {

        //    Selector covers both class variants                                
        // Variant A: font-semibold  (original working schemes)
        // Variant B: text-darkblue-900  (schemes that were failing)
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

            // Walk siblings until the next section heading
            let node = h3.nextElementSibling;
            while (node) {
                if (node === nextH3) break;
                const nodeCls = node.className || '';
                if (node.tagName === 'H3' &&
                    (nodeCls.includes('font-semibold') || nodeCls.includes('text-darkblue-900'))) break;

                const text = (node.innerText || '').trim();
                if (text) parts.push(text);
                node = node.nextElementSibling;
            }

            // If siblings gave nothing, try searching inside the whole page
            // by finding the content between this heading and the next
            if (parts.length === 0) {
                const parent = h3.parentElement;
                if (parent) {
                    let sibling = h3.nextElementSibling;
                    while (sibling) {
                        const sc = sibling.className || '';
                        if (sibling.tagName === 'H3' &&
                            (sc.includes('font-semibold') || sc.includes('text-darkblue-900'))) break;
                        const t = (sibling.innerText || '').trim();
                        if (t) parts.push(t);
                        sibling = sibling.nextElementSibling;
                    }
                }
            }

            results[label] = parts.join('\\n').trim();
        });

        return results;
    }""", SECTIONS)


def scrape_scheme(page, name: str, url: str, state: str, category: str) -> dict:
    print(f"  {name[:70]}")
    result = {
        "scheme_name": name,
        "scheme_link": url,
        "state":       state,
        "category":    category,
        "error":       "",
    }
    for s in SECTIONS:
        result[col(s)] = "Not found"

    success, final_url = try_load_page(page, name, url)

    if not success:
        print(f"    X All URL attempts failed")
        result["error"] = "404 - not found on myscheme.gov.in"
        return result

    if final_url != url:
        print(f"    Working URL: {final_url}")
        result["scheme_link"] = final_url

    # Wait for either known h3 class to appear
    try:
        page.wait_for_selector(
            "h3[class*='font-semibold'], h3[class*='text-darkblue-900']",
            timeout=15000
        )
    except PWTimeout:
        pass

    page.wait_for_timeout(1500)
    sections = extract_sections(page)

    for s in SECTIONS:
        text = sections.get(s, "Not found")
        if text:
            text = TRAILING_NOISE.sub('', text).strip()
        result[col(s)] = text or "Not found"
        preview = (text or "")[:90].replace('\n', ' ').strip()
        print(f"    [{s:20s}]: {preview}")

    return result


def main():
    schemes = []
    with open(INPUT_FILE, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name     = row.get("scheme_name", "").strip()
            url      = (row.get("scheme_link") or row.get("link") or "").strip()
            state    = row.get("state", "").strip()
            category = row.get("category", "").strip()
            if name and url:
                schemes.append((name, url, state, category))

    print(f"[+] Loaded {len(schemes)} schemes\n")

    fieldnames = (
        ["scheme_name", "scheme_link", "state", "category"]
        + [col(s) for s in SECTIONS]
        + ["error"]
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        bpage = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        ).new_page()

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for i, (name, url, state, category) in enumerate(schemes, 1):
                print(f"\n[{i}/{len(schemes)}]", end=" ")
                try:
                    row = scrape_scheme(bpage, name, url, state, category)
                except Exception as e:
                    print(f"  X {e}")
                    row = {k: "" for k in fieldnames}
                    row.update({"scheme_name": name, "scheme_link": url,
                                "state": state, "category": category, "error": str(e)})

                writer.writerow(row)
                out_f.flush()
                time.sleep(0.8)

        browser.close()

    print(f"\nDone! Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()