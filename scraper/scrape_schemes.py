"""
=============================================================
 myscheme.gov.in — Gujarat Schemes Scraper  (FINAL v7)

 Same route-intercept strategy as v6, but now:
 - Retries timed-out pages up to MAX_RETRIES times
 - After all pages attempted, retries any still-missing pages
 - Longer wait between retries with exponential backoff
 - Saves progress to CSV even if some pages still fail
 - Now includes "state" column in output CSV

RUN:
    python3 scrape_gujarat_schemes.py

OUTPUT:
    gujarat_schemes.csv
=============================================================
"""

# Import module used to write data into CSV files
import csv

# Import regular expressions module (used to search patterns in text/URLs)
import re

# Import time module (used for delays and timing operations)
import time

# Import sys module (used for exiting program and system operations)
import sys

# Import tools for modifying and analyzing URLs
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# Import Playwright tools for browser automation
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout, Route


# Main webpage listing Gujarat schemes
BASE_URL    = "https://www.myscheme.gov.in/search/state/Gujarat"

# Base URL used to build individual scheme links
SCHEME_BASE = "https://www.myscheme.gov.in/schemes/"

# Output CSV filename
OUTPUT_FILE = "data/raw/gujarat_schemes.csv"

# Number of schemes per page on the site
PAGE_SIZE   = 10

# Maximum page load time (milliseconds)
NAV_TIMEOUT = 60000

# Seconds to wait for API response per attempt
API_WAIT    = 20

# Maximum retries for a failed page
MAX_RETRIES = 5

# Base delay between retries (will double each retry attempt)
RETRY_DELAY = 4.0

# Normal delay between successful pages
PAGE_DELAY  = 0.5

# State name to tag all schemes with (matches the URL filter)
STATE_NAME  = "Gujarat"


# Function to extract scheme data from API response
def extract_from_v6(data: dict) -> list:

    # List to store extracted schemes
    schemes = []

    try:
        # Navigate JSON structure to get list of scheme items
        items = data["data"]["hits"]["items"]

    # If JSON structure missing or invalid
    except (KeyError, TypeError):
        return schemes

    # Loop through each scheme item
    for item in items:

        # Get the "fields" section safely
        fields = item.get("fields", {})

        # Extract scheme name
        name   = (fields.get("schemeName") or "").strip()

        # Extract slug (used to create scheme URL)
        slug   = (fields.get("slug") or "").strip()

        # Skip if scheme has no name
        if not name:
            continue

        # Try to get state from API data; fall back to STATE_NAME constant
        state_val = ""
        raw_state = fields.get("state") or fields.get("stateName") or []
        if isinstance(raw_state, list):
            state_val = ", ".join(raw_state).strip()
        elif isinstance(raw_state, str):
            state_val = raw_state.strip()
        if not state_val:
            state_val = STATE_NAME

        # Add scheme info to list
        schemes.append({

            # Scheme name
            "scheme_name": name,

            # Build full scheme URL using slug
            "scheme_link": f"{SCHEME_BASE}{slug}" if slug else "",

            # State the scheme belongs to
            "state":       state_val,

            # Join multiple categories into one string
            "category":    ", ".join(fields.get("schemeCategory") or []),

            # Short scheme description
            "description": (fields.get("briefDescription") or "").strip(),
        })

    # Return list of schemes
    return schemes


# Function that waits until a certain condition becomes true
def wait_for(holder: dict, key: str, timeout: float) -> bool:

    # Calculate deadline time
    deadline = time.time() + timeout

    # Keep checking until timeout
    while time.time() < deadline:

        # If the key exists and has value
        if holder.get(key):
            return True

        # Small sleep to prevent CPU overuse
        time.sleep(0.05)

    # If timeout reached without condition
    return False


# Function to fetch one page of schemes using offset
def fetch_offset(bpage, state, offset: int, page_no: int) -> list | None:
    """
    Fetch one page offset with retries.
    Returns list of schemes, or None if all retries failed.
    """

    # Retry loop
    for attempt in range(1, MAX_RETRIES + 1):

        # Set the desired API offset
        state["target_offset"] = offset

        # Reset stored API response
        state["api_body"]      = None

        # Mark API response as not received yet
        state["api_done"]      = False

        try:
            # Reload the page and wait until network activity stops
            bpage.reload(wait_until="networkidle", timeout=NAV_TIMEOUT)

        # If page loading times out
        except PlaywrightTimeout:
            try:
                # Try reloading again without waiting for network idle
                bpage.reload(timeout=NAV_TIMEOUT)
            except Exception:
                pass

        # Wait for API response to arrive
        if not wait_for(state, "api_done", API_WAIT):

            # Small extra wait if API response is late
            time.sleep(2)

        # If API response received
        if state["api_done"] and state["api_body"]:

            # Extract schemes from API response
            batch = extract_from_v6(state["api_body"])

            # If schemes found
            if batch:
                return batch

            # API returned empty data — retry
            print(f" [empty response, retry {attempt}/{MAX_RETRIES}]", end="", flush=True)

        else:

            # API response never arrived
            print(f" [timeout, retry {attempt}/{MAX_RETRIES}]", end="", flush=True)

        # Exponential backoff delay before retry
        wait = RETRY_DELAY * (2 ** (attempt - 1))

        time.sleep(wait)

    # All retries failed
    return None


# Main scraping function
def scrape():

    # List of all collected schemes
    all_schemes  = []

    # List of pages that failed
    failed_pages = []

    # Start Playwright
    with sync_playwright() as p:

        # Launch Chromium browser in headless mode
        browser = p.chromium.launch(headless=True)

        # Create browser context (like a new user session)
        ctx = browser.new_context(

            # Set browser user agent (pretend to be Chrome browser)
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),

            # Browser viewport size
            viewport={"width": 1280, "height": 900},
        )

        # Open new browser page/tab
        bpage = ctx.new_page()

        # Shared state dictionary for communication
        state = {
            "target_offset": 0,
            "api_body":      None,
            "api_done":      False,
        }


        # Function to intercept network requests
        def handle_route(route: Route):

            # Get request URL
            url = route.request.url

            # Check if request is scheme API
            if re.search(r"api\.myscheme", url) and "schemes" in url:

                # Parse URL components
                parsed     = urlparse(url)

                # Extract query parameters
                qs         = parse_qs(parsed.query, keep_blank_values=True)

                # Replace "from" parameter with target offset
                qs["from"] = [str(state["target_offset"])]

                # Rebuild query string
                new_qs     = urlencode({k: v[0] for k, v in qs.items()})

                # Create new modified URL
                new_url    = urlunparse(parsed._replace(query=new_qs))

                # Continue request using modified URL
                route.continue_(url=new_url)

            else:

                # Continue normal requests unchanged
                route.continue_()


        # Function triggered when any response is received
        def on_response(resp):

            # Check if response is schemes API
            if re.search(r"api\.myscheme", resp.url) and "schemes" in resp.url:

                try:

                    # Save JSON response
                    state["api_body"] = resp.json()

                    # Mark API response as received
                    state["api_done"] = True

                except Exception:
                    pass


        # Enable request interception
        bpage.route("**/*", handle_route)

        # Listen for network responses
        bpage.on("response", on_response)


        # ── Page 1 ─────────────────────────────────────────

        print("[*] Loading page 1 ...")

        # Set offset for first page
        state["target_offset"] = 0
        state["api_body"]      = None
        state["api_done"]      = False

        try:

            # Open Gujarat schemes page
            bpage.goto(BASE_URL, wait_until="networkidle", timeout=NAV_TIMEOUT)

        except PlaywrightTimeout:

            # Retry navigation without waiting for network idle
            bpage.goto(BASE_URL, timeout=NAV_TIMEOUT)

        # Wait for API response
        if not wait_for(state, "api_done", API_WAIT):

            print("[!] No API response on page 1. Exiting.")

            browser.close()

            sys.exit(1)

        # Get total number of schemes from API
        total       = state["api_body"].get("data", {}).get("summary", {}).get("total", 0)

        # Calculate number of pages
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

        print(f"[+] Total schemes : {total}  ({total_pages} pages)\n")

        # Extract schemes from first page
        batch = extract_from_v6(state["api_body"])

        # Add to master list
        all_schemes.extend(batch)

        print(f"     Page  1 →  {len(batch):3d} schemes  (total: {len(all_schemes)})")


        # ── Pages 2..N ─────────────────────────────────────

        for page_no in range(2, total_pages + 1):

            # Calculate offset
            offset = (page_no - 1) * PAGE_SIZE

            print(f"     Page {page_no:2d} (offset={offset:3d}) → ", end="", flush=True)

            # Fetch schemes for that page
            batch = fetch_offset(bpage, state, offset, page_no)

            # If page failed
            if batch is None:

                print(f" ✗ FAILED after {MAX_RETRIES} retries — will retry at end.")

                failed_pages.append((page_no, offset))

            else:

                # Add schemes to list
                all_schemes.extend(batch)

                print(f" {len(batch):3d} schemes  (total: {len(all_schemes)})")

            # Delay before next page
            time.sleep(PAGE_DELAY)


        # ── Retry failed pages ─────────────────────────────

        if failed_pages:

            print(f"\n[*] Retrying {len(failed_pages)} failed page(s) ...")

            still_failed = []

            for page_no, offset in failed_pages:

                print(f"     Retry page {page_no:2d} (offset={offset:3d}) → ", end="", flush=True)

                # Wait before retry
                time.sleep(RETRY_DELAY)

                batch = fetch_offset(bpage, state, offset, page_no)

                if batch is None:

                    print(f" ✗ Still failed.")

                    still_failed.append((page_no, offset))

                else:

                    all_schemes.extend(batch)

                    print(f" {len(batch):3d} schemes  (total: {len(all_schemes)})")

            # Report pages that still failed
            if still_failed:

                print(f"\n[!] {len(still_failed)} page(s) could not be retrieved:")

                for pno, off in still_failed:

                    print(f"     Page {pno} (offset {off})")

        # Close browser
        browser.close()

    # Return collected schemes
    return all_schemes


# Function to save schemes into CSV file
def save_csv(schemes: list, path: str):

    # CSV column names — "state" added after "scheme_link"
    fields = ["scheme_name", "scheme_link", "state", "category", "description"]

    # Open file for writing
    with open(path, "w", newline="", encoding="utf-8-sig") as f:

        # Create CSV writer
        writer = csv.DictWriter(f, fieldnames=fields)

        # Write header row
        writer.writeheader()

        # Write scheme rows
        writer.writerows(schemes)

    print(f"\n✅  Saved {len(schemes)} schemes → {path}")


# Main program function
def main():

    print("=" * 60)
    print("  myscheme.gov.in — Gujarat Schemes Scraper")
    print("=" * 60 + "\n")

    # Run scraper
    schemes = scrape()

    # Remove duplicates
    seen, unique = set(), []

    for s in schemes:

        # Use scheme link or name as unique key
        key = s["scheme_link"] or s["scheme_name"]

        if key not in seen:

            seen.add(key)

            unique.append(s)

    print(f"\n[+] Unique schemes collected: {len(unique)} / 646")

    # Save results
    save_csv(unique, OUTPUT_FILE)

    print("\nDone! 🎉")


# Run main function when script executed directly
if __name__ == "__main__":
    main()