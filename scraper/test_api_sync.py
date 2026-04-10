import time
import re
import json
import logging
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api_test")

BASE_URL = "https://www.myscheme.gov.in/search/state/Gujarat"

def test_api_fetch():
    """
    Test script to verify direct API fetching to bypass Page 46 timeouts.
    Does NOT affect the database.
    """
    print("\n" + "="*50)
    print("🚀 Yojana AI — Hybrid API Recovery Test")
    print("="*50)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Using a context to handle shared state
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()
        
        captured = {"url": None, "headers": None, "found": False}

        def on_request(request):
            if "api.myscheme.gov.in" in request.url and "schemes" in request.url and not captured["found"]:
                captured["url"]     = request.url
                captured["headers"] = request.headers
                captured["found"]   = True
                log.info("✅ Captured API Session!")

        page.on("request", on_request)
        
        log.info("🌐 Loading Page 1 to capture authentication...")
        try:
            page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        except Exception as e:
            log.warning(f"⚠️ Initial load timed out, checking if API was captured anyway... ({e})")
            
        # Give it a few seconds to fire the API call
        for _ in range(5):
            if captured["found"]: break
            time.sleep(1)
            
        if not captured["found"]:
            log.error("❌ Failed to capture API headers. The website might be slow. Please try running again.")
            browser.close()
            return

        # 🎯 Specifically test Page 46 (which was failing in the UI)
        target_page = 46
        offset = (target_page - 1) * 10
        log.info(f"🎯 Specifically testing Page {target_page} (offset {offset}) via Direct API...")
        
        # Construct the API URL for Page 46
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        parsed = urlparse(captured["url"])
        qs = parse_qs(parsed.query)
        qs["from"] = [str(offset)]
        new_url = urlunparse(parsed._replace(query=urlencode({k: v[0] for k, v in qs.items()})))

        log.info(f"📡 Fetching: {new_url}")
        
        log.info(f"📡 Fetching Page {target_page} from inside browser context...")
        
        # We perform the fetch inside the page to ensure cookies and security tokens are perfect
        fetch_script = f"""
        async () => {{
            const response = await fetch("{new_url}", {{
                headers: {{
                    "x-api-key": "{captured['headers'].get('x-api-key', '')}"
                }}
            }});
            return {{
                status: response.status,
                data: await response.json(),
                text: response.status !== 200 ? await response.text() : ""
            }};
        }}
        """
        
        try:
            result = page.evaluate(fetch_script)
            
            if result["status"] == 200:
                data = result["data"]
                total = data.get("data", {}).get("summary", {}).get("total", 0)
                # Corrected path to the actual list of items
                schemes = data.get("data", {}).get("hits", {}).get("items", [])
                log.info(f"✅ SUCCESS! Bypassed 401 and UI issues.")
                log.info(f"📊 Live Total Schemes: {total}")
                log.info(f"📦 Schemes on Page {target_page}: {len(schemes)}")
                
                # Slicing the list (which is now correctly identified)
                for s in schemes[:3]:
                    name = s.get("fields", {}).get("schemeName", "No Name")
                    print(f"   - {name}")
            else:
                log.error(f"❌ Inner Fetch failed with status {result['status']}")
                log.error(f"Response: {str(result['data'])[:200]}")
        except Exception as e:
            log.error(f"❌ Execution error: {e}")

        browser.close()
    
    print("="*50 + "\n")

if __name__ == "__main__":
    test_api_fetch()
