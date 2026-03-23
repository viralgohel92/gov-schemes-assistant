"""
scheduler.py
════════════════════════════════════════════════════════════════
Place this file in project ROOT (GOV-SCHEME-ASSISTANT/)

Run this file on your server instead of app.py.
It starts Flask AND runs both scheduled tasks together.

Usage:
    python scheduler.py

Schedule:
    02:00 AM  →  sync_schemes.py     (add new / delete old schemes)
    03:00 AM  →  rescrape_missing.py (fix schemes with missing details)
════════════════════════════════════════════════════════════════
"""

import schedule
import time
import threading
import logging

from scraper.sync_schemes import run_sync           # add/delete schemes
from scraper.rescrape_missing import main as fix_missing  # fix missing details

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("scheduler")

# ── Schedule both tasks ───────────────────────────────────

# Step 1: Sync new/deleted schemes at 2 AM
schedule.every().day.at("02:00").do(run_sync)
# schedule.every(10).minutes.do(run_sync)   # for testing: every 2 minutes

# Step 2: Fix missing details at 3 AM
# (runs AFTER sync so newly added schemes get their details fixed too)
schedule.every().day.at("03:00").do(fix_missing)
# schedule.every(2).minutes.do(fix_missing)   # for testing: every 2 minutes

# ── FOR LOCAL TESTING — uncomment these and comment above ─
# schedule.every(2).minutes.do(run_sync)
# schedule.every(4).minutes.do(fix_missing)

# ── Scheduler loop runs in background thread ──────────────
def run_scheduler():
    log.info("✅ Scheduler started!")
    log.info("   📅 sync_schemes     → runs daily at 02:00 AM")
    log.info("   📅 rescrape_missing → runs daily at 03:00 AM")
    while True:
        schedule.run_pending()   # checks if any task is due
        time.sleep(60)           # check every 60 seconds

# Start scheduler in background (won't block Flask)
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# ── Start Flask app ───────────────────────────────────────
from frontend.app import app   # app.py is in frontend/ folder

if __name__ == "__main__":
    log.info("🚀 Starting Yojana AI with auto-sync scheduler...")
    app.run(debug=False, port=5000)
